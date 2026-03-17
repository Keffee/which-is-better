#!/usr/bin/env python3

import argparse
import os
import subprocess
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.common.pipeline_utils import dump_json, load_json

BASE_DIM = 22
TYPED_END = 30
COARSE_END = 35
ADAPTIVE_END = 39
POSITIVE_SLOTS = [22, 23, 24, 25, 27, 28, 35, 36, 38]
CONFLICT_SLOTS = [26, 29, 37]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate structured preference memory graph variants.')
    parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), 'structured_memory_graph_config.yaml'))
    parser.add_argument('--variant', type=str, required=True)
    parser.add_argument('--output_prefix', type=str, required=True)
    parser.add_argument('--allow_cpu_fallback', action='store_true')
    return parser.parse_args()


def parse_gpu_indices(raw_value: str) -> List[int]:
    gpu_indices: List[int] = []
    for token in raw_value.split(","):
        token = token.strip()
        if token:
            gpu_indices.append(int(token))
    return gpu_indices


def select_device(cfg: dict) -> Tuple[torch.device, Dict[str, object]]:
    runtime_cfg = cfg.get('runtime', {})
    allowed_gpus = parse_gpu_indices(str(runtime_cfg.get('allowed_gpus', '')))
    visible_raw = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    visible_physical = parse_gpu_indices(visible_raw) if visible_raw else []
    visible_map = {physical_idx: local_idx for local_idx, physical_idx in enumerate(visible_physical)}

    probe: Dict[str, object] = {
        'allowed_gpus': allowed_gpus,
        'visible_cuda_devices': visible_physical,
        'gpu_candidates': [],
        'cuda_probe_errors': [],
    }

    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        candidates = []
        for line in result.strip().splitlines():
            idx_s, mem_s, util_s = [part.strip() for part in line.split(",")]
            physical_idx = int(idx_s)
            if allowed_gpus and physical_idx not in allowed_gpus:
                continue
            if visible_map and physical_idx not in visible_map:
                continue
            candidates.append(
                {
                    'physical_index': physical_idx,
                    'torch_index': visible_map.get(physical_idx, physical_idx),
                    'memory_used_mb': int(mem_s),
                    'utilization_gpu': int(util_s),
                }
            )
        candidates.sort(key=lambda row: (row['memory_used_mb'], row['utilization_gpu'], row['physical_index']))
        probe['gpu_candidates'] = candidates
    except Exception as exc:
        probe['cuda_probe_errors'].append({'stage': 'nvidia-smi', 'error': str(exc)})
        candidates = []

    for candidate in candidates:
        torch_index = int(candidate['torch_index'])
        try:
            torch.cuda.set_device(torch_index)
            _ = torch.empty(1, device=f"cuda:{torch_index}")
            probe['selected_gpu_index'] = int(candidate['physical_index'])
            probe['selected_torch_index'] = torch_index
            return torch.device(f"cuda:{torch_index}"), probe
        except Exception as exc:
            probe['cuda_probe_errors'].append(
                {
                    'stage': 'torch_cuda_alloc',
                    'physical_index': int(candidate['physical_index']),
                    'torch_index': torch_index,
                    'error': str(exc),
                }
            )

    return torch.device("cpu"), probe


def load_split(data_dir: str, split: str) -> dict:
    return torch.load(os.path.join(data_dir, f'{split}_memory_graph.pt'), map_location='cpu', weights_only=False)


def batch_iter(num_rows: int, batch_size: int):
    for start in range(0, num_rows, batch_size):
        yield start, min(num_rows, start + batch_size)


class CandidateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ConflictCalibratedScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.base_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        aux_hidden = max(16, hidden_dim // 4)
        self.positive_head = nn.Sequential(
            nn.Linear(len(POSITIVE_SLOTS), aux_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(aux_hidden, 1),
        )
        self.conflict_head = nn.Sequential(
            nn.Linear(len(CONFLICT_SLOTS), aux_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(aux_hidden, 1),
        )
        self.final = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_hidden = self.base_net(x)
        base_score = self.final(base_hidden)
        positive_bonus = self.positive_head(x[:, :, POSITIVE_SLOTS])
        conflict_penalty = F.softplus(self.conflict_head(x[:, :, CONFLICT_SLOTS]))
        return (base_score + 0.25 * positive_bonus - 0.5 * conflict_penalty).squeeze(-1)


def compute_norm_stats(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(dim=0, keepdim=True)
    std = flat.std(dim=0, keepdim=True).clamp(min=1e-6)
    return mean, std


def normalize_features(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (features - mean.view(1, 1, -1)) / std.view(1, 1, -1)


def top1_score_margin(scores: torch.Tensor) -> float:
    if scores.shape[1] < 2:
        return 0.0
    top2_scores = torch.topk(scores, k=2, dim=1).values
    return float((top2_scores[:, 0] - top2_scores[:, 1]).mean().item())


def average_target_vs_best_negative_margin(scores: torch.Tensor, target_positions: torch.Tensor) -> float:
    hit_mask = target_positions >= 0
    if not hit_mask.any():
        return 0.0
    scores_hit = scores[hit_mask]
    targets_hit = target_positions[hit_mask]
    target_scores = scores_hit.gather(1, targets_hit.unsqueeze(1))
    negatives = scores_hit.clone()
    negatives.scatter_(1, targets_hit.unsqueeze(1), float('-inf'))
    best_negative = negatives.max(dim=1, keepdim=True).values
    return float((target_scores - best_negative).mean().item())


def ranking_metrics(candidate_scores: torch.Tensor, target_positions: torch.Tensor, k_values: List[int]) -> Dict[str, float]:
    total = int(target_positions.shape[0])
    order = torch.argsort(candidate_scores, dim=1, descending=True).cpu()
    targets_cpu = target_positions.cpu()
    metrics: Dict[str, float] = {}
    for k in k_values:
        hits = 0.0
        ndcg = 0.0
        for i in range(total):
            target_pos = int(targets_cpu[i].item())
            if target_pos < 0:
                continue
            row = order[i, :k].tolist()
            if target_pos in row:
                hits += 1.0
                rank = row.index(target_pos) + 1
                ndcg += 1.0 / np.log2(rank + 1)
        metrics[f'HR@{k}'] = hits / total
        metrics[f'NDCG@{k}'] = ndcg / total
    return metrics


def evaluate_scores(split: dict, scores: torch.Tensor) -> Dict[str, float]:
    metrics = ranking_metrics(scores, split['target_positions'], [1, 5, 10, 20])
    order = torch.argsort(scores, dim=1, descending=True).cpu()
    top1_slots = order[:, 0]
    mem = split['candidate_features']
    metrics['stage1_recall@20'] = float(split['stage1_metrics']['Recall@20'])
    metrics['candidate_hit_rate'] = float(split['candidate_hit_rate'])
    metrics['avg_top1_positive_support_ratio'] = float(mem[torch.arange(mem.shape[0]), top1_slots, 28].mean().item())
    metrics['avg_top1_negative_conflict_ratio'] = float(mem[torch.arange(mem.shape[0]), top1_slots, 29].mean().item())
    metrics['avg_top1_affinity_overlap'] = float(mem[torch.arange(mem.shape[0]), top1_slots, 27].mean().item())
    metrics['avg_top1_score_margin'] = top1_score_margin(scores)
    metrics['avg_target_vs_best_negative_margin'] = average_target_vs_best_negative_margin(scores, split['target_positions'])
    return metrics


def compute_training_loss(
    logits: torch.Tensor,
    target_positions: torch.Tensor,
    candidate_features: torch.Tensor,
    topk_weight: float,
    topk_margin: float,
    topk_count: int,
    negative_weight_mode: str,
    support_weight: float,
    conflict_weight: float,
    affinity_weight: float,
    min_neg_weight: float,
    max_neg_weight: float,
    skip_threshold: float,
    margin_support_weight: float,
    margin_conflict_weight: float,
    certainty_temperature: float,
    sample_gate_margin: float,
    sample_gate_temperature: float,
    sample_gate_mode: str,
    sample_gate_conflict_floor: float,
    sample_gate_support_ceiling: float,
    sample_gate_affinity_ceiling: float,
    sample_gate_reliability_floor: float,
    negative_selection_mode: str,
    selection_reliability_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hit_mask = target_positions >= 0
    if not hit_mask.any():
        zero = logits.sum() * 0.0
        return zero, zero.detach(), zero.detach()

    logits_hit = logits[hit_mask]
    targets_hit = target_positions[hit_mask]
    features_hit = candidate_features[hit_mask]
    base_loss = F.cross_entropy(logits_hit, targets_hit)
    margin_penalty = logits_hit.new_zeros(())
    if topk_weight > 0.0 and logits_hit.shape[1] > 1:
        target_scores = logits_hit.gather(1, targets_hit.unsqueeze(1))
        negatives = logits_hit.clone()
        negatives.scatter_(1, targets_hit.unsqueeze(1), float('-inf'))
        k = max(1, min(topk_count, negatives.shape[1] - 1))
        selection_scores = negatives
        if negative_selection_mode == 'reliability_adjusted':
            all_affinity = features_hit[:, :, 27].clamp(min=0.0, max=1.0)
            all_support = features_hit[:, :, 28].clamp(min=0.0, max=1.0)
            all_conflict = features_hit[:, :, 29].clamp(min=0.0, max=1.0)
            reliability = all_conflict - support_weight * all_support - affinity_weight * all_affinity
            selection_scores = negatives + selection_reliability_weight * reliability
            selection_scores.scatter_(1, targets_hit.unsqueeze(1), float('-inf'))
        topk_indices = torch.topk(selection_scores, k=k, dim=1).indices
        topk_negatives = negatives.gather(1, topk_indices)
        discounts = torch.tensor(
            [1.0 / np.log2(rank + 2.0) for rank in range(k)],
            device=logits_hit.device,
            dtype=logits_hit.dtype,
        ).view(1, -1)
        pair_margins = torch.full_like(topk_negatives, fill_value=topk_margin)
        weights = discounts.expand_as(topk_negatives)

        gathered = features_hit.gather(
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, features_hit.shape[-1]),
        )
        neg_affinity = gathered[:, :, 27].clamp(min=0.0, max=1.0)
        neg_support = gathered[:, :, 28].clamp(min=0.0, max=1.0)
        neg_conflict = gathered[:, :, 29].clamp(min=0.0, max=1.0)

        if negative_weight_mode == 'conflict_weighted':
            certainty = (1.0 + conflict_weight * neg_conflict) / (
                1.0 + support_weight * neg_support + affinity_weight * neg_affinity
            )
            weights = weights * certainty.clamp(min=min_neg_weight, max=max_neg_weight)
        elif negative_weight_mode == 'false_negative_skip':
            keep_score = neg_conflict - support_weight * neg_support - affinity_weight * neg_affinity
            keep_mask = (keep_score >= skip_threshold).to(weights.dtype)
            if keep_mask.shape[1] > 0:
                fallback = torch.zeros_like(keep_mask)
                fallback[:, 0] = 1.0
                zero_rows = keep_mask.sum(dim=1, keepdim=True) == 0
                keep_mask = torch.where(zero_rows, fallback, keep_mask)
            weights = weights * keep_mask
        elif negative_weight_mode == 'adaptive_margin':
            pair_margins = pair_margins + margin_conflict_weight * neg_conflict - margin_support_weight * neg_support
        elif negative_weight_mode == 'certainty_topk':
            certainty_logits = certainty_temperature * (
                neg_conflict - support_weight * neg_support - affinity_weight * neg_affinity
            )
            certainty = torch.sigmoid(certainty_logits)
            weights = weights * certainty.clamp(min=min_neg_weight, max=max_neg_weight)
        elif negative_weight_mode == 'corrected_weighted':
            certainty_logits = certainty_temperature * (
                neg_conflict - support_weight * neg_support - affinity_weight * neg_affinity
            )
            certainty = torch.sigmoid(certainty_logits)
            weights = weights * certainty.clamp(min=min_neg_weight, max=max_neg_weight)
            pair_margins = pair_margins + margin_conflict_weight * certainty - margin_support_weight * (1.0 - certainty)

        pair_penalty = F.softplus(pair_margins - (target_scores - topk_negatives))
        if sample_gate_temperature > 0.0:
            hardest_margin = target_scores - topk_negatives[:, :1]
            sample_gate = torch.sigmoid(sample_gate_temperature * (sample_gate_margin - hardest_margin))
            hardest_conflict = neg_conflict[:, :1]
            hardest_support = neg_support[:, :1]
            hardest_affinity = neg_affinity[:, :1]
            hardest_reliability = hardest_conflict - support_weight * hardest_support - affinity_weight * hardest_affinity
            if sample_gate_mode == 'reliability_hard_case':
                sample_gate = sample_gate * torch.sigmoid(
                    sample_gate_temperature * (hardest_reliability - sample_gate_reliability_floor)
                )
            elif sample_gate_mode == 'conflict_first':
                sample_gate = sample_gate * torch.sigmoid(
                    sample_gate_temperature * (hardest_conflict - sample_gate_conflict_floor)
                ) * torch.sigmoid(
                    sample_gate_temperature * (sample_gate_support_ceiling - hardest_support)
                )
            elif sample_gate_mode == 'support_rescue':
                sample_gate = sample_gate * torch.sigmoid(
                    sample_gate_temperature * (sample_gate_support_ceiling - hardest_support)
                )
            elif sample_gate_mode == 'affinity_veto':
                sample_gate = sample_gate * torch.sigmoid(
                    sample_gate_temperature * (sample_gate_affinity_ceiling - hardest_affinity)
                )
            pair_penalty = pair_penalty * sample_gate
        denom = weights.sum(dim=1).clamp(min=1e-6)
        margin_penalty = ((pair_penalty * weights).sum(dim=1) / denom).mean()
    total_loss = base_loss + topk_weight * margin_penalty
    return total_loss, base_loss.detach(), margin_penalty.detach()


def select_features(split: dict, mode: str) -> torch.Tensor:
    feats = split['candidate_features'].float().clone()
    if mode == 'typed':
        return feats[:, :, :TYPED_END]
    if mode == 'coarse_to_fine':
        return feats[:, :, :COARSE_END]
    if mode in ('adaptive', 'adaptive_penalty', 'conflict_calibrated'):
        adaptive = feats[:, :, :ADAPTIVE_END]
        adaptive[:, :, 24] *= adaptive[:, :, 35]  # recent overlap by recent weight
        adaptive[:, :, 25] *= adaptive[:, :, 35]
        adaptive[:, :, 22] *= adaptive[:, :, 36]  # stable overlap by stable weight
        adaptive[:, :, 23] *= adaptive[:, :, 36]
        adaptive[:, :, 26] *= (1.0 + adaptive[:, :, 37])  # emphasize dislike conflict when user has dislikes
        adaptive[:, :, 27] *= (1.0 + adaptive[:, :, 38])  # affinity overlap by affinity strength
        return adaptive
    return feats[:, :, :BASE_DIM]


def conflict_penalty_signal(split: dict) -> torch.Tensor:
    feats = split['candidate_features'].float()
    raw_conflict = feats[:, :, 26].clamp(min=0.0)
    conflict_ratio = feats[:, :, 29].clamp(min=0.0, max=1.0)
    dislike_strength = feats[:, :, 37].clamp(min=0.0)
    return raw_conflict + conflict_ratio * (1.0 + dislike_strength)


def merged_training_config(cfg: dict, variant_cfg: dict) -> dict:
    train_cfg = dict(cfg['training'])
    train_cfg.update(variant_cfg.get('training_overrides', {}))
    return train_cfg


def train_variant(train: dict, val: dict, train_cfg: dict, mode: str, device: torch.device):
    train_x = select_features(train, mode)
    val_x = select_features(val, mode)
    train_candidate_features = train['candidate_features'].float()
    mean, std = compute_norm_stats(train_x)
    x_train = normalize_features(train_x, mean, std).to(device)
    x_val = normalize_features(val_x, mean, std).to(device)
    y_train = train['target_positions'].to(device)
    y_val = val['target_positions'].to(device)

    if mode == 'conflict_calibrated':
        model = ConflictCalibratedScorer(x_train.shape[-1], int(train_cfg['hidden_dim']), float(train_cfg['dropout'])).to(device)
    else:
        model = CandidateMLP(x_train.shape[-1], int(train_cfg['hidden_dim']), float(train_cfg['dropout'])).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(train_cfg['lr']), weight_decay=float(train_cfg['weight_decay']))
    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    best_val = -1.0
    patience = 0
    history = []

    for epoch in range(int(train_cfg['num_epochs'])):
        model.train()
        total_loss = 0.0
        total_base_loss = 0.0
        total_margin_penalty = 0.0
        for start, end in batch_iter(x_train.shape[0], int(train_cfg['batch_size'])):
            logits = model(x_train[start:end])
            loss, base_loss, margin_penalty = compute_training_loss(
                logits,
                y_train[start:end],
                train_candidate_features[start:end].to(device),
                topk_weight=float(train_cfg.get('topk_weight', 0.0)),
                topk_margin=float(train_cfg.get('topk_margin', 0.0)),
                topk_count=int(train_cfg.get('topk_count', 3)),
                negative_weight_mode=str(train_cfg.get('negative_weight_mode', 'none')),
                support_weight=float(train_cfg.get('support_weight', 1.0)),
                conflict_weight=float(train_cfg.get('conflict_weight', 1.0)),
                affinity_weight=float(train_cfg.get('affinity_weight', 1.0)),
                min_neg_weight=float(train_cfg.get('min_neg_weight', 0.05)),
                max_neg_weight=float(train_cfg.get('max_neg_weight', 2.0)),
                skip_threshold=float(train_cfg.get('skip_threshold', 0.0)),
                margin_support_weight=float(train_cfg.get('margin_support_weight', 0.0)),
                margin_conflict_weight=float(train_cfg.get('margin_conflict_weight', 0.0)),
                certainty_temperature=float(train_cfg.get('certainty_temperature', 4.0)),
                sample_gate_margin=float(train_cfg.get('sample_gate_margin', 0.0)),
                sample_gate_temperature=float(train_cfg.get('sample_gate_temperature', 0.0)),
                sample_gate_mode=str(train_cfg.get('sample_gate_mode', 'score_only')),
                sample_gate_conflict_floor=float(train_cfg.get('sample_gate_conflict_floor', 0.0)),
                sample_gate_support_ceiling=float(train_cfg.get('sample_gate_support_ceiling', 1.0)),
                sample_gate_affinity_ceiling=float(train_cfg.get('sample_gate_affinity_ceiling', 1.0)),
                sample_gate_reliability_floor=float(train_cfg.get('sample_gate_reliability_floor', 0.0)),
                negative_selection_mode=str(train_cfg.get('negative_selection_mode', 'score_only')),
                selection_reliability_weight=float(train_cfg.get('selection_reliability_weight', 0.0)),
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += float(loss.item()) * (end - start)
            total_base_loss += float(base_loss.item()) * (end - start)
            total_margin_penalty += float(margin_penalty.item()) * (end - start)
        model.eval()
        with torch.no_grad():
            val_scores = model(x_val)
            val_metrics = ranking_metrics(val_scores, y_val, [10, 1])
            val_score = float(val_metrics['NDCG@10'])
            history.append(
                {
                    'epoch': epoch + 1,
                    'train_loss': total_loss / x_train.shape[0],
                    'train_base_loss': total_base_loss / x_train.shape[0],
                    'train_margin_penalty': total_margin_penalty / x_train.shape[0],
                    'val_ndcg10': val_score,
                    'val_hr1': float(val_metrics['HR@1']),
                    'val_top1_score_margin': top1_score_margin(val_scores),
                    'val_target_vs_best_negative_margin': average_target_vs_best_negative_margin(val_scores, y_val),
                }
            )
            if val_score > best_val:
                best_val = val_score
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= int(train_cfg['early_stop_patience']):
                    break
    model.load_state_dict(best_state)
    return model, history, mean, std


def tune_penalty_alpha(scores: torch.Tensor, split: dict) -> Tuple[float, Dict[str, float]]:
    signal = conflict_penalty_signal(split)
    best_alpha = 0.0
    best_metrics = ranking_metrics(scores, split['target_positions'], [10, 1])
    best_key = (best_metrics['NDCG@10'], best_metrics['HR@1'])
    for alpha in [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12]:
        adjusted = scores - alpha * signal
        metrics = ranking_metrics(adjusted, split['target_positions'], [10, 1])
        key = (metrics['NDCG@10'], metrics['HR@1'])
        if key > best_key:
            best_key = key
            best_alpha = alpha
            best_metrics = metrics
    return best_alpha, best_metrics


def build_sample_traces(split: dict, scores: torch.Tensor, item_ids: List[str], item_titles: Dict[str, str], limit: int) -> List[dict]:
    traces = []
    order = torch.argsort(scores, dim=1, descending=True).cpu()
    for i in range(min(limit, scores.shape[0])):
        ranked = []
        for pos in order[i, :10].tolist():
            internal_idx = int(split['candidate_indices'][i, pos].item())
            item_id = item_ids[internal_idx]
            ranked.append({
                'rank': len(ranked) + 1,
                'candidate_slot': pos,
                'item_id': item_id,
                'title': item_titles[item_id],
                'score': float(scores[i, pos].item()),
                'positive_support_ratio': float(split['candidate_features'][i, pos, 28].item()),
                'negative_conflict_ratio': float(split['candidate_features'][i, pos, 29].item()),
                'affinity_overlap': float(split['candidate_features'][i, pos, 27].item()),
            })
        traces.append({
            'user_id': split['user_ids'][i],
            'target_item_id': split['target_item_ids'][i],
            'target_position_in_candidates': int(split['target_positions'][i].item()),
            'top10': ranked,
            'memory_stats': split['memory_stats'][i],
        })
    return traces


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    data_dir = os.path.join(script_dir, cfg['paths']['data_dir'])
    result_dir = os.path.join(script_dir, cfg['paths']['results_dir'])
    os.makedirs(result_dir, exist_ok=True)

    variant_cfg = next(v for v in cfg['variants'] if v['name'] == args.variant)
    mode = variant_cfg['mode']
    train_cfg = merged_training_config(cfg, variant_cfg)
    train = load_split(data_dir, 'train')
    val = load_split(data_dir, 'val')
    test = load_split(data_dir, 'test')
    item_ids = load_json(os.path.join(script_dir, cfg['paths']['candidate17_data_dir'], 'item_ids.json'))
    title_rows = load_json(os.path.join(script_dir, cfg['paths']['item_titles_path']))
    item_title_lookup = {row['item_id']: row['condensed_title'] for row in title_rows}
    device, device_probe = select_device(cfg)
    if device.type != "cuda" and not args.allow_cpu_fallback:
        raise RuntimeError(f"CUDA device selection failed; refusing CPU fallback for this experiment. probe={device_probe}")

    extra: Dict[str, object] = {
        'variant_mode': mode,
        'training_config': train_cfg,
        'device_probe': device_probe,
        'runtime_device': str(device),
        'cpu_fallback_used': device.type != 'cuda',
    }
    if mode == 'control':
        test_scores = test['candidate_features'][:, :, 5].clone()
    else:
        model, history, mean, std = train_variant(train, val, train_cfg, mode, device)
        x_val = normalize_features(select_features(val, mode), mean, std).to(device)
        x_test = normalize_features(select_features(test, mode), mean, std).to(device)
        with torch.no_grad():
            val_scores = model(x_val).cpu()
            test_scores = model(x_test).cpu()
        if mode == 'adaptive_penalty':
            best_alpha, tuned_val_metrics = tune_penalty_alpha(val_scores, val)
            test_scores = test_scores - best_alpha * conflict_penalty_signal(test)
            extra['selected_penalty_alpha'] = best_alpha
            extra['penalty_tuned_val_metrics'] = tuned_val_metrics
        extra['val_metrics_at_best_checkpoint'] = evaluate_scores(val, val_scores)
        extra['training_history'] = history
        extra['feature_mean'] = mean.tolist()
        extra['feature_std'] = std.tolist()

    metrics = evaluate_scores(test, test_scores)
    metrics['variant'] = args.variant
    metrics['output_prefix'] = args.output_prefix
    metrics.update(extra)
    traces = build_sample_traces(test, test_scores, item_ids, item_title_lookup, int(cfg['data']['sample_trace_limit']))
    dump_json(os.path.join(result_dir, f'{args.output_prefix}_metrics.json'), metrics)
    dump_json(os.path.join(result_dir, f'{args.output_prefix}_sample_predictions.json'), traces)
    print(metrics)


if __name__ == '__main__':
    main()
