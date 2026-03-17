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


def compute_norm_stats(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(dim=0, keepdim=True)
    std = flat.std(dim=0, keepdim=True).clamp(min=1e-6)
    return mean, std


def normalize_features(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (features - mean.view(1, 1, -1)) / std.view(1, 1, -1)


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
    return metrics


def listwise_loss(logits: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
    hit_mask = target_positions >= 0
    if not hit_mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[hit_mask], target_positions[hit_mask])


def select_features(split: dict, mode: str) -> torch.Tensor:
    feats = split['candidate_features'].float().clone()
    if mode == 'typed':
        return feats[:, :, :TYPED_END]
    if mode == 'coarse_to_fine':
        return feats[:, :, :COARSE_END]
    if mode == 'adaptive':
        adaptive = feats[:, :, :ADAPTIVE_END]
        adaptive[:, :, 24] *= adaptive[:, :, 35]  # recent overlap by recent weight
        adaptive[:, :, 25] *= adaptive[:, :, 35]
        adaptive[:, :, 22] *= adaptive[:, :, 36]  # stable overlap by stable weight
        adaptive[:, :, 23] *= adaptive[:, :, 36]
        adaptive[:, :, 26] *= (1.0 + adaptive[:, :, 37])  # emphasize dislike conflict when user has dislikes
        adaptive[:, :, 27] *= (1.0 + adaptive[:, :, 38])  # affinity overlap by affinity strength
        return adaptive
    return feats[:, :, :BASE_DIM]


def train_variant(train: dict, val: dict, cfg: dict, mode: str, device: torch.device):
    train_cfg = cfg['training']
    train_x = select_features(train, mode)
    val_x = select_features(val, mode)
    mean, std = compute_norm_stats(train_x)
    x_train = normalize_features(train_x, mean, std).to(device)
    x_val = normalize_features(val_x, mean, std).to(device)
    y_train = train['target_positions'].to(device)
    y_val = val['target_positions'].to(device)

    model = CandidateMLP(x_train.shape[-1], int(train_cfg['hidden_dim']), float(train_cfg['dropout'])).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(train_cfg['lr']), weight_decay=float(train_cfg['weight_decay']))
    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    best_val = -1.0
    patience = 0
    history = []

    for epoch in range(int(train_cfg['num_epochs'])):
        model.train()
        total_loss = 0.0
        for start, end in batch_iter(x_train.shape[0], int(train_cfg['batch_size'])):
            logits = model(x_train[start:end])
            loss = listwise_loss(logits, y_train[start:end])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += float(loss.item()) * (end - start)
        model.eval()
        with torch.no_grad():
            val_scores = model(x_val)
            val_metrics = ranking_metrics(val_scores, y_val, [10, 1])
            val_score = val_metrics['NDCG@10']
            history.append({'epoch': epoch + 1, 'train_loss': total_loss / x_train.shape[0], 'val_ndcg10': val_score, 'val_hr1': val_metrics['HR@1']})
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
    control_feature_slot = int(variant_cfg.get('control_feature_slot', 5))
    train = load_split(data_dir, 'train')
    val = load_split(data_dir, 'val')
    test = load_split(data_dir, 'test')
    manifest = load_json(os.path.join(data_dir, 'data_manifest.json'))
    item_ids = load_json(os.path.join(script_dir, cfg['paths']['candidate17_data_dir'], 'item_ids.json'))
    title_rows = load_json(os.path.join(script_dir, cfg['paths']['item_titles_path']))
    item_title_lookup = {row['item_id']: row['condensed_title'] for row in title_rows}
    device, device_probe = select_device(cfg)
    if device.type != "cuda" and not args.allow_cpu_fallback:
        raise RuntimeError(f"CUDA device selection failed; refusing CPU fallback for this experiment. probe={device_probe}")

    extra: Dict[str, object] = {
        'variant_mode': mode,
        'control_feature_slot': control_feature_slot if mode == 'control' else None,
        'device_probe': device_probe,
        'runtime_device': str(device),
        'cpu_fallback_used': device.type != 'cuda',
    }
    if mode == 'control':
        test_scores = test['candidate_features'][:, :, control_feature_slot].clone()
    else:
        model, history, mean, std = train_variant(train, val, cfg, mode, device)
        x_test = normalize_features(select_features(test, mode), mean, std).to(device)
        with torch.no_grad():
            test_scores = model(x_test).cpu()
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
