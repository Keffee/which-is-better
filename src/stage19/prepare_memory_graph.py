#!/usr/bin/env python3

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List, Sequence

import torch
import yaml

from src.common.pipeline_utils import dump_json, load_item_titles, load_jsonl, normalize_title

STOPWORDS = {
    'and', 'the', 'for', 'with', 'from', 'into', 'your', 'you', 'that', 'this', 'are', 'was', 'have',
    'has', 'had', 'use', 'using', 'used', 'to', 'of', 'in', 'on', 'a', 'an', 'is', 'it', 'at', 'by',
    'as', 'be', 'or', 'if', 'but', 'not', 'so', 'than', 'then', 'too', 'very', 'can', 'will'
}

BASE_DIM = 22
TYPED_DIM = 8
COARSE_EXTRA_DIM = 5
ADAPTIVE_EXTRA_DIM = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare structured preference memory graph features for 19_.')
    parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), 'structured_memory_graph_config.yaml'))
    return parser.parse_args()


def load_split_candidates(data_dir: str, split: str) -> dict:
    return torch.load(os.path.join(data_dir, f'{split}_candidates.pt'), map_location='cpu', weights_only=False)


def tokenize(text: str) -> List[str]:
    text = normalize_title(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [tok for tok in text.split() if len(tok) >= 3 and tok not in STOPWORDS]


def token_overlap(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return inter / union


def evidence_overlap(candidate_tokens: List[str], nodes: List[dict]) -> List[float]:
    return [token_overlap(candidate_tokens, node['tokens']) for node in nodes]


def align_rows(raw_rows: List[dict], user_ids: List[str], target_item_ids: List[str]) -> List[dict]:
    lookup = {(row['user_id'], row['target_item_id']): row for row in raw_rows}
    aligned = []
    for user_id, target_item_id in zip(user_ids, target_item_ids):
        key = (user_id, target_item_id)
        if key not in lookup:
            raise KeyError(f'missing raw row for {key}')
        aligned.append(lookup[key])
    return aligned


def build_memory_graph(row: dict, recent_window: int, stable_positive_threshold: int, dislike_threshold: int, affinity_min_repeat: int) -> Dict[str, object]:
    titles = row.get('history_titles', [])
    ratings = row.get('history_ratings', [])
    reviews = row.get('history_reviews', [])
    total = len(titles)
    positive_nodes = []
    stable_nodes = []
    recent_nodes = []
    dislike_nodes = []
    affinity_counter: Counter = Counter()

    for idx, (title, rating, review) in enumerate(zip(titles, ratings, reviews)):
        rating = int(rating)
        title_norm = normalize_title(title)
        tokens = tokenize(title_norm + ' ' + review)
        recency_rank = idx + 1
        node = {
            'title': title_norm,
            'tokens': tokens,
            'rating': rating,
            'recency': recency_rank / max(total, 1),
        }
        if rating >= stable_positive_threshold:
            positive_nodes.append(node)
            affinity_counter.update(set(tokenize(title_norm)))
            if idx >= max(0, total - recent_window):
                recent_nodes.append(node)
            else:
                stable_nodes.append(node)
        elif rating <= dislike_threshold:
            dislike_nodes.append(node)

    affinity_tokens = [tok for tok, count in affinity_counter.items() if count >= affinity_min_repeat]
    return {
        'positive_nodes': positive_nodes,
        'stable_nodes': stable_nodes,
        'recent_nodes': recent_nodes,
        'dislike_nodes': dislike_nodes,
        'affinity_tokens': affinity_tokens,
        'stats': {
            'positive_count': len(positive_nodes),
            'stable_count': len(stable_nodes),
            'recent_count': len(recent_nodes),
            'dislike_count': len(dislike_nodes),
            'affinity_count': len(affinity_tokens),
            'history_len': total,
        },
    }


def candidate_memory_features(candidate_tokens: List[str], graph: Dict[str, object]) -> List[float]:
    stable_nodes = graph['stable_nodes']
    recent_nodes = graph['recent_nodes']
    dislike_nodes = graph['dislike_nodes']
    positive_nodes = graph['positive_nodes']
    affinity_tokens = graph['affinity_tokens']
    stats = graph['stats']

    stable_overlaps = evidence_overlap(candidate_tokens, stable_nodes)
    recent_overlaps = evidence_overlap(candidate_tokens, recent_nodes)
    dislike_overlaps = evidence_overlap(candidate_tokens, dislike_nodes)
    positive_overlaps = evidence_overlap(candidate_tokens, positive_nodes)

    stable_max = max(stable_overlaps) if stable_overlaps else 0.0
    stable_avg = float(sum(stable_overlaps) / len(stable_overlaps)) if stable_overlaps else 0.0
    recent_max = max(recent_overlaps) if recent_overlaps else 0.0
    recent_avg = float(sum(recent_overlaps) / len(recent_overlaps)) if recent_overlaps else 0.0
    dislike_max = max(dislike_overlaps) if dislike_overlaps else 0.0
    affinity_overlap = token_overlap(candidate_tokens, affinity_tokens)
    positive_support_ratio = float(sum(x > 0 for x in positive_overlaps)) / max(stats['positive_count'], 1)
    negative_conflict_ratio = float(sum(x > 0 for x in dislike_overlaps)) / max(stats['dislike_count'], 1)

    top_positive = sorted(positive_overlaps, reverse=True)[:2]
    evidence_top1 = top_positive[0] if len(top_positive) >= 1 else 0.0
    evidence_top2 = top_positive[1] if len(top_positive) >= 2 else 0.0
    recent_recency_support = 0.0
    for node, overlap in zip(recent_nodes, recent_overlaps):
        recent_recency_support = max(recent_recency_support, overlap * float(node['recency']))
    stable_recent_gap = stable_max - recent_max
    positive_negative_gap = positive_support_ratio - negative_conflict_ratio

    total_pos = max(stats['positive_count'], 1)
    weight_recent = stats['recent_count'] / total_pos
    weight_stable = stats['stable_count'] / total_pos
    weight_dislike = stats['dislike_count'] / max(stats['history_len'], 1)
    weight_affinity = min(stats['affinity_count'], 8) / 8.0

    return [
        stable_max,
        stable_avg,
        recent_max,
        recent_avg,
        dislike_max,
        affinity_overlap,
        positive_support_ratio,
        negative_conflict_ratio,
        evidence_top1,
        evidence_top2,
        recent_recency_support,
        stable_recent_gap,
        positive_negative_gap,
        weight_recent,
        weight_stable,
        weight_dislike,
        weight_affinity,
    ]


def process_split(split: str, raw_rows: List[dict], candidate_payload: dict, item_ids: List[str], item_title_lookup: Dict[str, str], cfg: dict, out_dir: str):
    data_cfg = cfg['data']
    recent_window = int(data_cfg['recent_window'])
    stable_positive_threshold = int(data_cfg['stable_positive_threshold'])
    dislike_threshold = int(data_cfg['dislike_threshold'])
    affinity_min_repeat = int(data_cfg['affinity_min_repeat'])

    aligned_rows = align_rows(raw_rows, candidate_payload['user_ids'], candidate_payload['target_item_ids'])
    num_rows, top_k, _ = candidate_payload['candidate_features'].shape
    memory_features = torch.zeros((num_rows, top_k, TYPED_DIM + COARSE_EXTRA_DIM + ADAPTIVE_EXTRA_DIM), dtype=torch.float32)
    memory_stats = []

    for row_idx, row in enumerate(aligned_rows):
        graph = build_memory_graph(row, recent_window, stable_positive_threshold, dislike_threshold, affinity_min_repeat)
        stats = graph['stats']
        memory_stats.append(stats)
        for cand_slot in range(top_k):
            item_internal_idx = int(candidate_payload['candidate_indices'][row_idx, cand_slot].item())
            item_id = item_ids[item_internal_idx]
            title = item_title_lookup[item_id]
            feats = candidate_memory_features(tokenize(title), graph)
            memory_features[row_idx, cand_slot] = torch.tensor(feats, dtype=torch.float32)

    combined = torch.cat([candidate_payload['candidate_features'].float(), memory_features], dim=2)
    feature_names = list(candidate_payload['feature_names']) + [
        'stable_overlap_max',
        'stable_overlap_avg',
        'recent_overlap_max',
        'recent_overlap_avg',
        'dislike_overlap_max',
        'affinity_overlap',
        'positive_support_ratio',
        'negative_conflict_ratio',
        'evidence_top1_overlap',
        'evidence_top2_overlap',
        'recent_recency_support',
        'stable_recent_gap',
        'positive_negative_gap',
        'weight_recent',
        'weight_stable',
        'weight_dislike',
        'weight_affinity',
    ]

    payload = dict(candidate_payload)
    payload['candidate_features'] = combined
    payload['feature_names'] = feature_names
    payload['memory_feature_slices'] = {
        'typed_end': BASE_DIM + TYPED_DIM,
        'coarse_end': BASE_DIM + TYPED_DIM + COARSE_EXTRA_DIM,
        'adaptive_end': BASE_DIM + TYPED_DIM + COARSE_EXTRA_DIM + ADAPTIVE_EXTRA_DIM,
    }
    payload['memory_stats'] = memory_stats
    path = os.path.join(out_dir, f'{split}_memory_graph.pt')
    torch.save(payload, path)
    return {
        'rows': num_rows,
        'feature_dim': combined.shape[-1],
        'stage1_hr10': float(candidate_payload['stage1_metrics']['HR@10']),
        'candidate_hit_rate': float(candidate_payload['candidate_hit_rate']),
        'avg_positive_nodes': float(sum(x['positive_count'] for x in memory_stats) / max(len(memory_stats), 1)),
        'avg_dislike_nodes': float(sum(x['dislike_count'] for x in memory_stats) / max(len(memory_stats), 1)),
        'avg_affinity_tokens': float(sum(x['affinity_count'] for x in memory_stats) / max(len(memory_stats), 1)),
    }


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    data_dir = os.path.join(script_dir, cfg['paths']['data_dir'])
    os.makedirs(data_dir, exist_ok=True)

    train_rows = load_jsonl(os.path.join(script_dir, cfg['paths']['source_train_path']))
    val_rows = load_jsonl(os.path.join(script_dir, cfg['paths']['source_val_path']))
    test_rows = load_jsonl(os.path.join(script_dir, cfg['paths']['source_test_path']))

    candidate_dir = os.path.join(script_dir, cfg['paths']['candidate17_data_dir'])
    train_payload = load_split_candidates(candidate_dir, 'train')
    val_payload = load_split_candidates(candidate_dir, 'val')
    test_payload = load_split_candidates(candidate_dir, 'test')

    item_title_lookup, _ = load_item_titles(os.path.join(script_dir, cfg['paths']['item_titles_path']))
    with open(os.path.join(candidate_dir, 'item_ids.json'), 'r', encoding='utf-8') as f:
        item_ids = json.load(f)

    summary = {
        'experiment': '19_memory_graph_toys',
        'splits': {
            'train': process_split('train', train_rows, train_payload, item_ids, item_title_lookup, cfg, data_dir),
            'val': process_split('val', val_rows, val_payload, item_ids, item_title_lookup, cfg, data_dir),
            'test': process_split('test', test_rows, test_payload, item_ids, item_title_lookup, cfg, data_dir),
        },
        'source_paths': {
            'train': os.path.abspath(os.path.join(script_dir, cfg['paths']['source_train_path'])),
            'val': os.path.abspath(os.path.join(script_dir, cfg['paths']['source_val_path'])),
            'test': os.path.abspath(os.path.join(script_dir, cfg['paths']['source_test_path'])),
            'candidate17_data_dir': os.path.abspath(candidate_dir),
        },
    }
    dump_json(os.path.join(data_dir, 'data_manifest.json'), summary)
    print(summary)


if __name__ == '__main__':
    main()
