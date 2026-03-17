#!/usr/bin/env python3

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from src.common.pipeline_utils import dump_json, load_item_titles, load_jsonl, normalize_title


FEATURE_NAMES = [
    'dense_z',
    'lexical_z',
    'hybrid_score',
    'sas_z',
    'dense_minus_sas',
    'hybrid_rank_recip',
    'dense_top5_flag',
    'sas_top5_flag',
    'generated_overlap',
    'positive_overlap_max',
    'positive_overlap_avg',
    'negative_overlap_max',
    'title_length_norm',
    'history_positive_count_norm',
    'stat_0',
    'stat_1',
    'stat_2',
    'stat_3',
    'stat_4',
    'stat_5',
    'stat_6',
    'stat_7',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare stage-1 top20 candidate data for 17_ reranker.')
    parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), 'recommendation_specific_reranker_config.yaml'))
    return parser.parse_args()


def tokenize_title(text: str) -> List[str]:
    text = normalize_title(text).lower()
    return [tok for tok in text.replace('/', ' ').replace('-', ' ').split() if tok]


def token_overlap(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return inter / union


def zscore_rows(scores: torch.Tensor) -> torch.Tensor:
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True).clamp(min=1e-6)
    return (scores - mean) / std


def dense_row_z(scores: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (scores - mean) / max(std, 1e-6)


def rank_metrics(topk_indices: torch.Tensor, targets: torch.Tensor, k_values: List[int]) -> Dict[str, float]:
    total = int(targets.shape[0])
    metrics: Dict[str, float] = {}
    targets_cpu = targets.cpu()
    for k in k_values:
        hits = 0.0
        ndcg = 0.0
        subset = topk_indices[:, :k]
        for i in range(total):
            target = int(targets_cpu[i].item())
            row = subset[i].tolist()
            if target in row:
                hits += 1.0
                rank = row.index(target) + 1
                ndcg += 1.0 / np.log2(rank + 1)
        metrics[f'HR@{k}'] = hits / total
        metrics[f'NDCG@{k}'] = ndcg / total
        if k == 20:
            metrics['Recall@20'] = hits / total
    return metrics


def build_lexical_index(item_titles: List[str]) -> Dict[str, object]:
    word_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), analyzer='word')
    char_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(3, 5), analyzer='char_wb')
    word_matrix = word_vectorizer.fit_transform(item_titles)
    char_matrix = char_vectorizer.fit_transform(item_titles)
    return {
        'word_vectorizer': word_vectorizer,
        'char_vectorizer': char_vectorizer,
        'word_matrix': word_matrix,
        'char_matrix': char_matrix,
    }


def lexical_scores(index: Dict[str, object], query_texts: List[str], word_weight: float, char_weight: float) -> torch.Tensor:
    word_q = index['word_vectorizer'].transform(query_texts)
    char_q = index['char_vectorizer'].transform(query_texts)
    word_scores = word_q @ index['word_matrix'].T
    char_scores = char_q @ index['char_matrix'].T
    combined = word_weight * np.asarray(word_scores.todense(), dtype=np.float32) + char_weight * np.asarray(char_scores.todense(), dtype=np.float32)
    return torch.from_numpy(combined)


def align_rows(raw_rows: List[dict], user_ids: List[str], target_item_ids: List[str]) -> List[dict]:
    lookup = {(row['user_id'], row['target_item_id']): row for row in raw_rows}
    aligned = []
    for user_id, target_item_id in zip(user_ids, target_item_ids):
        key = (user_id, target_item_id)
        if key not in lookup:
            raise KeyError(f'missing raw row for {key}')
        aligned.append(lookup[key])
    return aligned


def build_query_text(row: dict, generated_text: str, recent_k: int, positive_threshold: int) -> Tuple[str, List[List[str]], List[List[str]], List[str]]:
    history_titles = row.get('history_titles', [])[-recent_k:]
    history_ratings = row.get('history_ratings', [])[-recent_k:]
    positive_tokens: List[List[str]] = []
    negative_tokens: List[List[str]] = []
    recent_positive_titles: List[str] = []
    for title, rating in zip(history_titles, history_ratings):
        tokens = tokenize_title(title)
        if int(rating) >= positive_threshold:
            positive_tokens.append(tokens)
            recent_positive_titles.append(normalize_title(title))
        else:
            negative_tokens.append(tokens)
    base = normalize_title(generated_text) or 'unknown'
    parts = [base] + [title for title in recent_positive_titles if title]
    query = ' ; '.join(parts)
    return query or base, positive_tokens, negative_tokens, tokenize_title(base)


def process_split(
    split_name: str,
    rows: List[dict],
    cache_payload: dict,
    index: Dict[str, object],
    item_ids: List[str],
    item_titles: List[str],
    item_token_lists: List[List[str]],
    item_embeddings: torch.Tensor,
    cfg: dict,
    data_dir: str,
) -> Dict[str, float]:
    data_cfg = cfg['data']
    top_k = int(data_cfg['top_k_candidates'])
    lex_batch_size = int(data_cfg['lexical_batch_size'])
    dense_weight = float(data_cfg['dense_weight_hybrid'])
    lexical_weight = float(data_cfg['lexical_weight_hybrid'])
    word_weight = float(data_cfg['lexical_word_weight'])
    char_weight = float(data_cfg['lexical_char_weight'])
    recent_k = int(data_cfg['history_recent_k'])
    positive_threshold = int(data_cfg['positive_rating_threshold'])

    num_rows = len(rows)
    candidate_indices = torch.empty((num_rows, top_k), dtype=torch.int64)
    candidate_features = torch.empty((num_rows, top_k, len(FEATURE_NAMES)), dtype=torch.float32)
    candidate_hybrid_scores = torch.empty((num_rows, top_k), dtype=torch.float32)
    candidate_llm_raw_scores = torch.empty((num_rows, top_k), dtype=torch.float32)
    candidate_sas_raw_scores = torch.empty((num_rows, top_k), dtype=torch.float32)
    target_positions = torch.full((num_rows,), -1, dtype=torch.int64)
    targets = cache_payload['targets'].clone().long()

    queries: List[str] = []
    positive_history_tokens: List[List[List[str]]] = []
    negative_history_tokens: List[List[List[str]]] = []
    generated_tokens: List[List[str]] = []
    history_positive_counts: List[float] = []
    for row, generated_text in zip(rows, cache_payload['generated_texts']):
        query, pos_tokens, neg_tokens, gen_tokens = build_query_text(row, generated_text, recent_k, positive_threshold)
        queries.append(query)
        positive_history_tokens.append(pos_tokens)
        negative_history_tokens.append(neg_tokens)
        generated_tokens.append(gen_tokens)
        history_positive_counts.append(min(len(pos_tokens), recent_k) / max(recent_k, 1))

    llm_prompt_embeddings = cache_payload['llm_prompt_embeddings'].float()
    llm_topk_indices = cache_payload['llm_topk_indices'].long()
    llm_topk_scores = cache_payload['llm_topk_scores'].float()
    sas_topk_indices = cache_payload['sas_topk_indices'].long()
    sas_topk_scores = cache_payload['sas_topk_scores'].float()
    llm_score_mean = cache_payload['llm_score_mean'].float()
    llm_score_std = cache_payload['llm_score_std'].float()
    sas_score_mean = cache_payload['sas_score_mean'].float()
    sas_score_std = cache_payload['sas_score_std'].float()
    stats = cache_payload['stats'].float()
    dense_pool_k = min(llm_topk_indices.shape[1], max(top_k * 3, 60))
    sas_pool_k = min(sas_topk_indices.shape[1], max(top_k * 2, 40))
    lexical_pool_k = max(top_k * 3, 60)

    for start in tqdm(range(0, num_rows, lex_batch_size), desc=f'prepare_{split_name}'):
        end = min(num_rows, start + lex_batch_size)
        lexical_batch = lexical_scores(index, queries[start:end], word_weight, char_weight)
        lex_row_mean = lexical_batch.mean(dim=1)
        lex_row_std = lexical_batch.std(dim=1).clamp(min=1e-6)

        for local_idx in range(end - start):
            row_idx = start + local_idx
            lexical_row = lexical_batch[local_idx]
            lex_top = torch.topk(lexical_row, k=min(lexical_pool_k, lexical_row.numel())).indices.tolist()
            dense_top = llm_topk_indices[row_idx, :dense_pool_k].tolist()
            sas_top = sas_topk_indices[row_idx, :sas_pool_k].tolist()
            pool = []
            seen = set()
            for cand in dense_top + lex_top + sas_top:
                if cand not in seen:
                    seen.add(cand)
                    pool.append(cand)

            pool_tensor = torch.tensor(pool, dtype=torch.long)
            prompt_embedding = llm_prompt_embeddings[row_idx]
            dense_raw = torch.matmul(item_embeddings[pool_tensor].float(), prompt_embedding)
            dense_z = dense_row_z(dense_raw, float(llm_score_mean[row_idx].item()), float(llm_score_std[row_idx].item()))
            lex_scores = lexical_row[pool_tensor]
            lex_z = dense_row_z(lex_scores, float(lex_row_mean[local_idx].item()), float(lex_row_std[local_idx].item()))

            sas_lookup = {
                int(idx): float(score)
                for idx, score in zip(sas_topk_indices[row_idx].tolist(), sas_topk_scores[row_idx].tolist())
            }
            sas_fallback = max(float(sas_score_mean[row_idx].item() - sas_score_std[row_idx].item()), 0.0)
            sas_raw = torch.tensor([sas_lookup.get(int(c), sas_fallback) for c in pool], dtype=torch.float32)
            sas_z = dense_row_z(sas_raw, float(sas_score_mean[row_idx].item()), float(sas_score_std[row_idx].item()))
            hybrid_scores = dense_weight * dense_z + lexical_weight * lex_z
            top_hybrid = torch.topk(hybrid_scores, k=min(top_k, hybrid_scores.numel()))
            row_candidates = [pool[idx] for idx in top_hybrid.indices.tolist()]
            candidate_indices[row_idx] = torch.tensor(row_candidates, dtype=torch.int64)
            candidate_hybrid_scores[row_idx] = top_hybrid.values.float()
            target = int(targets[row_idx].item())
            if target in row_candidates:
                target_positions[row_idx] = row_candidates.index(target)
            pos_tokens = positive_history_tokens[row_idx]
            neg_tokens = negative_history_tokens[row_idx]
            gen_tokens = generated_tokens[row_idx]
            row_stats = stats[row_idx]
            pos_count_norm = history_positive_counts[row_idx]
            dense_top5_set = set(llm_topk_indices[row_idx, :5].tolist())
            sas_top5_set = set(sas_topk_indices[row_idx, :5].tolist())

            for rank, cand_idx in enumerate(row_candidates):
                pool_pos = pool.index(cand_idx)
                cand_tokens = item_token_lists[cand_idx]
                gen_overlap = token_overlap(cand_tokens, gen_tokens)
                pos_overlaps = [token_overlap(cand_tokens, toks) for toks in pos_tokens] or [0.0]
                neg_overlaps = [token_overlap(cand_tokens, toks) for toks in neg_tokens] or [0.0]
                dense_val = float(dense_z[pool_pos].item())
                sas_val = float(sas_z[pool_pos].item())
                feature_row = [
                    dense_val,
                    float(lex_z[pool_pos].item()),
                    float(hybrid_scores[pool_pos].item()),
                    sas_val,
                    float(dense_val - sas_val),
                    1.0 / float(rank + 1),
                    float(cand_idx in dense_top5_set),
                    float(cand_idx in sas_top5_set),
                    gen_overlap,
                    max(pos_overlaps),
                    float(sum(pos_overlaps) / len(pos_overlaps)),
                    max(neg_overlaps),
                    min(len(cand_tokens), 8) / 8.0,
                    pos_count_norm,
                ] + [float(x) for x in row_stats.tolist()]
                candidate_features[row_idx, rank] = torch.tensor(feature_row, dtype=torch.float32)
                candidate_llm_raw_scores[row_idx, rank] = float(dense_raw[pool_pos].item())
                candidate_sas_raw_scores[row_idx, rank] = float(sas_raw[pool_pos].item())

    stage1_metrics = rank_metrics(candidate_indices, targets, [1, 5, 10, 20])
    hit_mask = target_positions >= 0
    hit_positions = target_positions[hit_mask].float()
    avg_target_pos = float(hit_positions.mean().item()) if hit_positions.numel() else -1.0

    payload = {
        'split': split_name,
        'feature_names': FEATURE_NAMES,
        'candidate_indices': candidate_indices,
        'candidate_features': candidate_features,
        'candidate_hybrid_scores': candidate_hybrid_scores,
        'candidate_llm_raw_scores': candidate_llm_raw_scores,
        'candidate_sas_raw_scores': candidate_sas_raw_scores,
        'target_positions': target_positions,
        'targets': targets,
        'user_ids': cache_payload['user_ids'],
        'target_item_ids': cache_payload['target_item_ids'],
        'generated_texts': cache_payload['generated_texts'],
        'queries': queries,
        'stage1_metrics': stage1_metrics,
        'avg_target_position_if_hit': avg_target_pos,
        'candidate_hit_rate': float(hit_mask.float().mean().item()),
    }
    torch.save(payload, os.path.join(data_dir, f'{split_name}_candidates.pt'))
    return {
        'num_rows': num_rows,
        'candidate_hit_rate': float(hit_mask.float().mean().item()),
        'avg_target_position_if_hit': avg_target_pos,
        **stage1_metrics,
    }


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    data_dir = os.path.join(script_dir, cfg['paths']['data_dir'])
    os.makedirs(data_dir, exist_ok=True)

    raw_data_dir = os.path.abspath(os.path.join(script_dir, cfg['paths']['raw_data_dir']))
    aligned_cache_dir = os.path.abspath(os.path.join(script_dir, cfg['paths']['aligned_cache_dir']))
    titles_path = os.path.abspath(os.path.join(script_dir, cfg['paths']['item_titles_path']))

    item_to_title, _ = load_item_titles(titles_path)
    item_embed_payload = torch.load(os.path.join(aligned_cache_dir, 'item_embeddings.pt'), map_location='cpu', weights_only=False)
    item_ids = item_embed_payload['item_ids']
    item_embeddings = item_embed_payload['item_embeddings'].float()
    item_titles = [normalize_title(item_to_title[item_id]) for item_id in item_ids]
    item_token_lists = [tokenize_title(title) for title in item_titles]

    index = build_lexical_index(item_titles)

    split_summaries: Dict[str, Dict[str, float]] = {}
    for split_name in ['train', 'val', 'test']:
        raw_rows = load_jsonl(os.path.join(raw_data_dir, f'{split_name}.jsonl'))
        cache_payload = torch.load(os.path.join(aligned_cache_dir, f'{split_name}_features.pt'), map_location='cpu', weights_only=False)
        aligned_rows = align_rows(raw_rows, cache_payload['user_ids'], cache_payload['target_item_ids'])
        split_summaries[split_name] = process_split(
            split_name=split_name,
            rows=aligned_rows,
            cache_payload=cache_payload,
            index=index,
            item_ids=item_ids,
            item_titles=item_titles,
            item_token_lists=item_token_lists,
            item_embeddings=item_embeddings,
            cfg=cfg,
            data_dir=data_dir,
        )

    manifest = {
        'experiment': '17_reranker_toys',
        'feature_names': FEATURE_NAMES,
        'num_items': len(item_ids),
        'top_k_candidates': int(cfg['data']['top_k_candidates']),
        'item_ids_path': os.path.join(data_dir, 'item_ids.json'),
        'split_summaries': split_summaries,
        'source_cache_dir': aligned_cache_dir,
        'source_raw_data_dir': raw_data_dir,
    }
    dump_json(os.path.join(data_dir, 'item_ids.json'), item_ids)
    dump_json(os.path.join(data_dir, 'data_manifest.json'), manifest)
    print(manifest)


if __name__ == '__main__':
    main()
