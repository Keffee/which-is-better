#!/usr/bin/env python3
import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from src.common.pipeline_utils import dump_json

def parse_args():
    p=argparse.ArgumentParser(description='Evaluate dynamic alpha policies.')
    p.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), 'dynamic_alpha_config.yaml'))
    p.add_argument('--variant', type=str, required=True)
    p.add_argument('--output_prefix', type=str, required=True)
    return p.parse_args()

def load_split(cache_dir, split):
    return torch.load(os.path.join(cache_dir, f'{split}_features.pt'), map_location='cpu', weights_only=False)

def batch_iter(n, bs):
    for s in range(0, n, bs): yield s, min(n, s+bs)

def fuse_scores(sas_scores, llm_scores, alpha): return alpha*sas_scores + (1.0-alpha)*llm_scores

def rank_metrics(scores, targets, k_values):
    total=int(targets.shape[0]); max_k=max(k_values); topk=torch.topk(scores, k=max_k, dim=1).indices.cpu(); targets_cpu=targets.cpu(); metrics={}
    for k in k_values:
        hits=0.0; ndcg=0.0; rows=topk[:, :k]
        for i in range(total):
            t=int(targets_cpu[i].item()); row=rows[i].tolist()
            if t in row:
                hits+=1.0; rank=row.index(t)+1; ndcg += 1.0/np.log2(rank+1)
        metrics[f'HR@{k}']=hits/total; metrics[f'NDCG@{k}']=ndcg/total
    return metrics

def best_alpha_labels(sas_scores, llm_scores, targets, alpha_grid):
    labels=[]
    for i in range(targets.shape[0]):
        best_idx=0; best_score=-1e18; best_hr1=-1
        t=int(targets[i].item())
        for j,a in enumerate(alpha_grid):
            scores = a*sas_scores[i] + (1.0-a)*llm_scores[i]
            hr1 = int(int(torch.argmax(scores).item()) == t)
            target_score=float(scores[t].item())
            if hr1 > best_hr1 or (hr1 == best_hr1 and target_score > best_score):
                best_hr1=hr1; best_score=target_score; best_idx=j
        labels.append(best_idx)
    return torch.tensor(labels, dtype=torch.long)

def build_features(split, sas_scores, llm_scores, alpha_fixed, feature_mode):
    fixed=fuse_scores(sas_scores, llm_scores, alpha_fixed)
    context=split['context_hidden'].float(); stats=split['stats'].float()
    fixed_top=torch.argmax(fixed, dim=1); llm_top=torch.argmax(llm_scores, dim=1); sas_top=torch.argmax(sas_scores, dim=1)
    rows=[]
    for i in range(fixed.shape[0]):
        b=int(fixed_top[i].item()); l=int(llm_top[i].item()); s=int(sas_top[i].item())
        scalar=torch.tensor([
            float(fixed[i,b].item()), float(llm_scores[i,l].item()), float(sas_scores[i,s].item()),
            float(fixed[i,l].item()-fixed[i,b].item()), float(fixed[i,s].item()-fixed[i,b].item()),
            float(b==l), float(b==s), float(l==s),
        ], dtype=torch.float32)
        if feature_mode == 'score_only':
            rows.append(torch.cat([stats[i], scalar], dim=0))
        else:
            rows.append(torch.cat([context[i], stats[i], scalar], dim=0))
    return torch.stack(rows, dim=0)

class AlphaNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, out_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))
    def forward(self, x): return self.net(x)

def compute_norm_stats(x):
    mean=x.mean(dim=0, keepdim=True); std=x.std(dim=0, keepdim=True).clamp(min=1e-6); return mean, std

def normalize_x(x, mean, std): return (x-mean)/std

def train_classifier(x_train, y_train, x_val, y_val, cfg, device):
    tc=cfg['training']; model=AlphaNet(x_train.shape[1], int(tc['hidden_dim']), float(tc['dropout']), int(tc['num_alpha_bins'])).to(device); opt=torch.optim.AdamW(model.parameters(), lr=float(tc['lr']), weight_decay=float(tc['weight_decay']))
    best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}; best_acc=-1.0; patience=0; history=[]
    for epoch in range(int(tc['num_epochs'])):
        model.train(); total_loss=0.0
        for s,e in batch_iter(x_train.shape[0], int(tc['batch_size'])):
            logits=model(x_train[s:e]); loss=F.cross_entropy(logits, y_train[s:e]); opt.zero_grad(); loss.backward(); opt.step(); total_loss += float(loss.item())*(e-s)
        model.eval()
        with torch.no_grad(): acc=float((model(x_val).argmax(dim=1) == y_val).float().mean().item())
        history.append({'epoch':epoch+1,'train_loss':total_loss/max(1,x_train.shape[0]),'val_alpha_acc':acc})
        if acc > best_acc: best_acc=acc; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}; patience=0
        else:
            patience += 1
            if patience >= int(tc['early_stop_patience']): break
    model.load_state_dict(best_state); return model, history

def apply_alpha_policy(sas_scores, llm_scores, alpha_values, alpha_fixed, mode, predicted_alpha, confidence=None, conf_thresh=0.0):
    if mode == 'fixed':
        alpha=torch.full((sas_scores.shape[0],), alpha_fixed)
    elif mode == 'oracle':
        alpha=alpha_values[predicted_alpha]
    elif mode == 'gated':
        use = confidence >= conf_thresh
        alpha=torch.where(use, alpha_values[predicted_alpha], torch.full_like(predicted_alpha, alpha_fixed, dtype=torch.float32))
    else:
        alpha=alpha_values[predicted_alpha]
    return alpha.unsqueeze(1) * sas_scores + (1.0 - alpha.unsqueeze(1)) * llm_scores, alpha

def main():
    args=parse_args(); script_dir=os.path.dirname(os.path.abspath(__file__))
    cfg=yaml.safe_load(open(args.config,'r',encoding='utf-8')); cache_dir=os.path.abspath(os.path.join(script_dir, cfg['paths']['cache_dir'])); result_dir=os.path.join(script_dir, cfg['paths']['results_dir']); os.makedirs(result_dir, exist_ok=True)
    variant_cfg=next(v for v in cfg['variants'] if v['name']==args.variant); mode=variant_cfg['mode']; tc=cfg['training']
    alpha_grid=torch.tensor([float(x) for x in tc['alpha_grid']], dtype=torch.float32)
    alpha_fixed=float(tc['alpha_fixed'])
    train=load_split(cache_dir,'train'); val=load_split(cache_dir,'val'); test=load_split(cache_dir,'test')
    train_sas=train['sas_scores'].float(); train_llm=train['llm_scores'].float(); val_sas=val['sas_scores'].float(); val_llm=val['llm_scores'].float(); test_sas=test['sas_scores'].float(); test_llm=test['llm_scores'].float()
    feature_mode = tc.get('feature_mode', 'full')
    train_x=build_features(train, train_sas, train_llm, alpha_fixed, feature_mode); val_x=build_features(val, val_sas, val_llm, alpha_fixed, feature_mode); test_x=build_features(test, test_sas, test_llm, alpha_fixed, feature_mode)
    mean,std=compute_norm_stats(train_x)
    train_x=normalize_x(train_x, mean, std); val_x=normalize_x(val_x, mean, std); test_x=normalize_x(test_x, mean, std)
    train_y=best_alpha_labels(train_sas, train_llm, train['targets'], alpha_grid); val_y=best_alpha_labels(val_sas, val_llm, val['targets'], alpha_grid); test_y=best_alpha_labels(test_sas, test_llm, test['targets'], alpha_grid)
    if mode == 'fixed':
        scores, alpha = apply_alpha_policy(test_sas, test_llm, alpha_grid, alpha_fixed, 'fixed', torch.zeros(test_x.shape[0], dtype=torch.long))
        metrics=rank_metrics(scores, test['targets'], [1,5,10,20]); metrics.update({'variant':args.variant,'mean_alpha':float(alpha.mean().item()),'alpha_histogram':{str(alpha_fixed):int(alpha.shape[0])}})
    elif mode == 'oracle':
        scores, alpha = apply_alpha_policy(test_sas, test_llm, alpha_grid, alpha_fixed, 'oracle', test_y)
        metrics=rank_metrics(scores, test['targets'], [1,5,10,20]); metrics.update({'variant':args.variant,'mean_alpha':float(alpha.mean().item())})
    else:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, history = train_classifier(train_x.to(device), train_y.to(device), val_x.to(device), val_y.to(device), cfg, device)
        with torch.no_grad():
            val_logits=model(val_x.to(device)).cpu(); test_logits=model(test_x.to(device)).cpu()
        val_pred=val_logits.argmax(dim=1); test_pred=test_logits.argmax(dim=1)
        val_conf=F.softmax(val_logits, dim=1).max(dim=1).values; test_conf=F.softmax(test_logits, dim=1).max(dim=1).values
        apply_mode='gated' if mode == 'gated' else 'predicted'
        scores, alpha = apply_alpha_policy(test_sas, test_llm, alpha_grid, alpha_fixed, apply_mode, test_pred.long(), test_conf, float(tc.get('confidence_threshold', 0.0)))
        metrics=rank_metrics(scores, test['targets'], [1,5,10,20])
        alpha_hist={str(float(a)): int((alpha == float(a)).sum().item()) for a in alpha_grid.tolist()}
        metrics.update({'variant':args.variant,'training_history':history,'val_alpha_acc':float((val_pred == val_y).float().mean().item()),'test_alpha_acc':float((test_pred == test_y).float().mean().item()),'mean_alpha':float(alpha.mean().item()),'alpha_histogram':alpha_hist,'used_predicted_rate':float((test_conf >= float(tc.get('confidence_threshold', 0.0))).float().mean().item()) if mode == 'gated' else 1.0})
    dump_json(os.path.join(result_dir, f'{args.output_prefix}_metrics.json'), metrics)
    print(metrics)

if __name__=='__main__':
    main()
