#!/usr/bin/env python3
import argparse, os, yaml
from src.common.pipeline_utils import dump_json

def parse_args():
    p=argparse.ArgumentParser(description='Prepare dynamic alpha metadata.')
    p.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), 'dynamic_alpha_config.yaml'))
    return p.parse_args()

def main():
    args=parse_args(); script_dir=os.path.dirname(os.path.abspath(__file__))
    cfg=yaml.safe_load(open(args.config,'r',encoding='utf-8'))
    cache_dir=os.path.abspath(os.path.join(script_dir, cfg['paths']['cache_dir']))
    dump_json(os.path.join(script_dir, cfg['paths']['data_dir'], 'data_manifest.json'), {'source_cache_dir': cache_dir, 'experiment': cfg['meta']['experiment_name']})
    print({'source_cache_dir': cache_dir, 'experiment': cfg['meta']['experiment_name']})

if __name__=='__main__':
    main()
