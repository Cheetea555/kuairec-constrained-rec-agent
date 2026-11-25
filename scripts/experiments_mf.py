# -*- coding: utf-8 -*-
"""BPR-MF 对照实验（报告表 2、表 3 的数据来源，不改动 models/mf.npz）。

四组配置：
  1) 默认：lr=0.01, dim=32, 30 epoch
  2) 学习率 lr=0.05
  3) 隐维 dim=64
  4) 负采样消融：不用已知负样本，改为均匀随机负采样（--no-neg-from-label）

用法：python scripts/experiments_mf.py   （约 6 分钟，结果写 results/mf_experiments.json）
"""
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = [
    dict(name='base_lr0.01_d32_e30', args=['--epochs', '30']),
    dict(name='lr0.05_d32_e30', args=['--epochs', '30', '--lr', '0.05']),
    dict(name='lr0.01_d64_e30', args=['--epochs', '30', '--dim', '64']),
    dict(name='uniform_neg_e30', args=['--epochs', '30', '--no-neg-from-label']),
]
PAT = re.compile(r'best HR@10=([0-9.]+) @ep(\d+)')


def main():
    rows = []
    for cfg in CONFIGS:
        out = os.path.join('results', 'exp_' + cfg['name'] + '.npz')
        hist = os.path.join('results', 'exp_' + cfg['name'] + '.json')
        cmd = [sys.executable, '-X', 'utf8', '-u', 'scripts/train_mf.py',
               '--out', out, '--history-out', hist] + cfg['args']
        print('=' * 70)
        print(' '.join(cmd[3:]))
        t0 = time.perf_counter()
        p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
        text = p.stdout.decode('utf-8', errors='replace')
        print(text.strip().splitlines()[-1] if text.strip() else '(no output)')
        m = PAT.search(text)
        rows.append({'config': cfg['name'], 'args': cfg['args'],
                     'best_hr': float(m.group(1)) if m else None,
                     'best_epoch': int(m.group(2)) if m else None,
                     'sec': round(time.perf_counter() - t0, 1)})
    with open(os.path.join(ROOT, 'results', 'mf_experiments.json'), 'w',
              encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    for r in rows:
        print('%-24s HR@10=%.4f @ep%s  %.0fs'
              % (r['config'], r['best_hr'] or 0, r['best_epoch'], r['sec']))


if __name__ == '__main__':
    main()
