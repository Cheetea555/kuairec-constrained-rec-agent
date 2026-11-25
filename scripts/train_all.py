# -*- coding: utf-8 -*-
"""一键训练全部模型：ItemCF/热门 → BPR-MF → K-Means 冷启动分群。

用法：
  python scripts/train_all.py            # 默认 40 epoch
  python scripts/train_all.py --epochs 60
每个步骤的完整输出同时写入 logs/*.log，便于复现与排错。
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = [
    ('scripts/train_cf.py', 'logs/train_cf.log', []),
    ('scripts/train_mf.py', 'logs/train_mf.log', []),
    ('scripts/train_kmeans.py', 'logs/train_kmeans.log', []),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, 'logs'), exist_ok=True)

    t_all = time.perf_counter()
    for script, log, extra in STEPS:
        cmd = [sys.executable, '-X', 'utf8', '-u', script] + extra
        if script.endswith('train_mf.py'):
            cmd += ['--epochs', str(args.epochs)]
        print('=' * 70)
        print('[%s] %s' % (time.strftime('%H:%M:%S'), ' '.join(cmd)))
        t0 = time.perf_counter()
        with open(os.path.join(ROOT, log), 'w', encoding='utf-8') as f:
            p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
            out = p.stdout.decode('utf-8', errors='replace')
            f.write(out)
        print(out.rstrip())
        if p.returncode != 0:
            print('!! %s 失败（exit=%d），见 %s' % (script, p.returncode, log))
            sys.exit(p.returncode)
        print('[%s] 完成，耗时 %.1fs' % (script, time.perf_counter() - t0))
    print('=' * 70)
    print('全部模型训练完成，总耗时 %.1fs' % (time.perf_counter() - t_all))


if __name__ == '__main__':
    main()
