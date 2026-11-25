# -*- coding: utf-8 -*-
"""一键数据准备：清洗与划分 → 导出 npz → 导出文本元数据。

用法：python scripts/prepare_all.py
（原始数据需先用 scripts/download_data.py 下载到 data/raw/）
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = [('scripts/prepare_data.py', 'logs/prepare.log'),
         ('scripts/export_npz.py', 'logs/export_npz.log'),
         ('scripts/export_meta.py', 'logs/export_meta.log')]


def main():
    os.makedirs(os.path.join(ROOT, 'logs'), exist_ok=True)
    for script, log in STEPS:
        cmd = [sys.executable, '-X', 'utf8', '-u', script]
        print('=' * 70)
        print('[%s] %s' % (time.strftime('%H:%M:%S'), ' '.join(cmd)))
        t0 = time.perf_counter()
        p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
        out = p.stdout.decode('utf-8', errors='replace')
        with open(os.path.join(ROOT, log), 'w', encoding='utf-8') as f:
            f.write(out)
        print(out.rstrip())
        if p.returncode != 0:
            print('!! %s 失败（exit=%d），见 %s' % (script, p.returncode, log))
            sys.exit(p.returncode)
        print('完成，耗时 %.1fs' % (time.perf_counter() - t0))


if __name__ == '__main__':
    main()
