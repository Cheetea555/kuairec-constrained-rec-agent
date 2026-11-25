# -*- coding: utf-8 -*-
"""导出 GitHub Pages 纯前端演示所需的静态数据。

思路：Flask 跑不了在 GitHub Pages 上，但 BPR-MF 的推理只是
score(u, i) = P[u] · Q[i]，完全可以在浏览器里算。本脚本把
  - 模型参数 P(1411×32)、Q(3327×32)  -> docs/assets/model.bin (float32)
  - 训练期已看矩阵（位图）            -> docs/assets/seen.bin
  - 物品类目与文案、用户 id 列表       -> docs/assets/items.json, users.json
导出后页面即可离线完成“召回→打分→重排→理由”全流程。

用法：python scripts/export_web.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, 'docs', 'assets')
PROC = os.path.join(ROOT, 'data', 'processed')


def main():
    os.makedirs(DOCS, exist_ok=True)
    meta = json.load(open(os.path.join(PROC, 'meta.json'), encoding='utf-8'))
    users = [int(u) for u in meta['users']]
    vids = [int(v) for v in meta['vids']]
    n_u, n_i = len(users), len(vids)

    mf = np.load(os.path.join(ROOT, 'models', 'mf.npz'), allow_pickle=False)
    P = mf['P'].astype('<f4')
    Q = mf['Q'].astype('<f4')
    assert P.shape == (n_u, 32) and Q.shape == (n_i, 32), (P.shape, Q.shape)
    with open(os.path.join(DOCS, 'model.bin'), 'wb') as f:
        f.write(P.tobytes())
        f.write(Q.tobytes())

    d = np.load(os.path.join(PROC, 'dataset.npz'), allow_pickle=False)
    seen = np.zeros(n_u * n_i, dtype=bool)
    seen[d['tr_u'].astype(np.int64) * n_i + d['tr_i'].astype(np.int64)] = True
    bits = np.packbits(seen)                      # 高位在前
    bits.tofile(os.path.join(DOCS, 'seen.bin'))

    item_cat = meta['item_cat']
    item_cap = meta['item_caption']
    items = [[int(v), item_cat[str(v)], (item_cap.get(str(v), '') or '')[:48]]
             for v in vids]
    with open(os.path.join(DOCS, 'items.json'), 'w', encoding='utf-8') as f:
        json.dump({'items': items, 'users': users, 'dims': 32, 'n_i': n_i}, f,
                  ensure_ascii=False, separators=(',', ':'))
    print('items=%d users=%d model=%.1fKB seen=%.1fKB'
          % (n_i, n_u, os.path.getsize(os.path.join(DOCS, 'model.bin')) / 1024,
             os.path.getsize(os.path.join(DOCS, 'seen.bin')) / 1024))


if __name__ == '__main__':
    main()
