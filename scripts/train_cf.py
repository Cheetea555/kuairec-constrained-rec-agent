# -*- coding: utf-8 -*-
"""ItemCF 协同过滤基线 + 全局热门基线。

ItemCF：物品-物品余弦相似（训练期共现矩阵），打分 = 用户历史物品的相似度和。
热门：训练期物品正样本计数。
输出：models/itemcf.npz、models/popularity.npz
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rec_common as rc  # noqa: E402


def train_cf():
    items_df, train_df, test_df = rc.load_processed()
    # 只用品类“喜欢”（label=1）的配对建共现矩阵，负样本不参与
    train_df = train_df[train_df['label'] == 1]
    users, vids, user2idx, item2idx = rc.build_index(train_df, items_df)
    n_u, n_i = len(users), len(vids)
    print('users=%d items=%d' % (n_u, n_i))

    tr_u = train_df['user_id'].map(user2idx).values
    tr_i = train_df['video_id'].map(item2idx).values

    # 物品共现（用户内历史物品对的计数），用逐用户外积累加
    cooc = np.zeros((n_i, n_i), dtype=np.float32)
    by_user = {}
    for u, i in zip(tr_u, tr_i):
        by_user.setdefault(u, []).append(i)
    for u, lst in by_user.items():
        arr = np.array(sorted(set(lst)), dtype=np.int64)
        cooc[np.ix_(arr, arr)] += 1.0
    # 余弦归一：对角为物品自身热度
    diag = np.sqrt(np.diag(cooc).copy())
    norm = np.outer(diag, diag)
    norm[norm == 0] = 1.0
    sim = cooc / norm
    np.fill_diagonal(sim, 0.0)
    print('itemcf sim built: shape=%s nonzero=%d'
          % (sim.shape, int((sim > 0).sum())))

    os.makedirs('models', exist_ok=True)
    np.savez_compressed('models/itemcf.npz', sim=sim,
                        users=np.array(users), vids=np.array(vids))

    # 热门基线
    pop = np.bincount(tr_i, minlength=n_i).astype(np.float32)
    print('popularity: max=%d median=%d' % (pop.max(), int(np.median(pop))))
    np.savez_compressed('models/popularity.npz', pop=pop,
                        users=np.array(users), vids=np.array(vids))
    print('saved: models/itemcf.npz, models/popularity.npz')


if __name__ == '__main__':
    train_cf()
