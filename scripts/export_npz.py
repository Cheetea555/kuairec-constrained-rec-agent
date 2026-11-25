# -*- coding: utf-8 -*-
"""把 prepare_data 的 parquet 结果导出为 npz（训练/评估脚本统一的索引格式）。

输出：data/processed/dataset.npz
  tr_u / tr_i         训练集正样本（label=1）的用户/物品下标
  tr_neg_u/tr_neg_i   训练集负样本（label=0，即该用户明确没看完的配对）
  te_u / te_i         测试集正样本
  te_neg_u/te_neg_i   测试集负样本
  users / vids        下标 -> 原始 id

正负标签都是“已知”的（全观测矩阵），因此训练时负样本直接取自 tr_neg，
不需要把未观测当作负样本。
"""
import os
import sys

import numpy as np

sys.path.insert(0, 'scripts')
import rec_common as rc  # noqa: E402


def main():
    items_df, train_df, test_df = rc.load_processed()
    users, vids, user2idx, item2idx = rc.build_index(train_df, items_df)
    print('users=%d items=%d' % (len(users), len(vids)))

    for df, name in ((train_df, 'train'), (test_df, 'test')):
        if 'label' not in df.columns:
            raise SystemExit('%s.parquet 缺少 label 列，请重跑 prepare_data.py' % name)

    def idx(df, label):
        sub = df[df['label'] == label]
        return (sub['user_id'].map(user2idx).values.astype(np.int64),
                sub['video_id'].map(item2idx).values.astype(np.int64))

    tr_u, tr_i = idx(train_df, 1)
    tr_nu, tr_ni = idx(train_df, 0)
    te_u, te_i = idx(test_df, 1)
    te_nu, te_ni = idx(test_df, 0)

    np.savez_compressed(
        'data/processed/dataset.npz',
        tr_u=tr_u, tr_i=tr_i, tr_neg_u=tr_nu, tr_neg_i=tr_ni,
        te_u=te_u, te_i=te_i, te_neg_u=te_nu, te_neg_i=te_ni,
        users=np.array([int(u) for u in users], dtype=np.int64),
        vids=np.array([int(v) for v in vids], dtype=np.int64))
    print('saved: data/processed/dataset.npz '
          '(train pos/neg=%d/%d, test pos/neg=%d/%d)'
          % (len(tr_u), len(tr_nu), len(te_u), len(te_nu)))


if __name__ == '__main__':
    main()
