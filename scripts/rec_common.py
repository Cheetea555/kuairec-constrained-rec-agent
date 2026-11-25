# -*- coding: utf-8 -*-
"""共享工具：加载已准备的数据与模型，统一索引映射与评估指标。"""
import os

import numpy as np
import pandas as pd

PROC = 'data/processed'
MODELS = 'models'


def load_processed():
    """读取 items/train/test 三张表，返回 (items_df, train_df, test_df)。"""
    items = pd.read_parquet(os.path.join(PROC, 'items.parquet'))
    train = pd.read_parquet(os.path.join(PROC, 'train.parquet'))
    test = pd.read_parquet(os.path.join(PROC, 'test.parquet'))
    return items, train, test


def build_index(train_df, items_df):
    """构建 user/item -> 连续下标的映射。"""
    users = sorted(train_df['user_id'].unique())
    vids = sorted(items_df['video_id'].unique())
    user2idx = {u: k for k, u in enumerate(users)}
    item2idx = {v: k for k, v in enumerate(vids)}
    return users, vids, user2idx, item2idx


def user_history(train_df):
    """{user_id: [video_id, ...]} 训练期历史。"""
    return train_df.groupby('user_id')['video_id'].apply(list).to_dict()


def item_categories(items_df, vids):
    """{video_id: 一级类目}。"""
    m = items_df.set_index('video_id')['first_level_category_name'].to_dict()
    return {v: m.get(v, '其他') for v in vids}


def recall_at_k(relevant, ranked, k):
    r = set(relevant)
    if not r:
        return None
    hit = len(r & set(ranked[:k]))
    return hit / len(r)


def ndcg_at_k(relevant, ranked, k):
    r = set(relevant)
    if not r:
        return None
    dcg = 0.0
    for pos, it in enumerate(ranked[:k]):
        if it in r:
            dcg += 1.0 / np.log2(pos + 2)
    ideal = min(len(r), k)
    idcg = sum(1.0 / np.log2(p + 2) for p in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mean_metric(metric_fn, ground_truth, rankings, k):
    """对每个有 ground truth 的用户计算指标并求均值。"""
    vals = []
    for u, ranked in rankings.items():
        rel = ground_truth.get(u, [])
        if rel:
            v = metric_fn(rel, ranked, k)
            if v is not None:
                vals.append(v)
    return float(np.mean(vals)), len(vals)
