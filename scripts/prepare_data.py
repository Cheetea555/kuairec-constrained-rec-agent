# -*- coding: utf-8 -*-
"""数据准备：KuaiRec 小矩阵的全观测特性 → 用户内留出法划分 + 显式正负标签。

数据结构（数据探查结论，写入报告 3.2）
  KuaiRec 的“小矩阵”是 **全观测** 矩阵：1411 用户 × 3327 视频共 469 万格，
  每一格都有 watch_ratio（用户被要求逐条观看打分），其中 71% 的格子
  watch_ratio ≥ 0.5。因此“没看过”在这份数据里几乎不存在，
  传统的“未来观看 / 隐式反馈”口径会把大量真实正样本当成负样本。

  本项目采用用户内留出法（leave-pairs-out），与全观测设计匹配：
    标签   positive = watch_ratio ≥ 0.5（看完一半以上），否则 negative
    划分   每个用户随机留出 10% 的 (用户, 视频) 配对作为测试，
           其余 90% 作为训练；训练集里正负样本标签都是已知的，
           因此 BPR 可以直接从“已知不喜欢”的配对中采样负样本，
           不必依赖“未观测 ≈ 负样本”的假设。
    用户/物品 全量 1411 / 3327 都参与训练与测试（不冷物品），
           冷启动能力另由 K-Means 分群 + 声明兴趣模拟（见 evaluate.py）。

输出：data/processed/{items,train,test}.parquet
      train/test 三列：user_id、video_id、label（0/1）
"""
import os

import numpy as np
import pandas as pd

RAW = 'data/raw'
OUT = 'data/processed'
POS_RATIO = 0.5
TEST_FRAC = 0.10
SEED = 2026


def main():
    cap = pd.read_parquet(os.path.join(RAW, 'item_caption_category/train/0000.parquet'))
    small = pd.read_parquet(os.path.join(RAW, 'interactions_small/train/0000.parquet'),
                            columns=['user_id', 'video_id', 'watch_ratio'])
    small['watch_ratio'] = pd.to_numeric(small['watch_ratio'], errors='coerce')
    small = small.dropna(subset=['watch_ratio']).copy()
    # KuaiRec 再发布版的 id 是字符串，全项目统一成 int64，避免 parquet/npz 索引错位
    small['user_id'] = small['user_id'].astype('int64')
    small['video_id'] = small['video_id'].astype('int64')
    cap['video_id'] = cap['video_id'].astype('int64')
    small['label'] = (small['watch_ratio'] >= POS_RATIO).astype('int8')

    users = sorted(small['user_id'].unique())
    vids = sorted(small['video_id'].unique())
    print('users=%d items=%d pairs=%d' % (len(users), len(vids), len(small)))
    print('positive rate=%.4f' % small['label'].mean())
    print('cells / (users*items)=%.4f' % (len(small) / (len(users) * len(vids))))

    # ── 用户内 10% 留出 ──
    rng = np.random.default_rng(SEED)
    test_mask = np.zeros(len(small), dtype=bool)
    for _, sub in small.groupby('user_id', sort=False):
        idx = sub.index.values
        k = max(1, int(round(TEST_FRAC * len(idx))))
        test_mask[rng.choice(idx, size=k, replace=False)] = True
    train = small[~test_mask]
    test = small[test_mask]
    print('train pairs=%d (pos=%.4f)  test pairs=%d (pos=%.4f)'
          % (len(train), train['label'].mean(), len(test), test['label'].mean()))
    print('train users=%d test users=%d'
          % (train['user_id'].nunique(), test['user_id'].nunique()))

    # ── 物品表：类目 + 文案 ──
    item_meta = cap[['video_id', 'first_level_category_name', 'caption']].copy()
    item_meta = item_meta[item_meta['video_id'].isin(set(vids))]
    item_meta = item_meta.drop_duplicates('video_id').set_index('video_id')
    print('items with caption/category: %d / %d' % (len(item_meta), len(vids)))
    print('pool categories=%d' % item_meta['first_level_category_name'].nunique())

    os.makedirs(OUT, exist_ok=True)
    item_meta.reset_index().to_parquet(os.path.join(OUT, 'items.parquet'), index=False)
    train[['user_id', 'video_id', 'label']].to_parquet(
        os.path.join(OUT, 'train.parquet'), index=False)
    test[['user_id', 'video_id', 'label']].to_parquet(
        os.path.join(OUT, 'test.parquet'), index=False)
    print('saved: items/train/test parquet ->', OUT)


if __name__ == '__main__':
    main()
