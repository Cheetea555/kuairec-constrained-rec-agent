# -*- coding: utf-8 -*-
"""数据侧图表：观看比例分布、类目长尾、用户历史/测试规模。

输出（results/）：
  fig_watch_ratio.png   正样本阈值 0.5 的选择依据
  fig_cat_tail.png      38 个一级类目的物品数长尾（决定“小类目不可行”的设计）
  fig_user_scale.png    每用户训练历史与测试正样本数分布
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.dpi'] = 160

BLUE, ORANGE, GRAY = '#2b6cb0', '#dd6b20', '#a0aec0'
OUT = 'results'
os.makedirs(OUT, exist_ok=True)


def fig_watch_ratio():
    small = pd.read_parquet('data/raw/interactions_small/train/0000.parquet',
                            columns=['watch_ratio'])
    wr = pd.to_numeric(small['watch_ratio'], errors='coerce').dropna()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.hist(wr, bins=80, color=BLUE, alpha=0.85)
    ax.axvline(0.5, color=ORANGE, ls='--', lw=1.4)
    ax.set_yscale('log')
    ax.set_xlabel('watch_ratio（观看完成度）')
    ax.set_ylabel('交互条数（对数）')
    ax.annotate('正样本阈值 0.5\n（≥0.5 视为喜欢）', xy=(0.5, 4e5), xytext=(0.62, 8e5),
                fontsize=8, color=ORANGE,
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_watch_ratio.png'))
    plt.close(fig)


def fig_cat_tail():
    items = pd.read_parquet('data/processed/items.parquet')
    items['video_id'] = items['video_id'].astype('int64')
    cnt = items['first_level_category_name'].value_counts()
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    colors = [BLUE if v >= 10 else ORANGE for v in cnt.values]
    ax.bar(range(len(cnt)), cnt.values, color=colors)
    ax.set_xticks(range(len(cnt)))
    ax.set_xticklabels(cnt.index, rotation=75, fontsize=6.5)
    ax.set_ylabel('视频数')
    ax.axhline(10, color=GRAY, ls=':', lw=1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_cat_tail.png'))
    plt.close(fig)


def fig_user_scale():
    tr = pd.read_parquet('data/processed/train.parquet')
    te = pd.read_parquet('data/processed/test.parquet')
    tr = tr.assign(user_id=tr['user_id'].astype('int64'))
    te = te.assign(user_id=te['user_id'].astype('int64'))
    h = tr.groupby('user_id').size().values
    t = te.groupby('user_id').size().reindex(tr.groupby('user_id').size().index,
                                             fill_value=0).values
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].hist(h, bins=40, color=BLUE)
    axes[0].set_xlabel('条数')
    axes[0].set_ylabel('用户数')
    axes[1].hist(t, bins=40, color=ORANGE)
    axes[1].set_xlabel('条数')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_user_scale.png'))
    plt.close(fig)


if __name__ == '__main__':
    fig_watch_ratio()
    fig_cat_tail()
    fig_user_scale()
    print('saved: fig_watch_ratio.png, fig_cat_tail.png, fig_user_scale.png ->', OUT)
