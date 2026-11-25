# -*- coding: utf-8 -*-
"""冷启动：K-Means 用户兴趣分群 + 簇内热门。

思路：对有历史的用户，以“类目观看占比”向量做 K-Means 分群，
无历史的新用户按声明的兴趣类目映射到最近的簇，推荐该簇内热门视频。
这给 Agent 的 --new-user 分支提供课程方法支撑（聚类 + 热门先验）。

输出：models/kmeans.joblib（含 kmeans、簇质心、簇内热门榜、类目词表）
"""
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rec_common as rc  # noqa: E402

N_CLUSTERS = 8
SEED = 1234


def main():
    items_df, train_df, _ = rc.load_processed()
    # 用户兴趣画像只统计“喜欢”的配对
    train_df = train_df[train_df['label'] == 1]
    users, vids, user2idx, item2idx = rc.build_index(train_df, items_df)
    cats = rc.item_categories(items_df, vids)
    cat_names = sorted(set(cats.values()))
    cat_pos = {c: k for k, c in enumerate(cat_names)}
    print('users=%d items=%d categories=%d' % (len(users), len(vids), len(cat_names)))

    # 用户 × 类目 观看占比矩阵
    M = np.zeros((len(users), len(cat_names)), dtype=np.float64)
    v2c = [cat_pos[cats[v]] for v in vids]
    v2c = np.array(v2c)
    i_idx = train_df['video_id'].map(item2idx).values
    u_idx = train_df['user_id'].map(user2idx).values
    np.add.at(M, (u_idx, v2c[i_idx]), 1.0)
    row_sum = M.sum(1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    M = M / row_sum  # 类目占比（兴趣画像）

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    labels = km.fit_predict(M)
    sizes = np.bincount(labels, minlength=N_CLUSTERS)
    print('cluster sizes:', sizes.tolist())

    # 簇内热门：按簇内用户训练期观看次数
    popularity = np.zeros(len(vids))
    np.add.at(popularity, i_idx, 1.0)
    cluster_pop = []
    cluster_scores = np.zeros((N_CLUSTERS, len(vids)), dtype=np.float32)
    for c in range(N_CLUSTERS):
        u_in = np.where(labels == c)[0]
        pop_c = np.zeros(len(vids))
        sub = train_df[train_df['user_id'].map(user2idx).isin(u_in)]
        ii = sub['video_id'].map(item2idx).values
        np.add.at(pop_c, ii, 1.0)
        cluster_scores[c] = pop_c.astype(np.float32)
        order = np.argsort(-pop_c)[:200]
        cluster_pop.append([vids[t] for t in order])
        top_cats = [cats[vids[t]] for t in order[:10]]
        print('cluster %d: n=%d top_cats=%s' % (c, len(u_in), top_cats[:5]))

    os.makedirs('models', exist_ok=True)
    joblib.dump({
        'kmeans': km,
        'cat_names': cat_names,
        'cluster_pop': cluster_pop,
        'cluster_scores': cluster_scores,
        'users': users,
        'vids': vids,
        'popularity': popularity,
        'labels': labels,
    }, 'models/kmeans.joblib')
    print('saved: models/kmeans.joblib')


if __name__ == '__main__':
    main()
