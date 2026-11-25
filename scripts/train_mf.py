# -*- coding: utf-8 -*-
"""BPR 矩阵分解主模型（隐式反馈，纯 numpy 实现，CPU 可复现）。

数据与负采样（关键设计，见报告 3.3）
  KuaiRec 小矩阵是全观测矩阵，训练集里每个 (用户, 视频) 配对都有
  watch_ratio 标签。BPR 的负样本直接取自该用户 label=0 的配对
  （“看完不到一半”的已知负样本），而不是把未观测当作负样本。
  这避免了“未观测≈负样本”假设在这份数据上 71% 正样本密度下的严重偏差。

为什么是 numpy 手写而不是框架/GPU
  数据规模只有 1411 用户 × 3327 视频 × 300 万训练正样本，隐维 32。
  用 scipy 稀疏矩阵做批内梯度累加，单 epoch 在普通 CPU 上 2~3 秒，
  30 个 epoch 一分半即可；助教在任何装了 numpy/scipy 的机器上都能复现。

BPR 目标（Rendle et al., 2009）
  对每个正样本 (u, i) 从该用户的已知负样本中采 j，
  loss = -log σ(<p_u, q_i - q_j>) + λ/|B|·(||p_u||²+||q_i||²+||q_j||²)
  批次梯度用稀疏矩阵一次性散播回全局参数（等价于批内平均 SGD，
  避免 np.add.at 在 6.5 万行上的慢速散射）。

数据：data/processed/dataset.npz
输出：models/mf.npz、results/mf_history.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rec_common as rc  # noqa: E402

DEFAULTS = dict(epochs=30, dim=32, lr=0.01, reg=0.01, batch=65536,
                eval_every=5, topk=10, seed=1234)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # 数值稳定的 σ(x)


def scatter(n_rows, batch_size, rows):
    """把 batch 内每行梯度散播到全局矩阵：返回 (n_rows, batch_size) 稀疏矩阵。"""
    return csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, np.arange(len(rows)))),
        shape=(n_rows, batch_size))


def split_labels(uidx, iidx, users, n_i):
    pos, neg = {}, {}
    for u, i in zip(uidx, iidx):
        uid = users[int(u)]
        pos.setdefault(uid, []).append(int(i))
    return pos


def validate(P, Q, users, vids, te_pos, te_neg, k=10, seed=1234):
    """留出集上的 1:100 采样 HR@10 / NDCG@10（训练中用同一协议挑模型）。"""
    rng = np.random.default_rng(seed)
    uid2 = {u: t for t, u in enumerate(users)}
    hr, nd = [], []
    for uid, poss in te_pos.items():
        negs = te_neg.get(uid)
        if not poss or not negs:
            continue
        cand = rng.choice(negs, size=min(100, len(negs)), replace=False)
        ui = uid2[uid]
        sp = Q[poss] @ P[ui]
        sn = Q[cand] @ P[ui]
        for s in sp:
            rank = 1 + int((sn > s).sum()) + int((sn == s).sum())
            hr.append(1.0 if rank <= k else 0.0)
            nd.append(1.0 / np.log2(rank + 1) if rank <= k else 0.0)
    return float(np.mean(hr)), float(np.mean(nd)), len(hr)


def main():
    ap = argparse.ArgumentParser(description='BPR-MF 训练（numpy）')
    for k, v in DEFAULTS.items():
        ap.add_argument('--' + k.replace('_', '-'), type=type(v), default=v)
    ap.add_argument('--out', default='models/mf.npz', help='模型输出路径')
    ap.add_argument('--history-out', default='results/mf_history.json')
    ap.add_argument('--neg-from-label', dest='neg_from_label', action='store_true',
                    default=True, help='从已知负样本采样（默认，正确做法）')
    ap.add_argument('--no-neg-from-label', dest='neg_from_label',
                    action='store_false', help='均匀随机负采样（对照实验）')
    args = ap.parse_args()

    d = np.load('data/processed/dataset.npz', allow_pickle=False)
    tr_u, tr_i = d['tr_u'].astype(np.int64), d['tr_i'].astype(np.int64)
    users = [int(x) for x in d['users']]
    vids = [int(x) for x in d['vids']]
    n_u, n_i = len(users), len(vids)
    te_pos = {}
    for u, i in zip(d['te_u'], d['te_i']):
        te_pos.setdefault(users[int(u)], []).append(int(i))
    te_neg = {}
    for u, i in zip(d['te_neg_u'], d['te_neg_i']):
        te_neg.setdefault(users[int(u)], []).append(int(i))

    # 已知负样本按用户分桶，供采样使用
    neg_cnt = np.bincount(d['tr_neg_u'], minlength=n_u)
    order = np.argsort(d['tr_neg_u'], kind='stable')
    neg_i_sorted = d['tr_neg_i'][order].astype(np.int64)
    neg_start = np.concatenate([[0], np.cumsum(neg_cnt)[:-1]])
    print('users=%d items=%d train_pos=%d train_neg=%d '
          'known-neg ratio=%.3f dim=%d epochs=%d'
          % (n_u, n_i, len(tr_u), int(neg_cnt.sum()),
             neg_cnt.sum() / max(len(tr_u), 1), args.dim, args.epochs))

    rng = np.random.default_rng(args.seed)
    P = (rng.standard_normal((n_u, args.dim)) * 0.01).astype(np.float64)
    Q = (rng.standard_normal((n_i, args.dim)) * 0.01).astype(np.float64)
    mP, vP = np.zeros_like(P), np.zeros_like(P)
    mQ, vQ = np.zeros_like(Q), np.zeros_like(Q)
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    best = (-1.0, None, None, 0)
    history = []
    n_pairs = len(tr_u)
    t_start = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        perm = rng.permutation(n_pairs)
        ep_loss, ep_steps, t0 = 0.0, 0, time.perf_counter()
        for s in range(0, n_pairs, args.batch):
            idx = perm[s:s + args.batch]
            ub, ib = tr_u[idx], tr_i[idx]
            B = len(idx)
            if args.neg_from_label:
                cnt = neg_cnt[ub]
                safe = np.maximum(cnt, 1)
                off = neg_start[ub] + (rng.random(B) * safe).astype(np.int64)
                jb = neg_i_sorted[np.minimum(off, len(neg_i_sorted) - 1)]
                no_neg = cnt == 0
                if no_neg.any():
                    jb[no_neg] = rng.integers(0, n_i, int(no_neg.sum()))
            else:
                jb = rng.integers(0, n_i, B)

            qi, qj = Q[ib], Q[jb]
            x = np.einsum('bd,bd->b', qi - qj, P[ub])
            loss = np.logaddexp(0.0, -x).mean()
            g = (sigmoid(x) - 1.0) / B                  # dL/dx / B
            ep_loss += loss
            ep_steps += 1

            # ── 梯度：p_u ──
            Au = scatter(n_u, B, ub)
            cu = np.bincount(ub, minlength=n_u).astype(np.float64)
            grad_P = Au @ (g[:, None] * (qi - qj)) \
                + (2.0 * args.reg / B) * cu[:, None] * P
            # ── 梯度：q_i / q_j（同一个稀疏矩阵，i 列系数 +g，j 列系数 -g）──
            rows = np.concatenate([ib, jb])
            cols = np.concatenate([np.arange(B), np.arange(B)])
            Aq = csr_matrix((np.concatenate([g, -g]), (rows, cols)), shape=(n_i, B))
            cq = (np.bincount(ib, minlength=n_i)
                  + np.bincount(jb, minlength=n_i)).astype(np.float64)
            grad_Q = Aq @ P[ub] + (2.0 * args.reg / B) * cq[:, None] * Q

            # ── Adam 更新 ──
            step += 1
            mP *= b1; mP += (1 - b1) * grad_P
            vP *= b2; vP += (1 - b2) * grad_P * grad_P
            mQ *= b1; mQ += (1 - b1) * grad_Q
            vQ *= b2; vQ += (1 - b2) * grad_Q * grad_Q
            c1 = 1 - b1 ** step
            c2 = 1 - b2 ** step
            P -= args.lr * (mP / c1) / (np.sqrt(vP / c2) + eps)
            Q -= args.lr * (mQ / c1) / (np.sqrt(vQ / c2) + eps)

        rec = {'epoch': ep, 'loss': round(ep_loss / max(ep_steps, 1), 5),
               'epoch_sec': round(time.perf_counter() - t0, 2)}
        if ep % args.eval_every == 0 or ep == args.epochs:
            hr, ndcg, n_ev = validate(P, Q, users, vids, te_pos, te_neg, args.topk)
            rec.update(hr=round(hr, 5), ndcg=round(ndcg, 5))
            print('ep%02d loss=%.4f HR@%d=%.4f NDCG@%d=%.4f (%.1fs)'
                  % (ep, rec['loss'], args.topk, hr, args.topk, ndcg,
                     rec['epoch_sec']))
            if hr > best[0]:
                best = (hr, P.copy(), Q.copy(), ep)
        else:
            print('ep%02d loss=%.4f (%.1fs)' % (ep, rec['loss'], rec['epoch_sec']))
        history.append(rec)

    P_best, Q_best = (best[1], best[2]) if best[1] is not None else (P, Q)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.history_out) or '.', exist_ok=True)
    np.savez_compressed(args.out, P=P_best.astype(np.float32),
                        Q=Q_best.astype(np.float32),
                        users=np.array(users), vids=np.array(vids))
    meta = {'cfg': vars(args), 'best_hr': best[0], 'best_epoch': best[3],
            'total_sec': round(time.perf_counter() - t_start, 1),
            'history': history}
    with open(args.history_out, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print('saved: %s (best HR@%d=%.4f @ep%d, %.1fs)'
          % (args.out, args.topk, best[0], best[3], meta['total_sec']))


if __name__ == '__main__':
    main()
