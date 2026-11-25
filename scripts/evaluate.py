# -*- coding: utf-8 -*-
"""测试集评估：留出集上的排序质量 + 消融 + 冷启动 + 约束满足率 + 延迟。

评估协议（与报告 5.1 一致）
  数据：KuaiRec 小矩阵（1411 用户 × 3327 视频，99.6% 的格子有 watch_ratio）
  划分：每个用户随机留出 10% 的 (用户, 视频) 配对做测试，其余 90% 训练；
        训练/测试的正负标签都是已知的（positive = watch_ratio ≥ 0.5）
  测试集：340403 个正样本 + 127250 个负样本，1411 个用户全覆盖
  指标：
    AUC      每个用户的全部测试正样本 vs 全部测试负样本的排序 AUC，再对用户取均值
    HR@10    每条测试正样本在该用户全部测试负样本中的排名 ≤ 10 即命中
    NDCG@10  命中时的折损增益
    （HR/NDCG 的“随机”水平分别约为 10/(n_neg+1) 与对应折损，n_neg 中位数见报告）

防调参泄漏
  用户按下标奇偶分成 A/B 两半：A 半用于选 BM25 混合权重，B 半只用于报告。

输出
  results/eval_summary.json   全部指标（报告表格的唯一数据来源）
  results/eval_users.npz      逐用户指标（画图/复核）
  stdout 键=值 行（写入 logs/evaluate.log）
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rec_common as rc  # noqa: E402

TOPK = 10
SEED = 2026
N_CONSTRAINT = 300
N_COLD = 150
W_GRID = [0.0, 0.1, 0.25, 0.4, 0.6, 1.0]


# ─────────────────────────── 指标 ───────────────────────────
def summarize(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=np.float64)
    if len(v) == 0:
        return {'mean': None, 'ci95': None, 'n': 0}
    return {'mean': round(float(v.mean()), 5),
            'ci95': round(float(1.96 * v.std(ddof=1) / np.sqrt(len(v))), 5)
            if len(v) > 1 else None,
            'n': int(len(v))}


def topk(scores, mask_out, k=TOPK):
    s = scores.astype(np.float64, copy=True)
    if mask_out is not None:
        s[mask_out] = -np.inf
    kk = min(k, len(s) - 1)
    order = np.argpartition(-s, kk)[:kk]
    return order[np.argsort(-s[order])]


# ─────────────────────────── 数据装配 ───────────────────────────
def load_all():
    d = np.load('data/processed/dataset.npz', allow_pickle=False)
    users = [int(x) for x in d['users']]
    vids = [int(x) for x in d['vids']]
    n_u, n_i = len(users), len(vids)
    pos, neg = {}, {}
    for u, i in zip(d['te_u'], d['te_i']):
        pos.setdefault(users[int(u)], []).append(int(i))
    for u, i in zip(d['te_neg_u'], d['te_neg_i']):
        neg.setdefault(users[int(u)], []).append(int(i))
    pos = {u: np.array(sorted(v)) for u, v in pos.items()}
    neg = {u: np.array(sorted(v)) for u, v in neg.items()}
    train_pos = np.zeros((n_u, n_i), dtype=bool)
    train_pos[d['tr_u'], d['tr_i']] = True
    train_neg = np.zeros((n_u, n_i), dtype=bool)
    train_neg[d['tr_neg_u'], d['tr_neg_i']] = True
    items = pd.read_parquet('data/processed/items.parquet')
    items['video_id'] = items['video_id'].astype('int64')
    cat_of = items.set_index('video_id')['first_level_category_name'].to_dict()
    item_cat = np.array([cat_of.get(int(v), '其他') for v in vids])
    caps = items.set_index('video_id')['caption'].fillna('').astype(str).to_dict()
    return dict(users=users, vids=vids, n_u=n_u, n_i=n_i, pos=pos, neg=neg,
                train_pos=train_pos, train_neg=train_neg,
                item_cat=item_cat, cat_names=sorted(set(item_cat)), caps=caps)


# ─────────────────────────── 评估器 ───────────────────────────
def eval_metrics(D, score_fn, uidx, name):
    """AUC / HR@10 / NDCG@10（正样本在该用户全部测试负样本中的名次）。"""
    aucs, hrs, nds = [], [], []
    for ui in uidx:
        uid = D['users'][ui]
        p = D['pos'].get(uid)
        n = D['neg'].get(uid)
        if p is None or n is None or len(p) == 0 or len(n) == 0:
            continue
        vec = score_fn(ui)
        sp = vec[p][:, None]
        sn = vec[n][None, :]
        gt = (sp > sn).sum(1)                 # 比该正样本分数低的负样本数
        eq = (sp == sn).sum(1)
        aucs.append(float((gt + 0.5 * eq).mean() / len(n)))
        # 名次 = 1 + 分数高于它的负样本数；并列按更差名次处理（保守）
        rank = 1 + len(n) - gt
        hrs.append(float((rank <= TOPK).mean()))
        gain = np.where(rank <= TOPK, 1.0 / np.log2(rank + 1), 0.0)
        nds.append(float(gain.mean()))
    res = {'auc': summarize(aucs), 'hr': summarize(hrs), 'ndcg': summarize(nds)}
    print('[留出] %-16s AUC=%.4f HR@10=%.4f NDCG@10=%.4f (n=%d)'
          % (name, res['auc']['mean'], res['hr']['mean'], res['ndcg']['mean'],
             res['auc']['n']))
    return res


def coverage(D, vec_of, uidx, name):
    """Agent 端可见的 Top-10（剔除训练期已看）的目录/类目覆盖。"""
    hits, cats = set(), []
    for ui in uidx:
        ordered = topk(vec_of(ui), D['train_pos'][ui])
        hits.update(ordered.tolist())
        cats.append(len(set(D['item_cat'][ordered].tolist())))
    res = {'catalog_coverage': round(len(hits) / D['n_i'], 4),
           'cat_coverage': round(float(np.mean(cats)), 3)}
    print('[覆盖] %-16s 目录=%.3f 类目=%.2f' % (name, res['catalog_coverage'],
                                             res['cat_coverage']))
    return res


def diversity_rerank(cand_idx_sorted, cats, k, mpc):
    picked, cnt = [], {}
    for i in cand_idx_sorted:
        c = cats[i]
        if cnt.get(c, 0) >= mpc:
            continue
        picked.append(i)
        cnt[c] = cnt.get(c, 0) + 1
        if len(picked) >= k:
            break
    return picked


def main():
    os.makedirs('results', exist_ok=True)
    D = load_all()
    n_u, n_i = D['n_u'], D['n_i']
    cats = D['item_cat']
    all_uidx = list(range(n_u))
    half_a = [i for i in all_uidx if i % 2 == 0]
    half_b = [i for i in all_uidx if i % 2 == 1]
    npos = sum(len(v) for v in D['pos'].values())
    nneg = sum(len(v) for v in D['neg'].values())
    neg_med = int(np.median([len(v) for v in D['neg'].values()]))
    print('users=%d items=%d test_pos=%d test_neg=%d (neg/user median=%d)'
          % (n_u, n_i, npos, nneg, neg_med))

    # ── 载入模型 ──
    mf = np.load('models/mf.npz', allow_pickle=False)
    P, Q = mf['P'].astype(np.float32), mf['Q'].astype(np.float32)
    pop = np.load('models/popularity.npz', allow_pickle=False)['pop'].astype(np.float32)
    sim = np.load('models/itemcf.npz', allow_pickle=False)['sim']
    S_mf = Q @ P.T
    S_cf = sim @ D['train_pos'].T.astype(np.float32)
    print('score matrices ready: mf=%s cf=%s' % (S_mf.shape, S_cf.shape))

    from agent.retrieval import get_index
    bm25 = get_index('data/processed/meta.json')
    qrng = np.random.default_rng(SEED + 1)
    query_of = {}
    for ui in all_uidx:
        hist_idx = np.flatnonzero(D['train_pos'][ui])
        pool = [D['caps'].get(int(D['vids'][i]), '') for i in hist_idx]
        pool = [c for c in pool if len(c) >= 6]
        if pool:
            pick = qrng.choice(len(pool), size=min(3, len(pool)), replace=False)
            query_of[ui] = ' '.join(pool[int(t)] for t in pick)[:120]

    _bm = {}

    def bm25_vec(ui):
        if ui not in _bm:
            q = query_of.get(ui, '')
            _bm[ui] = bm25.scores(q) if q else np.zeros(n_i)
        return _bm[ui]

    def zpos(v):
        out = np.zeros_like(v)
        m = v > 0
        if m.any():
            x = v[m]
            out[m] = (x - x.mean()) / (x.std() + 1e-9)
        return out

    def zcol(ui):
        v = S_mf[:, ui].astype(np.float64)
        return (v - v.mean()) / (v.std() + 1e-9)

    def hybrid_vec(ui, w):
        return zcol(ui) + w * zpos(bm25_vec(ui))

    _div = {}

    def mf_div_vec(ui):
        """对全量物品做同类上限重排（上限 2），得到完整排序向量用于消融。"""
        if ui in _div:
            return _div[ui]
        s = S_mf[:, ui].astype(np.float64).copy()
        s[D['train_pos'][ui]] = -np.inf
        order = np.argsort(-s)
        picked, cnt, rest = [], {}, []
        for i in order:
            c = cats[i]
            if cnt.get(c, 0) < 2:
                picked.append(int(i))
                cnt[c] = cnt.get(c, 0) + 1
            else:
                rest.append(int(i))
        full = np.array(picked + rest)
        v = np.empty(n_i)
        v[full] = np.arange(n_i, 0, -1)
        _div[ui] = v
        return v

    models = {
        'random': lambda ui: np.random.default_rng(SEED + ui).random(n_i),
        'popular': lambda ui: pop,
        'itemcf': lambda ui: S_cf[:, ui],
        'mf': lambda ui: S_mf[:, ui],
        'mf_div': mf_div_vec,
        'bm25': lambda ui: bm25_vec(ui),
    }

    out = {'dataset': {'users': n_u, 'items': n_i, 'train_pairs': int(D['train_pos'].sum()
                      + D['train_neg'].sum()), 'test_pos': int(npos),
                      'test_neg': int(nneg), 'neg_per_user_median': neg_med,
                      'pos_ratio': round(float(D['train_pos'].sum()) /
                                         (D['train_pos'].sum() + D['train_neg'].sum()), 4),
                      'categories': len(D['cat_names'])},
           'protocol': {'split': 'per-user 10% leave-pairs-out', 'seed': SEED,
                        'topk': TOPK, 'pos_ratio_threshold': 0.5}}

    out['models'] = {}
    out['coverage'] = {}
    for name, fn in models.items():
        out['models'][name] = eval_metrics(D, fn, all_uidx, name)
        out['coverage'][name] = coverage(D, fn, all_uidx, name)

    # ── BM25 内容一致性：Top-10 中属于历史主类目的比例 ──
    agree = []
    for ui in all_uidx:
        if ui not in query_of:
            continue
        hist_cats = Counter(cats[np.flatnonzero(D['train_pos'][ui])].tolist())
        main_cat = hist_cats.most_common(1)[0][0]
        top = topk(bm25_vec(ui), D['train_pos'][ui])
        agree.append(float(np.mean(cats[top] == main_cat)))
    out['bm25_main_cat_agree'] = round(float(np.mean(agree)), 4)
    print('bm25 main-cat agreement=%.4f' % out['bm25_main_cat_agree'])

    # ── 混合权重：A 半选，B 半报 ──
    sweep = []
    for w in W_GRID:
        r = eval_metrics(D, lambda ui: hybrid_vec(ui, w), half_a,
                         'hybrid_A w=%.2f' % w)
        sweep.append({'w': w, 'hr': r['hr']['mean'], 'auc': r['auc']['mean']})
    out['hybrid_sweep_halfA'] = sweep
    best_w = max(sweep, key=lambda x: x['hr'])['w']
    base = [s for s in sweep if s['w'] == 0.0][0]['hr']
    gain = max(s['hr'] for s in sweep) - base
    if gain < 0.002:
        best_w = 0.0
    out['hybrid_best_w'] = best_w
    out['hybrid_gain_halfA'] = round(gain, 4)
    print('hybrid best w=%.2f (A 半 HR 增益 %.4f)' % (best_w, gain))
    out['hybrid_halfB'] = eval_metrics(D, lambda ui: hybrid_vec(ui, best_w), half_b,
                                       'hybrid_B w=%.2f' % best_w)
    out['mf_halfB'] = eval_metrics(D, lambda ui: S_mf[:, ui], half_b, 'mf_B')
    out['hybrid_halfB_w015'] = eval_metrics(D, lambda ui: hybrid_vec(ui, 0.15), half_b, 'hybrid_B w=0.15')
    out['hybrid_full'] = eval_metrics(D, lambda ui: hybrid_vec(ui, best_w),
                                      all_uidx, 'hybrid_full')

    # ── 约束满足率 / 可行域 / 延迟 ──
    from agent.agent import Agent
    t0 = time.perf_counter()
    agent = Agent()
    print('agent init %.2fs' % (time.perf_counter() - t0))
    prand = np.random.default_rng(SEED + 2)
    sat = tot = need_relax = low_conf = 0
    cat_names = D['cat_names']
    for ui in prand.choice(all_uidx, N_CONSTRAINT, replace=False):
        target = cat_names[int(prand.integers(0, len(cat_names)))]
        r = agent.run({'user_id': D['users'][int(ui)],
                       'constraints': '%s k=5 mpc=2' % target})
        if r['status'] == 'NEED_RELAX':
            need_relax += 1
        elif r['status'] == 'OK':
            tot += 1
            items = r['answer']['items']
            low_conf += int(r['answer'].get('low_confidence', False))
            if all(it['category'] == target for it in items):
                sat += 1
    out['constraint'] = {'n': N_CONSTRAINT, 'ok': tot, 'need_relax': need_relax,
                         'satisfied': sat,
                         'sat_rate': round(sat / tot, 4) if tot else None,
                         'need_relax_rate': round(need_relax / N_CONSTRAINT, 4),
                         'low_conf_rate': round(low_conf / tot, 4) if tot else None}
    print('constraint sat=%.4f (n=%d) need_relax_rate=%.4f'
          % (out['constraint']['sat_rate'], tot, out['constraint']['need_relax_rate']))

    lat = []
    for ui in prand.choice(all_uidx, 200, replace=False):
        target = cat_names[int(prand.integers(0, len(cat_names)))]
        r = agent.run({'user_id': D['users'][int(ui)],
                       'constraints': '%s k=5 mpc=2' % target})
        if r['status'] == 'OK':
            lat.append(r['latency_ms'])
    lat_q = []
    for ui in prand.choice(all_uidx, 100, replace=False):
        r = agent.run({'user_id': D['users'][int(ui)], 'constraints': 'k=5',
                       'query': query_of.get(int(ui), '')})
        if r['status'] == 'OK':
            lat_q.append(r['latency_ms'])
    out['latency'] = {
        'plain_ms_p50': round(float(np.percentile(lat, 50)), 1),
        'plain_ms_p95': round(float(np.percentile(lat, 95)), 1),
        'query_ms_p50': round(float(np.percentile(lat_q, 50)), 1),
        'query_ms_p95': round(float(np.percentile(lat_q, 95)), 1),
        'n_plain': len(lat), 'n_query': len(lat_q)}
    print('latency plain p50=%.1f p95=%.1f; query p50=%.1f p95=%.1f ms'
          % (out['latency']['plain_ms_p50'], out['latency']['plain_ms_p95'],
             out['latency']['query_ms_p50'], out['latency']['query_ms_p95']))

    # ── 冷启动：KMeans 簇内热度 vs 全局热门 vs 类目热门（同一留出集） ──
    from agent.engine import RecEngine
    eng = RecEngine()
    crand = np.random.default_rng(SEED + 3)
    cold_users = list(crand.choice(all_uidx, N_COLD, replace=False))
    cold_a = cold_users[0::2]
    cold_b = cold_users[1::2]
    cat_idx = np.array([eng.cats.get(v, '其他') for v in eng.vids])

    def pair_hr_auc(vec, p, n, k=TOPK):
        sp = vec[p][:, None]
        sn = vec[n][None, :]
        gt = (sp > sn).sum(1)
        eq = (sp == sn).sum(1)
        auc = float((gt + 0.5 * eq).mean() / len(n))
        rank = 1 + len(n) - gt
        return float((rank <= k).mean()), auc

    def declared_of(ui):
        hist_cats = Counter(cats[np.flatnonzero(D['train_pos'][ui])].tolist())
        return [c for c, _ in hist_cats.most_common(2)]

    def cold_eval(alpha, uidx):
        """只在声明类目的物品上比较（与 Agent 冷启动的类目约束口径一致）。"""
        hrs, aucs = [], []
        for ui in uidx:
            uid = D['users'][int(ui)]
            p, n = D['pos'].get(uid), D['neg'].get(uid)
            if p is None or n is None or len(p) == 0 or len(n) == 0:
                continue
            declared = declared_of(ui)
            p = p[np.isin(cats[p], declared)]
            n = n[np.isin(cats[n], declared)]
            if len(p) == 0 or len(n) == 0:
                continue
            vec, _cl = eng.cold_scores(declared, alpha)
            hr, auc = pair_hr_auc(np.nan_to_num(vec, neginf=-1e9), p, n)
            hrs.append(hr)
            aucs.append(auc)
        return {'hr': summarize(hrs), 'auc': summarize(aucs)}

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    sweep = []
    for a in alphas:
        r = cold_eval(a, cold_a)
        sweep.append({'alpha': a, 'hr': r['hr']['mean'], 'auc': r['auc']['mean']})
        print('[冷启动A] alpha=%.2f HR=%.4f AUC=%.4f' % (a, r['hr']['mean'],
                                                       r['auc']['mean']))
    best_alpha = max(sweep, key=lambda x: x['hr'])['alpha']
    out['cold_sweep_halfA'] = sweep
    out['cold_best_alpha'] = best_alpha
    out['cold'] = {'cluster_only': cold_eval(1.0, cold_b),
                   'cat_pop': cold_eval(0.0, cold_b),
                   'blend': cold_eval(best_alpha, cold_b),
                   'n': len(cold_b), 'alpha': best_alpha}
    print('cold B 半 HR@10: 纯簇热度=%.4f 纯类目热门=%.4f 混合(alpha=%.2f)=%.4f'
          % (out['cold']['cluster_only']['hr']['mean'],
             out['cold']['cat_pop']['hr']['mean'],
             best_alpha, out['cold']['blend']['hr']['mean']))

    # ── 失败案例：留出集 HR 最低的 5 个用户 ──
    per_user = {}
    for ui in all_uidx:
        uid = D['users'][ui]
        p, n = D['pos'].get(uid), D['neg'].get(uid)
        if p is None or n is None or len(p) == 0 or len(n) == 0:
            continue
        vec = S_mf[:, ui]
        sp = vec[p][:, None]
        sn = vec[n][None, :]
        rank = 1 + len(n) - (sp > sn).sum(1)
        per_user[int(uid)] = float((rank <= TOPK).mean())
    worst = sorted(per_user.items(), key=lambda kv: kv[1])[:5]
    out['error_cases'] = [
        {'user_id': int(u), 'hr10': round(v, 4),
         'train_pos': int(D['train_pos'][D['users'].index(int(u))].sum()),
         'test_pos': int(len(D['pos'][int(u)])),
         'test_neg': int(len(D['neg'][int(u)]))} for u, v in worst]
    print('worst users:', out['error_cases'])

    np.savez('results/eval_users.npz',
             hr_mf=np.array([per_user[u] for u in per_user]),
             user_ids=np.array(list(per_user)))
    with open('results/eval_summary.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('saved: results/eval_summary.json, results/eval_users.npz')


def recall_like(vec, pos, neg, k=TOPK):
    """给定打分向量，返回该用户留出正样本在留出负样本中的 HR@k。"""
    sp = vec[pos][:, None]
    sn = vec[neg][None, :]
    rank = 1 + len(neg) - (sp > sn).sum(1)
    return float((rank <= k).mean())


if __name__ == '__main__':
    main()



