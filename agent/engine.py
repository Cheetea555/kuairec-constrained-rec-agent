# -*- coding: utf-8 -*-
"""推荐引擎：加载模型产物，完成 召回→打分→多样性重排→理由生成。

职责边界：
  - engine 只做“算分与选品”，不做输入解析与状态编排（那是 agent.py）。
  - 所有模型懒加载，缺失时抛 FileNotFoundError 由 Agent 转成 SYSTEM_ERROR。
  - 路径一律相对项目根解析，从任何工作目录启动 Web/CLI 都能跑。

评分策略（strategy）：
  mf       BPR 矩阵分解（有历史的用户）
  hybrid   MF + BM25 文本意图（给了 query 时）
  bm25     仅文本检索（无历史 + 有 query）
  pop      全局热门（兜底基线）
  cf       ItemCF（课程协同过滤基线，用于对照实验）
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, 'data', 'processed')
MODELS = os.path.join(ROOT, 'models')

CONF_FLAT = 1e-6      # 分数极差小于此值视为“区分度不足”
CONF_MIN_POOL = 3     # 候选池小于 k 的倍数视为“低置信”


class RecEngine:
    def __init__(self):
        with open(os.path.join(PROC, 'meta.json'), encoding='utf-8') as f:
            self.meta = json.load(f)
        self.users = [int(u) for u in self.meta['users']]
        self.vids = [int(v) for v in self.meta['vids']]
        self.v2idx = {v: k for k, v in enumerate(self.vids)}
        self.u2idx = {u: k for k, u in enumerate(self.users)}
        self.cats = {int(k): v for k, v in self.meta['item_cat'].items()}
        self.captions = {int(k): v for k, v in self.meta['item_caption'].items()}
        self._mf = self._cf = self._pop = self._km = self._bm25 = None
        self._hist = None
        self._hist_cat = None

    # ── 模型加载（懒） ──
    def _npz(self, name, hint):
        path = os.path.join(MODELS, name)
        if not os.path.exists(path):
            raise FileNotFoundError('未找到 models/%s，请先运行 %s' % (name, hint))
        return np.load(path, allow_pickle=False)

    def mf(self):
        if self._mf is None:
            self._mf = self._npz('mf.npz', 'python scripts/train_mf.py')
        return self._mf

    def itemcf(self):
        if self._cf is None:
            self._cf = self._npz('itemcf.npz', 'python scripts/train_cf.py')
        return self._cf

    def popularity(self):
        if self._pop is None:
            self._pop = self._npz('popularity.npz', 'python scripts/train_cf.py')
        return self._pop

    def kmeans(self):
        if self._km is None:
            path = os.path.join(MODELS, 'kmeans.joblib')
            if not os.path.exists(path):
                raise FileNotFoundError('未找到 models/kmeans.joblib，'
                                        '请先运行 python scripts/train_kmeans.py')
            import joblib
            self._km = joblib.load(path)
        return self._km

    def bm25(self):
        if self._bm25 is None:
            from agent.retrieval import get_index
            self._bm25 = get_index(os.path.join(PROC, 'meta.json'))
        return self._bm25

    # ── 历史（用布尔矩阵一次性建好，避免 334 万条逐条建 set 的慢循环） ──
    def _load_history(self):
        if self._hist is not None:
            return
        d = np.load(os.path.join(PROC, 'dataset.npz'), allow_pickle=False)
        n_u, n_i = len(self.users), len(self.vids)
        self._seen = np.zeros((n_u, n_i), dtype=bool)
        self._seen[d['tr_u'], d['tr_i']] = True
        self._cat_names = sorted(set(self.cats.values()))
        self._cat_pos = {c: k for k, c in enumerate(self._cat_names)}
        self._item_cat = np.array([self._cat_pos[self.cats[v]] for v in self.vids])
        self._hist = {}
        self._hist_cat = {}

    def has_history(self, user_id):
        self._load_history()
        ui = self.u2idx.get(int(user_id))
        return bool(ui is not None and self._seen[ui].any())

    def history(self, user_id):
        """该用户训练期已观看的 video_id 集合（懒生成并缓存）。"""
        self._load_history()
        ui = self.u2idx.get(int(user_id))
        if ui is None:
            return set()
        if ui not in self._hist:
            self._hist[ui] = {self.vids[i] for i in np.flatnonzero(self._seen[ui])}
        return self._hist[ui]

    def cat_share(self, user_id):
        """该用户训练期历史的类目占比（推荐理由用）。"""
        self._load_history()
        ui = self.u2idx.get(int(user_id))
        if ui is None:
            return {}
        if ui not in self._hist_cat:
            cnt = np.bincount(self._item_cat[self._seen[ui]],
                              minlength=len(self._cat_names)).astype(float)
            tot = cnt.sum() or 1.0
            self._hist_cat[ui] = {self._cat_names[k]: cnt[k] / tot
                                  for k in np.flatnonzero(cnt)}
        return self._hist_cat[ui]

    # ── 打分器 ──
    def score_mf(self, user_id, candidate_vids):
        m = self.mf()
        ui = self.u2idx.get(int(user_id))
        if ui is None:
            return {v: 0.0 for v in candidate_vids}
        P = m['P'][ui].astype(np.float64)
        idx = np.array([self.v2idx[v] for v in candidate_vids])
        s = m['Q'][idx].astype(np.float64) @ P
        return {v: float(x) for v, x in zip(candidate_vids, s)}

    def score_pop(self, candidate_vids):
        pop = self.popularity()['pop']
        idx = np.array([self.v2idx[v] for v in candidate_vids])
        return {v: float(x) for v, x in zip(candidate_vids, pop[idx])}

    def score_bm25(self, query, candidate_vids):
        idx = np.array([self.v2idx[v] for v in candidate_vids])
        s = self.bm25().scores(query)[idx]
        return {v: float(x) for v, x in zip(candidate_vids, s)}

    def score_cf(self, user_id, candidate_vids):
        sim = self.itemcf()['sim']
        hist = self.history(user_id)
        if not hist:
            return self.score_pop(candidate_vids)
        ci = np.array([self.v2idx[v] for v in candidate_vids])
        hi = np.array([self.v2idx[v] for v in hist])
        s = sim[np.ix_(ci, hi)].sum(1)
        return {v: float(x) for v, x in zip(candidate_vids, s)}

    def cluster_for(self, declared_cats):
        """把声明的兴趣类目映射到最近的 KMeans 簇；无有效类目返回 None。"""
        km = self.kmeans()
        cat_names = list(km['cat_names'])
        used = [c for c in declared_cats if c in cat_names]
        if not used:
            return None
        vec = np.zeros(len(cat_names))
        for c in used:
            vec[cat_names.index(c)] += 1.0
        return int(km['kmeans'].predict((vec / len(used)).reshape(1, -1))[0])

    def cold_scores(self, declared_cats, alpha=0.5):
        """冷启动打分向量：簇内热度与全局热门的 z-score 加权（默认各半）。

        只用簇内热度会在“人人都看热门”的全观测数据上过拟合到小众内容，
        与全局先验各占一半后，验证集上明显优于任一单独策略（见报告 5.5）。
        声明了类目时把类目外物品置为 -inf（用户明确说了想看什么）。
        """
        km = self.kmeans()
        kmv = [int(v) for v in km['vids']]
        cl = self.cluster_for(declared_cats)
        if cl is None:
            return km['popularity'].astype(np.float64), None
        aligned = np.zeros(len(self.vids), dtype=np.float64)
        for j, v in enumerate(kmv):
            aligned[self.v2idx[v]] = float(km['cluster_scores'][cl][j])
        gp = km['popularity'].astype(np.float64)
        score = alpha * _zscore_vec(aligned) + (1.0 - alpha) * _zscore_vec(gp)
        used = [c for c in declared_cats if c in km['cat_names']]
        if used:
            mask = np.array([self.cats.get(v) in used for v in self.vids])
            score = np.where(mask, score, -np.inf)
        return score, cl

    # ── 冷启动：声明兴趣 → KMeans 簇 → 簇内热门 ──
    def cold_candidates(self, declared_cats, top_n=300, alpha=0.5):
        """返回候选 video_id 列表（按冷启动打分排序）+ 说明 + 簇号。"""
        score, cl = self.cold_scores(declared_cats, alpha)
        if cl is None:
            order = np.argsort(-np.nan_to_num(score, neginf=-1e9))[:top_n]
            return [self.vids[i] for i in order], '全局热门（兴趣未映射到簇）', None
        order = np.argsort(-np.nan_to_num(score, neginf=-1e9))
        cand = [self.vids[i] for i in order][:top_n]
        return (cand, '兴趣簇 %d（兴趣=%s）' % (cl, '、'.join(declared_cats)), cl)

    # ── 主流程 ──
    def recommend(self, user_id, constraints, query=None, strategy='auto'):
        """返回 dict：status ∈ {OK, NEED_RELAX}；含 items/理由/覆盖/低置信标记。"""
        cats = constraints.categories
        top_k = constraints.top_k
        seen = self.history(user_id) if (constraints.exclude_seen and user_id is not None
                                        and int(user_id) in self.u2idx) else set()
        has_hist = user_id is not None and self.has_history(user_id)

        # 1) 召回：类目过滤 + 去重已看
        cand = self.vids
        if cats:
            cand = [v for v in cand if self.cats.get(v) in cats]
        if seen:
            cand = [v for v in cand if v not in seen]
        notes = []
        if len(cand) < top_k:
            return {'status': 'NEED_RELAX', 'n_candidates': len(cand),
                    'mode': '-',
                    'hint': '当前约束下候选只有 %d 条，不足 %d 条；请减少类目或提高同类上限'
                            % (len(cand), top_k)}

        # 2) 策略选择
        q = (query or '').strip()
        if strategy == 'auto':
            if q and has_hist:
                strategy = 'hybrid'
            elif q:
                strategy = 'bm25'
            elif has_hist:
                strategy = 'mf'
            else:
                strategy = 'pop'
        if strategy in ('mf', 'hybrid', 'cf') and not has_hist:
            notes.append('该用户训练期无观看历史，已回退为热门/文本召回')
            strategy = 'bm25' if q else 'pop'

        # 3) 打分
        parts = {}   # video_id -> {source: score}
        if strategy == 'pop':
            scores = self.score_pop(cand)
            source = '热门(全局)'
        elif strategy == 'cf':
            scores = self.score_cf(user_id, cand)
            source = 'ItemCF(协同过滤)'
        elif strategy == 'bm25':
            scores = self.score_bm25(q, cand)
            source = 'BM25 文本检索'
            parts = {v: {'bm25': s} for v, s in scores.items()}
            if not any(s > 0 for s in scores.values()):
                notes.append('文案未命中任何关键词，已回退热门排序')
                scores = self.score_pop(cand)
                source = '热门(全局，文本无命中)'
        else:  # mf / hybrid
            scores = self.score_mf(user_id, cand)
            source = 'BPR-MF(个性化)'
            parts = {v: {'mf': s} for v, s in scores.items()}
            if strategy == 'hybrid':
                bs = self.score_bm25(q, cand)
                zm = _zscore(scores)
                zb = _zscore(bs)
                from agent.retrieval import DEFAULT_W
                w = DEFAULT_W
                scores = {v: zm[v] + w * zb[v] for v in cand}
                for v in cand:
                    parts[v] = {'mf': round(zm[v], 4), 'bm25': round(zb[v], 4)}
                source = 'BPR-MF + BM25 混合(权重 %.2f)' % w

        # 4) 多样性重排：贪心，同类目达上限后跳过
        picked, cat_note = self._diversify(cand, scores, constraints)

        # 5) 低置信判断：候选池过小或分数几乎无区分
        svals = [scores[v] for v in picked]
        flat = (max(svals) - min(svals)) < CONF_FLAT if len(svals) > 1 else True
        low_conf = bool(flat or len(cand) < top_k * CONF_MIN_POOL)
        if low_conf:
            notes.append('分数区分度低或候选池偏小，建议人工复核')

        # 6) 理由
        terms = self.bm25().top_terms(q) if q else []
        items = []
        for v in picked:
            items.append({
                'video_id': int(v),
                'category': self.cats.get(v, '其他'),
                'caption': (self.captions.get(v, '') or '')[:40],
                'score': round(float(scores[v]), 4),
                'reason': self._reason(user_id, v, has_hist, q, terms,
                                       constraints, parts.get(v, {})),
                'score_parts': parts.get(v, {}),
            })

        return {
            'status': 'OK',
            'mode': source,
            'score_source': source,
            'constraint_note': cat_note,
            'n_candidates': len(cand),
            'category_coverage': sorted(set(i['category'] for i in items)),
            'low_confidence': low_conf,
            'notes': notes,
            'items': items,
        }

    # ── 多样性重排（老用户/冷启动共用） ──
    def _diversify(self, cand, scores, constraints):
        order = sorted(cand, key=lambda v: -scores[v])
        picked, cat_cnt = [], {}
        for v in order:
            c = self.cats.get(v, '其他')
            if cat_cnt.get(c, 0) >= constraints.max_per_cat:
                continue
            picked.append(v)
            cat_cnt[c] = cat_cnt.get(c, 0) + 1
            if len(picked) >= constraints.top_k:
                break
        note = ''
        if len(picked) < constraints.top_k:
            rest = [v for v in order if v not in set(picked)]
            n = min(constraints.top_k - len(picked), len(rest))
            picked += rest[:n]
            note = '（同类上限截断后回填 %d 条）' % n
        return picked, note

    # ── 冷启动主流程：声明兴趣 → 簇内热门 → 约束过滤 → 重排 ──
    def recommend_cold(self, declared_cats, constraints, query=None):
        cats = constraints.categories or declared_cats
        cand, cl_note, cluster = self.cold_candidates(cats)
        filtered = [v for v in cand if self.cats.get(v) in cats] if cats else list(cand)
        if len(filtered) >= constraints.top_k:
            cand = filtered
        if len(cand) < constraints.top_k:
            return {'status': 'NEED_RELAX', 'n_candidates': len(cand), 'mode': '-',
                    'hint': '冷启动候选只有 %d 条，不足 %d 条；请减少类目或降低条数'
                            % (len(cand), constraints.top_k)}
        scores = {v: 1.0 / (rank + 1) for rank, v in enumerate(cand)}
        if query:
            bs = self.score_bm25(query, cand)
            if any(s > 0 for s in bs.values()):
                zb = _zscore(bs)
                zr = _zscore(scores)
                scores = {v: 0.7 * zr[v] + 0.3 * zb[v] for v in cand}
        picked, cat_note = self._diversify(cand, scores, constraints)
        items = [{
            'video_id': int(v),
            'category': self.cats.get(v, '其他'),
            'caption': (self.captions.get(v, '') or '')[:40],
            'score': round(float(scores[v]), 4),
            'reason': '冷启动（%s）；簇内热门排序' % cl_note,
            'score_parts': {'cluster_rank': round(float(scores[v]), 4)},
        } for v in picked]
        return {
            'status': 'OK',
            'mode': '冷启动 KMeans 簇内热门',
            'score_source': 'KMeans 用户分群 + 簇内热门',
            'constraint_note': cat_note,
            'cluster': cluster,
            'cluster_note': cl_note,
            'n_candidates': len(cand),
            'category_coverage': sorted(set(i['category'] for i in items)),
            'low_confidence': len(cand) < constraints.top_k * CONF_MIN_POOL,
            'notes': ['无历史，基于声明兴趣聚类冷启动，建议冷启动结果做少量人工确认'],
            'items': items,
        }

    # ── 单条理由 ──
    def _reason(self, user_id, v, has_hist, query, terms, constraints, parts):
        cat = self.cats.get(v, '其他')
        seg = []
        if has_hist:
            share = self.cat_share(user_id).get(cat, 0.0)
            if share >= 0.03:
                seg.append('与你历史中「%s」类目偏好一致（占历史 %.0f%%）' % (cat, share * 100))
            if 'mf' in parts:
                seg.append('MF 打分 %.2f' % parts['mf'])
        if constraints.categories and cat in constraints.categories:
            seg.append('满足指定类目')
        if terms:
            seg.append('文案命中「%s」' % '、'.join(terms[:2]))
        if not has_hist:
            seg.append('冷启动：来自兴趣簇内热门')
        if not seg:
            seg.append('全局热门排序')
        return '；'.join(seg)


def _zscore_vec(v):
    v = np.asarray(v, dtype=np.float64)
    sd = float(v.std())
    return (v - v.mean()) / sd if sd > 1e-9 else np.zeros_like(v)


def _zscore(d):
    vals = np.array([d[k] for k in d], dtype=np.float64)
    mu, sd = vals.mean(), vals.std()
    if sd < 1e-9:
        return {k: 0.0 for k in d}
    return {k: float((d[k] - mu) / sd) for k in d}

