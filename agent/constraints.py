# -*- coding: utf-8 -*-
"""约束解析器：把用户的自然语言/标签化输入解析为结构化约束。

支持的约束（与报告 2.1 工作流一致）：
  categories  想看的类目列表（KuaiRec 一级类目名）
  top_k       推荐条数（1..10）
  max_per_cat 同类目最多条数（多样性）
  exclude_seen 是否排除已看过的（默认是）

输入异常（未知类目、非法数值）不抛异常，而是记录 warnings 交由
Agent 决定是否降级处理——解析永远返回可用结果。
"""
import json
import os

META = 'data/processed/meta.json'
MIN_K, MAX_K = 1, 10
MIN_MPC, MAX_MPC = 1, 5


class Constraints:
    def __init__(self, categories=None, top_k=5, max_per_cat=2,
                 exclude_seen=True, warnings=None):
        self.categories = categories or []
        self.top_k = top_k
        self.max_per_cat = max_per_cat
        self.exclude_seen = exclude_seen
        self.warnings = warnings or []

    def to_dict(self):
        return {
            'categories': self.categories,
            'top_k': self.top_k,
            'max_per_cat': self.max_per_cat,
            'exclude_seen': self.exclude_seen,
            'warnings': self.warnings,
        }

    def __repr__(self):
        return 'Constraints(%s)' % json.dumps(self.to_dict(), ensure_ascii=False)


_cat_names = None


def category_vocab():
    """类目词表（数据驱动，来自 meta.json）。"""
    global _cat_names
    if _cat_names is None:
        with open(META, encoding='utf-8') as f:
            meta = json.load(f)
        _cat_names = sorted(set(meta['item_cat'].values()))
    return _cat_names


def parse(user_input):
    """解析约束。user_input 为 dict（Web/CLI 已分成字段）或字符串。

    字符串形式："剧情 搞笑" / "剧情,搞笑,k=8" / "剧情 k=8 mpc=3"
    """
    if isinstance(user_input, str):
        text = user_input.strip()
        cats = []
        top_k, mpc = None, None
        for tok in text.replace(',', ' ').replace('，', ' ').split():
            if tok.startswith('k='):
                top_k = tok[2:]
            elif tok.startswith('mpc='):
                mpc = tok[4:]
            else:
                cats.append(tok)
        raw = {'categories': cats, 'top_k': top_k, 'max_per_cat': mpc}
    else:
        raw = dict(user_input)

    warnings = []
    vocab = category_vocab()
    cats = []
    for c in (raw.get('categories') or []):
        name = str(c).strip()
        if name in vocab:
            if name not in cats:
                cats.append(name)
        else:
            # 模糊匹配：词表包含关系（如输入"美食"命中"美食"）
            hits = [v for v in vocab if name in v or v in name]
            if len(hits) == 1:
                if hits[0] not in cats:
                    cats.append(hits[0])
            else:
                warnings.append('未识别的类目: %s（可用类目见 /categories）' % name)
    if not cats:
        pass  # 不指定类目 = 全部类目，属于正常用法，不产生警告

    try:
        k = int(raw.get('top_k') or 5)
    except (TypeError, ValueError):
        k = 5
        warnings.append('top_k 非法，已回退为 5')
    if k < MIN_K or k > MAX_K:
        warnings.append('top_k=%d 超出 [%d,%d]，已截断' % (k, MIN_K, MAX_K))
        k = max(MIN_K, min(MAX_K, k))

    try:
        m = int(raw.get('max_per_cat') or 2)
    except (TypeError, ValueError):
        m = 2
        warnings.append('max_per_cat 非法，已回退为 2')
    if m < MIN_MPC or m > MAX_MPC:
        warnings.append('max_per_cat=%d 超出 [%d,%d]，已截断' % (m, MIN_MPC, MAX_MPC))
        m = max(MIN_MPC, min(MAX_MPC, m))

    return Constraints(categories=cats, top_k=k, max_per_cat=m,
                       exclude_seen=bool(raw.get('exclude_seen', True)),
                       warnings=warnings)

