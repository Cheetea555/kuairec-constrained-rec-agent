# -*- coding: utf-8 -*-
"""BM25 文本检索组件（复用开源库 rank_bm25，Apache-2.0）。

解决的问题：类目约束只能表达“哪一类”，而用户常常说的是“想看点讲做饭的”
这类内容意图。本模块把自然语言需求检索为候选视频，交给 BPR-MF 在同一
候选池内排序，形成“召回（BM25/类目）→ 排序（MF）→ 重排（多样性）”结构。

工程取舍：
  - 中文按字符二元组（char-bigram）切词，不引入分词器依赖（jieba 需要额外
    词典与约 50MB 资源，且对短视频标题这类口语化短文本收益有限）。
  - 3327 条文案的索引在内存中构建约 0.2 秒，进程内单例缓存，不落盘，
    避免产物与 meta.json 版本不一致。
  - 相比向量检索：3k 条短文案上 BM25 无需训练/无 GPU/可解释（能给出命中词），
    embedding 方案需要约 100MB 权重，性价比不足（见报告 4.3 的对照实验）。
"""
import json
import re

import numpy as np
from rank_bm25 import BM25Okapi

DEFAULT_W = 0.15  # 混合打分中 BM25 的权重：A 半权重扫描显示 0~0.4 的差异
                  # 小于 1 个百分点，取 0.15 让小权重文本意图可见（报告 5.4）

_ASCII = re.compile(r'^[a-z0-9]+$')
_PUNCT = re.compile(r'[^\w\u4e00-\u9fff]+')
# 高频但无区分度的虚词/平台话术，切词后剔除
STOP = {
    '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一',
    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '看',
    '这个', '那个', '我们', '你们', '他们', '什么', '怎么', '可以', '一个',
    '快手', '热门', '官方', '感谢', '支持', '点赞', '关注', '视频', '作者',
}


def tokenize(text):
    """中文按字符二元组切词，英文/数字按空白切分。"""
    text = _PUNCT.sub(' ', str(text or '').lower()).strip()
    toks = []
    for word in text.split():
        if _ASCII.match(word):
            toks.append(word)
        else:
            if len(word) == 1:
                toks.append(word)
            else:
                toks.extend(word[i:i + 2] for i in range(len(word) - 1))
    return [t for t in toks if t not in STOP]


class BM25Index:
    """视频文案索引：query -> 候选 video_id 及分数。"""

    def __init__(self, meta_path='data/processed/meta.json'):
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        self.vids = [int(v) for v in meta['vids']]
        caps = meta['item_caption']
        self.cats = {int(k): v for k, v in meta['item_cat'].items()}
        corpus = [tokenize(caps.get(str(v), '')) for v in self.vids]
        self.bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)
        self.n_docs = len(corpus)

    def scores(self, query):
        """返回与 self.vids 等长的分数向量（无命中为 0）。"""
        q = tokenize(query)
        if not q or self.n_docs == 0:
            return np.zeros(self.n_docs, dtype=np.float64)
        s = np.asarray(self.bm25.get_scores(q), dtype=np.float64)
        s[~np.isfinite(s)] = 0.0
        return np.maximum(s, 0.0)

    def search(self, query, top_n=300):
        """返回按分数降序的 [(video_id, score)]，只保留正分。"""
        s = self.scores(query)
        idx = np.argsort(-s)
        out = [(self.vids[int(i)], float(s[i])) for i in idx[:top_n] if s[i] > 0]
        return out

    def top_terms(self, query, k=3):
        """给出命中的高区分度词（用于推荐理由，可解释）。"""
        q = tokenize(query)
        if not q:
            return []
        idf = self.bm25.idf
        ranked = sorted(set(q), key=lambda t: -idf.get(t, 0.0))
        return ranked[:k]


_index = None


def get_index(meta_path='data/processed/meta.json'):
    """进程内单例。"""
    global _index
    if _index is None:
        _index = BM25Index(meta_path)
    return _index
