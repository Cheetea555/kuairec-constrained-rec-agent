# -*- coding: utf-8 -*-
"""评估侧图表：模型对比、消融、混合权重、冷启动、训练曲线。

读取 results/eval_summary.json 与 results/mf_history.json，
保证图与报告表格来自同一份数字。

输出（results/）：
  fig_models.png     六种排序器的 AUC / HR@10 / NDCG@10
  fig_ablation.png   多样性重排的消融（准确率 vs 多样性）
  fig_hybrid.png     BM25 混合权重扫描（A 半选权重，B 半验证）
  fig_cold.png       冷启动三种策略
  fig_mf_train.png   BPR 训练损失与留出 HR 曲线
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.dpi'] = 170

BLUE, ORANGE, GREEN, GRAY, RED = '#2b6cb0', '#dd6b20', '#38a169', '#a0aec0', '#c53030'
OUT = 'results'

with open(os.path.join(OUT, 'eval_summary.json'), encoding='utf-8') as f:
    S = json.load(f)

ORDER = [('random', '随机'), ('popular', '全局热门'), ('itemcf', 'ItemCF'),
         ('bm25', 'BM25 文本'), ('mf', 'BPR-MF'), ('mf_div', 'MF+多样性')]


def m(name, key):
    return S['models'][name][key]['mean']


def fig_models():
    names = [n for _, n in ORDER]
    auc = [m(k, 'auc') for k, _ in ORDER]
    hr = [m(k, 'hr') for k, _ in ORDER]
    nd = [m(k, 'ndcg') for k, _ in ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0))
    panels = [(auc, 'AUC', BLUE, 0.5), (hr, 'HR@10', ORANGE, 0),
              (nd, 'NDCG@10', GREEN, 0)]
    for ax, (vals, title, c, base) in zip(axes, panels):
        bars = ax.bar(names, vals, color=c)
        ax.tick_params(axis='x', rotation=42, labelsize=7)
        lo = 0.0 if base == 0 else float(min(vals)) * 0.85
        ax.set_ylim(lo, max(vals) * 1.15)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, '%.3f' % v,
                    ha='center', va='bottom', fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_models.png'))
    plt.close(fig)


def fig_ablation():
    keys = [('mf', '纯 MF Top-10'), ('mf_div', 'MF+同类上限重排')]
    hr = [m(k, 'hr') for k, _ in keys]
    cov = [S['coverage'][k]['catalog_coverage'] * 100 for k, _ in keys]
    catcov = [S['coverage'][k]['cat_coverage'] / 38 * 100 for k, _ in keys]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.9))
    for ax, vals, title, c in [(axes[0], hr, 'HR@10', BLUE),
                               (axes[1], cov, '目录覆盖率 (%)', GREEN),
                               (axes[2], catcov, '类目覆盖 (%)', ORANGE)]:
        bars = ax.bar([n for _, n in keys], vals, color=c, width=0.55)
        ax.tick_params(axis='x', labelsize=7)
        ax.set_ylim(0, max(vals) * 1.25 or 1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, '%.2f' % v,
                    ha='center', va='bottom', fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_ablation.png'))
    plt.close(fig)


def fig_hybrid():
    sw = S['hybrid_sweep_halfA']
    ws = [x['w'] for x in sw]
    hr = [x['hr'] for x in sw]
    best = S.get('hybrid_best_w', 0.0)
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.plot(ws, hr, 'o-', color=BLUE, label='A 半（选权重）')
    ax.axvline(best, color=ORANGE, ls='--', lw=1.2, label='选定 w=%.2f' % best)
    b = S.get('hybrid_halfB', {}).get('hr', {}).get('mean')
    mf = S.get('mf_halfB', {}).get('hr', {}).get('mean')
    if b is not None:
        ax.scatter([best], [b], color=GREEN, zorder=5, label='B 半验证 %.4f' % b)
    if mf is not None:
        ax.axhline(mf, color=GRAY, ls=':', lw=1, label='B 半纯 MF %.4f' % mf)
    ax.set_xlabel('BM25 混合权重 w')
    ax.set_ylabel('HR@10')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_hybrid.png'))
    plt.close(fig)


def fig_cold():
    c = S['cold']
    keys = [('cat_pop', '类目内热门'), ('cluster_only', '纯簇内热度'),
            ('blend', '混合(alpha=%.2f)' % c['alpha'])]
    vals = [c[k]['hr']['mean'] for k, _ in keys]
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    bars = ax.bar([n for _, n in keys], vals, color=[BLUE, ORANGE, GREEN],
                  width=0.55)
    ax.set_ylim(0, max(vals) * 1.25 or 0.01)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, '%.4f' % v,
                ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_cold.png'))
    plt.close(fig)


def fig_mf_train():
    with open(os.path.join(OUT, 'mf_history.json'), encoding='utf-8') as f:
        H = json.load(f)
    hist = H['history']
    ep = [h['epoch'] for h in hist]
    loss = [h['loss'] for h in hist]
    pts = [(h['epoch'], h['hr']) for h in hist if 'hr' in h]
    fig, ax1 = plt.subplots(figsize=(6.2, 3.0))
    ax1.plot(ep, loss, color=BLUE, lw=1.4, label='训练损失')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('BPR 损失', color=BLUE)
    ax1.tick_params(axis='y', labelcolor=BLUE)
    ax2 = ax1.twinx()
    ax2.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', color=ORANGE,
             ms=3, lw=1.2, label='留出 HR@10')
    ax2.set_ylabel('留出 HR@10', color=ORANGE)
    ax2.tick_params(axis='y', labelcolor=ORANGE)
    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_mf_train.png'))
    plt.close(fig)


if __name__ == '__main__':
    fig_models()
    fig_ablation()
    fig_hybrid()
    fig_cold()
    fig_mf_train()
    print('saved: fig_models/fig_ablation/fig_hybrid/fig_cold/fig_mf_train ->', OUT)
