# 约束感知的短视频推荐 Agent（KuaiRec）

> “机器学习”课程期末项目 · 个人项目
> 作者：Cheetea555

## 在线演示

- **GitHub Pages（纯前端，浏览器内实时打分）**：<https://cheetea555.github.io/kuairec-constrained-rec-agent/docs/>
  页面把 BPR-MF 的隐向量与用户历史导出为静态数据，召回、打分、同类上限重排与理由生成都在浏览器里完成；
  BM25 文本意图与 K-Means 冷启动需要下面「快速开始」里的完整 Python 版。
- **本地完整版**：python web/app.py → <http://127.0.0.1:5000>

## 1. 这是什么

一个围绕“刷短视频”真实痛点做的推荐 Agent。用户给出**约束**（看哪些类目、要几条、
同类最多几条、是否排除已看）和可选的**内容意图**（如“做饭 美食”），Agent 在快手
公开数据集 KuaiRec 的视频池上完成：

**约束解析 → 身份识别（老用户/冷启动） → 候选召回 → 打分排序 → 同类上限重排 → 逐条理由 → 自检**

- 老用户：BPR 矩阵分解（隐式反馈）个性化排序；给了内容意图时与 BM25 文本检索混合打分
- 新用户（冷启动）：K-Means 用户兴趣分群（类目画像）+ 簇内热门
- 约束不可行时不硬凑：返回 `NEED_RELAX` 并说明放宽哪一项
- 每条推荐都给出可核验的依据（类目占比、MF 分数、文案命中词），并输出七步执行轨迹

## 2. 快速开始（Python 3.11+，纯 CPU，无需 GPU）

```bash
pip install -r requirements.txt

# 1) 下载数据（约 630MB parquet，HuggingFace 镜像）
python scripts/download_data.py

# 2) 数据准备：正样本定义、按时间划分、类目/文案元数据
python scripts/prepare_all.py

# 3) 训练全部模型：ItemCF/热门基线 → BPR-MF（numpy）→ K-Means 冷启动
python scripts/train_all.py            # 约 2.5 分钟（纯 CPU）

# 4) 测试集评估：模型对比、消融、冷启动、约束满足率、延迟
python scripts/evaluate.py             # 结果写入 results/eval_summary.json

# 5) 生成报告图表
python scripts/fig_data.py && python scripts/fig_eval.py

# 6) 自测（16 项：正常路径 + 四类异常 + 冷启动 + 自检）
python tests/test_agent.py

# 7) CLI 运行
python -m agent.agent --user 1001 --categories 美食 --top-k 5
python -m agent.agent --user 1001 --query "做饭 美食"
python -m agent.agent --new-user 美食 时尚 --top-k 5      # 冷启动
python -m agent.agent --user 1001 --json                   # 结构化输出（含轨迹）

# 8) Web 演示
python web/app.py        # 打开 http://127.0.0.1:5000
```

> 数据量大，`data/raw/` 不入库；模型产物随仓库提供，因此不下载数据也能直接跑
> CLI/Web/自测。若要复现论文数字，需要按上面 1→4 完整执行。

## 3. 目录结构

```
project/
  agent/                 # Agent 核心
    agent.py             #   主编排：七步工作流、状态机、自检、轨迹
    constraints.py       #   约束解析（非法值回退、类目模糊匹配）
    engine.py            #   召回/打分/多样性重排/理由/冷启动
    retrieval.py         #   BM25 文本召回（复用 rank_bm25）
  scripts/
    download_data.py     # 从 HuggingFace 下载 KuaiRec parquet 再发布版
    prepare_data.py      # 正样本定义、时间划分、元数据
    export_npz.py        # 导出训练/评估用的 npz 索引
    export_meta.py       # 导出类目/文案 JSON
    train_cf.py          # ItemCF + 全局热门基线
    train_mf.py          # BPR 矩阵分解（numpy + scipy 稀疏梯度）
    train_kmeans.py      # K-Means 用户分群（冷启动）
    train_all.py         # 一键训练
    evaluate.py          # 全量评估（写 results/eval_summary.json）
    experiments_mf.py    # BPR 超参数对照实验
    fig_data.py          # 数据图表
    fig_eval.py          # 评估图表
  web/                   # Flask 演示（表单 / 结果 / 执行轨迹）
  tests/test_agent.py    # 16 项自测
  data/processed/        # 划分后的 parquet + npz + meta.json
  models/                # mf.npz / itemcf.npz / popularity.npz / kmeans.joblib
  results/               # 评估 JSON、图表
  logs/                  # 训练与评估日志
```

## 4. 机器学习方法

| 环节 | 方法 | 作用 |
|---|---|---|
| 主排序 | BPR 矩阵分解（隐式反馈，负采样 + Adam，numpy/scipy 实现） | 个性化打分，替代“只按热度” |
| 候选召回 | 类目过滤 + BM25 文本检索（char-bigram 中文切词） | 支持“内容意图”，不依赖分词器 |
| 冷启动 | K-Means（用户类目画像，k=8）+ 簇内热门 | 无历史用户也能给出可解释结果 |
| 基线 | 随机、全局热门、ItemCF（余弦相似） | 证明 MF 的增量 |
| 多样性 | 贪心同类上限重排（MMR 简化） | 避免一次推荐全是同一类目 |
| 评估 | AUC / HR@10 / NDCG@10 / 目录覆盖率 / 约束满足率 / 延迟 | 兼顾准确率、多样性与可用性 |

### 主要结果（用户内 10% 留出，1411 用户，详见 `results/eval_summary.json`）

| 排序器 | AUC | HR@10 | NDCG@10 |
|---|---|---|---|
| 随机 | 0.500 | 12.6% | 5.7% |
| 全局热门 | 0.853 | 59.7% | 38.4% |
| ItemCF | 0.852 | 59.7% | 38.4% |
| BM25 文本 | 0.494 | 9.4% | 4.6% |
| **BPR-MF** | **0.860** | **61.8%** | **40.3%** |
| BPR-MF + 同类上限重排 | 0.847 | 59.3% | 32.6% |

- 负采样用“已知负样本”比均匀随机高 **8.4%** 的 HR@10（消融见 `results/mf_experiments.json`）。
- 同类上限重排：类目覆盖 7.45 → 7.95（+6.7%），HR@10 代价 −4.1%。
- 约束满足率 100%（n=243 可行请求）；不可行请求 19% 全部返回 `NEED_RELAX`。
- 端到端延迟 p50 3.2 ms / p95 4.7 ms；带文本意图 p50 30.5 ms。
- BM25 混合与 K-Means 冷启动的对照实验为**负结果**，报告 5.4/5.5 给出原因。


## 5. 数据与许可

- 数据集：**KuaiRec**（快手短视频推荐数据集，CC BY-SA 4.0），
  本项目使用 HuggingFace 用户 `valeriol29` 的 parquet 再发布版：
  <https://huggingface.co/datasets/valeriol29/KuaiRec>
  原始发布：kuairec.com / Zenodo / Gao et al., CIKM 2022。
- 抖音/快手视频文案与类目均来自该公开数据集，不含任何个人身份信息；
  项目仅用于课程研究，不作商业分发。
- 复用组件（rank_bm25、scikit-learn 等）的版本、许可证与用途见报告附录 A。
- 本项目代码：MIT（见 `LICENSE`）；数据集遵循其原始 CC BY-SA 4.0 许可。

## 6. 复现说明

- 随机性：BPR 训练与负采样固定种子 1234；评估抽样固定种子 2026。
- 模型与数据索引顺序强相关：`prepare_all.py` 与 `train_all.py` 必须成对执行，
  否则 `mf.npz` 的用户/视频下标会与 `dataset.npz` 错位。
- 已核验：完整重跑一次数据准备 + 训练 + 评估约 4 分钟（CPU），
  16 项自测全部通过，评估指标可从 `results/eval_summary.json` 逐项复核。
- 本次提交使用生成式 AI 辅助完成部分代码与报告文字，所有实验数字均由脚本重跑得到；
  具体参与环节见报告 7.3 与附录 C。


