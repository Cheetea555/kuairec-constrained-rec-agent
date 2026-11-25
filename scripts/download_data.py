# -*- coding: utf-8 -*-
"""下载数据集：valeriol29/KuaiRec（HuggingFace 上的 KuaiRec parquet 再发布版，
原始数据来自 kuairec.com / Zenodo，CC BY-SA 4.0）。

只取本项目需要的 4 张表：
  interactions_small  全观测小矩阵（1411 用户 × 3327 视频，用于评估）
  interactions_big    稀疏大矩阵（训练交互）
  item_caption_category  视频文案与类目标签
  users               用户画像
用法：
    python scripts/download_data.py
"""
import os
import sys

# 国内镜像，避免直连 HuggingFace 下载失败
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from huggingface_hub import snapshot_download

REPO_ID = 'valeriol29/KuaiRec'
PATTERNS = [
    'interactions_small/train/*.parquet',
    'interactions_big/train/*.parquet',
    'item_caption_category/train/*.parquet',
    'users/train/*.parquet',
]


def main():
    dest = os.path.join('data', 'raw')
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type='dataset',
        allow_patterns=PATTERNS,
        local_dir=dest,
        max_workers=8,
    )
    print('done ->', path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
