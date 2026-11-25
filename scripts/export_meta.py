# -*- coding: utf-8 -*-
"""导出文本元数据为 JSON（类目与文案），供 Agent 与推理侧读取。
单独成文件以保持每个脚本单一写入点。
输出：data/processed/meta.json
"""
import json
import sys

import pandas as pd

sys.path.insert(0, 'scripts')
import rec_common as rc  # noqa: E402


def main():
    items_df, train_df, _ = rc.load_processed()
    users, vids, _, _ = rc.build_index(train_df, items_df)
    cats = rc.item_categories(items_df, vids)
    caps = items_df.set_index('video_id')['caption'].to_dict()
    meta = {
        'users': [int(u) for u in users],
        'vids': [int(v) for v in vids],
        'item_cat': {str(v): cats[v] for v in vids},
        'item_caption': {str(v): str(caps.get(v, ''))[:60] for v in vids},
    }
    with open('data/processed/meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)
    print('saved: data/processed/meta.json (users=%d items=%d)'
          % (len(users), len(vids)))


if __name__ == '__main__':
    main()
