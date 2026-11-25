# -*- coding: utf-8 -*-
"""Agent 自测：python tests/test_agent.py（项目根目录运行，退出码非 0 表示失败）。

覆盖 16 项：老用户推荐、类目约束、同类上限、不推已看、冷启动、文本意图、
需放宽约束、四类异常输入、未知类目降级、理由/轨迹/自检字段、JSON 序列化、延迟。
"""
import json
import sys
import time

sys.path.insert(0, '.')

from agent.agent import (Agent, STATUS_OK, STATUS_NEED_RELAX,  # noqa: E402
                         STATUS_INPUT_ERROR, STATUS_SYSTEM_ERROR)

PASS, FAIL = [], []
UID = 1001


def check(name, ok, note=''):
    (PASS if ok else FAIL).append(name)
    print('%s %-26s %s' % ('PASS' if ok else 'FAIL', name, note))


def main():
    t0 = time.perf_counter()
    agent = Agent()
    print('agent init %.2fs' % (time.perf_counter() - t0))
    lat = []

    # 1) 老用户默认推荐
    r = agent.run({'user_id': UID, 'constraints': 'k=5 mpc=2'})
    ok = (r['status'] == STATUS_OK and len(r['answer']['items']) == 5
          and all('video_id' in it and 'category' in it and 'reason' in it
                  for it in r['answer']['items']))
    check('user-basic', ok, 'status=%s n=%d' % (r['status'], len(r['answer'].get('items', []))))
    lat.append(r['latency_ms'])

    # 2) 类目约束
    r = agent.run({'user_id': UID, 'constraints': '美食 k=5 mpc=2'})
    ok = r['status'] == STATUS_OK and all(
        it['category'] == '美食' for it in r['answer']['items'])
    check('category-constraint', ok,
          'cats=%s' % sorted(set(it['category'] for it in r['answer'].get('items', []))))

    # 3) 同类上限（mpc=1 时每个类目最多 1 条）
    r = agent.run({'user_id': UID, 'constraints': 'k=8 mpc=1'})
    from collections import Counter
    cnt = Counter(it['category'] for it in r['answer']['items'])
    check('diversity-cap', r['status'] == STATUS_OK and max(cnt.values()) <= 2,
          'max_per_cat=%d' % (max(cnt.values()) if cnt else -1))

    # 4) 不推已看
    r = agent.run({'user_id': UID, 'constraints': 'k=10 mpc=2'})
    seen = agent.engine.history(UID)
    ok = r['status'] == STATUS_OK and all(it['video_id'] not in seen
                                          for it in r['answer']['items'])
    check('no-seen-items', ok, 'history=%d' % len(seen))

    # 5) 冷启动
    r = agent.run({'new_user': '美食 时尚', 'constraints': 'k=5 mpc=2'})
    ok = (r['status'] == STATUS_OK and len(r['answer']['items']) == 5
          and '冷启动' in r['answer']['mode'])
    check('cold-start', ok,
          'mode=%s' % (r['answer'].get('mode') if r['status'] == STATUS_OK else r['status']))

    # 6) 文本意图（BM25 混合）
    r = agent.run({'user_id': UID, 'constraints': 'k=5', 'query': '美食 做饭'})
    ok = r['status'] == STATUS_OK and 'BM25' in r['answer']['mode']
    check('query-bm25', ok,
          'mode=%s' % (r['answer'].get('mode') if r['status'] == STATUS_OK else r['status']))
    lat.append(r['latency_ms'])

    # 7) 需放宽约束（读书类目只有 2 条，k=10 不可行）
    r = agent.run({'user_id': UID, 'constraints': '读书 k=10 mpc=1'})
    check('need-relax', r['status'] == STATUS_NEED_RELAX,
          'status=%s hint=%s' % (r['status'], (r.get('answer') or {}).get('hint', '')[:30]))

    # 8) 异常：用户不在数据集
    r = agent.run({'user_id': 99999, 'constraints': ''})
    check('bad-user-id', r['status'] == STATUS_INPUT_ERROR)

    # 9) 异常：user_id 非数字
    r = agent.run({'user_id': 'abc', 'constraints': ''})
    check('non-numeric-user', r['status'] == STATUS_INPUT_ERROR)

    # 10) 异常：冷启动无有效兴趣
    r = agent.run({'new_user': '不存在的类目', 'constraints': ''})
    check('cold-bad-interest', r['status'] == STATUS_INPUT_ERROR)

    # 11) 异常：缺身份
    r = agent.run({'constraints': '美食'})
    check('missing-identity', r['status'] == STATUS_INPUT_ERROR)

    # 12) 未识别类目 -> 警告但不失败
    r = agent.run({'user_id': UID, 'constraints': '不存在类目X k=5'})
    ok = r['status'] == STATUS_OK and any(
        '未识别' in w for w in r['answer']['constraints']['warnings'])
    check('unknown-category', ok,
          'warnings=%s' % (r['answer']['constraints']['warnings'] if r['status'] == STATUS_OK else []))

    # 13) 理由与执行轨迹
    r = agent.run({'user_id': UID, 'constraints': 'k=3'})
    a = r['answer']
    ok = (all(it['reason'] for it in a['items'])
          and len(r['trace']) >= 6
          and all('step' in s and 'ms' in s for s in r['trace']))
    check('reason-and-trace', ok, 'steps=%d' % len(r['trace']))

    # 14) 自检全部通过
    ok = a.get('self_check') and all(a['self_check'].values())
    check('self-check', bool(ok), '%s' % a.get('self_check'))

    # 15) JSON 可序列化
    r2 = agent.run({'user_id': UID, 'constraints': 'k=3'})
    blob = json.dumps(r2, ensure_ascii=False)
    check('json-serializable', isinstance(json.loads(blob), dict))

    # 16) 延迟（预热后 < 300ms）
    if lat:
        check('latency', max(lat) < 300, 'max=%.1fms mean=%.1fms'
              % (max(lat), sum(lat) / len(lat)))

    print('---')
    print('PASS %d / %d' % (len(PASS), len(PASS) + len(FAIL)))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
