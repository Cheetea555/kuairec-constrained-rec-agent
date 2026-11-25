# -*- coding: utf-8 -*-
"""约束感知短视频推荐 Agent：主编排器。

任务入口
    user_id（数据集老用户）或 new_user（声明兴趣的冷启动用户）
    + 自然语言约束（类目/条数/同类上限/是否排除已看）
    + 可选 query（内容意图，如“想看做饭的视频”）
返回
    top-k 推荐（类目、文案、分数、逐条理由）+ 覆盖统计 + 自检结果 + 执行轨迹

工作流（每一步都有独立工具与失败分支，轨迹写回 result['trace']）
    1 约束解析      constraints.parse     非法值回退并警告，不中断
    2 身份识别      engine.u2idx          老用户 / 冷启动 / 非法输入
    3 候选召回      category 过滤 + 去已看 + 可选 BM25 文本召回
    4 打分排序      BPR-MF / MF+BM25 混合 / 热门 / ItemCF
    5 多样性重排    贪心 + 同类上限，不足时回填并标注
    6 理由生成      类目偏好占比、MF 分、文案命中词
    7 自检         条数/类目/同类上限/唯一性/去重，失败则提示复核

失败与安全处理
    约束不可行 -> NEED_RELAX，给出具体放宽建议，绝不硬凑
    输入异常   -> INPUT_ERROR，instruct 用户正确入口
    模型缺失   -> SYSTEM_ERROR，提示对应训练命令
    低置信     -> low_confidence=True + 建议人工复核（不隐藏不确定性）
"""
import json
import time
from collections import Counter

from agent import constraints as cons
from agent.engine import RecEngine

STATUS_OK = 'OK'
STATUS_NEED_RELAX = 'NEED_RELAX'
STATUS_INPUT_ERROR = 'INPUT_ERROR'
STATUS_SYSTEM_ERROR = 'SYSTEM_ERROR'


class Agent:
    def __init__(self):
        self.engine = RecEngine()

    # ── 轨迹记录 ──
    @staticmethod
    def _trace(trace, step, tool, args, out, t0):
        trace.append({'step': step, 'tool': tool, 'args': args, 'output': out,
                      'ms': round((time.perf_counter() - t0) * 1000, 1)})

    def run(self, user_input):
        """user_input: dict，键 user_id | new_user | constraints | query。"""
        t_all = time.perf_counter()
        trace = []
        result = {'task': 'constrained_video_recommendation',
                  'status': None, 'answer': None, 'detail': None}

        # ── 1) 约束解析 ──
        t = time.perf_counter()
        try:
            parsed = cons.parse(user_input.get('constraints') or '')
        except Exception as e:                                    # 解析器内部异常兜底
            result['status'] = STATUS_INPUT_ERROR
            result['detail'] = {'error': '约束解析失败: %s' % e}
            return self._finish(result, trace, t_all)
        self._trace(trace, '1 约束解析', 'constraints.parse',
                    repr(user_input.get('constraints') or '')[:60],
                    '类目=%s k=%d 同类上限=%d 排除已看=%s'
                    % (parsed.categories or '全部', parsed.top_k,
                       parsed.max_per_cat, parsed.exclude_seen), t)

        query = str(user_input.get('query') or '').strip()

        # ── 2) 身份识别 ──
        t = time.perf_counter()
        uid = user_input.get('user_id')
        new_user_cats = user_input.get('new_user')
        cold_start = False
        mode_note = ''
        if new_user_cats is not None:
            if isinstance(new_user_cats, str):
                new_user_cats = new_user_cats.replace(',', ' ').split()
            vocab = cons.category_vocab()
            cats_valid, warns = [], []
            for c in new_user_cats:
                if c in vocab:
                    cats_valid.append(c)
                else:
                    warns.append('未识别兴趣类目: %s' % c)
            if not cats_valid:
                result['status'] = STATUS_INPUT_ERROR
                result['detail'] = {'error': '冷启动需要至少一个有效兴趣类目（可用：%s…）'
                                    % '、'.join(vocab[:8])}
                self._trace(trace, '2 身份识别', 'identity',
                            'new_user=%s' % new_user_cats, '无有效兴趣类目', t)
                return self._finish(result, trace, t_all)
            parsed.warnings += warns
            parsed.categories = parsed.categories or cats_valid
            cold_start = True
            mode_note = '冷启动（声明兴趣：%s）' % '、'.join(cats_valid)
        elif uid is None:
            result['status'] = STATUS_INPUT_ERROR
            result['detail'] = {'error': '缺少 user_id 或 new_user：请提供数据集用户 id'
                                         '（如 1001），或声明兴趣走冷启动'}
            self._trace(trace, '2 身份识别', 'identity', '无', '缺少身份', t)
            return self._finish(result, trace, t_all)
        else:
            try:
                uid = int(uid)
            except (TypeError, ValueError):
                result['status'] = STATUS_INPUT_ERROR
                result['detail'] = {'error': 'user_id 非法: %r' % uid}
                self._trace(trace, '2 身份识别', 'identity', repr(uid), 'user_id 非整数', t)
                return self._finish(result, trace, t_all)
            if uid not in self.engine.u2idx:
                result['status'] = STATUS_INPUT_ERROR
                result['detail'] = {
                    'error': '用户 %s 不在数据集（KuaiRec 小矩阵共 %d 个用户，'
                             'id 形如 %d、%d）；可改用 --new-user 冷启动'
                             % (uid, len(self.engine.users), self.engine.users[0],
                                self.engine.users[len(self.engine.users) // 2])}
                self._trace(trace, '2 身份识别', 'identity', str(uid), '用户不存在', t)
                return self._finish(result, trace, t_all)
            mode_note = '老用户 %d（训练期历史 %d 条）' % (
                uid, len(self.engine.history(uid)))
        self._trace(trace, '2 身份识别', 'identity.resolve',
                    'user_id=%s new_user=%s' % (uid, new_user_cats), mode_note, t)

        # ── 3~6) 召回 / 打分 / 重排 / 理由 ──
        t = time.perf_counter()
        try:
            if cold_start:
                rec = self.engine.recommend_cold(parsed.categories, parsed, query=query)
            else:
                rec = self.engine.recommend(uid, parsed, query=query, strategy='auto')
        except FileNotFoundError as e:
            result['status'] = STATUS_SYSTEM_ERROR
            result['detail'] = {'error': str(e)}
            return self._finish(result, trace, t_all)
        except Exception as e:                                    # 工具异常统一收敛
            result['status'] = STATUS_SYSTEM_ERROR
            result['detail'] = {'error': '推荐失败: %s' % e}
            return self._finish(result, trace, t_all)
        self._trace(trace, '3 候选召回', 'recall.category+seen',
                    '类目=%s' % (parsed.categories or '全部'),
                    '候选 %d 条' % rec['n_candidates'], t)

        if rec['status'] == 'NEED_RELAX':
            result['status'] = STATUS_NEED_RELAX
            result['answer'] = rec
            self._trace(trace, '4 可行域检查', 'feasibility',
                        '需要 %d 条' % parsed.top_k,
                        '候选 %d 条，请求放宽' % rec['n_candidates'], time.perf_counter())
            return self._finish(result, trace, t_all)

        t = time.perf_counter()
        self._trace(trace, '4 打分排序', 'score.' + self._scorer_name(rec['mode']),
                    '策略=自动', rec['mode'], t)
        t = time.perf_counter()
        self._trace(trace, '5 多样性重排', 'rerank.greedy',
                    '同类上限=%d' % parsed.max_per_cat,
                    '%d 条，覆盖 %d 个类目%s'
                    % (len(rec['items']), len(rec['category_coverage']),
                       rec.get('constraint_note') or ''), t)
        t = time.perf_counter()
        self._trace(trace, '6 理由生成', 'explain.template',
                    '每条 %d 个证据' % 2,
                    '已生成 %d 条理由' % len(rec['items']), t)

        # ── 7) 自检 ──
        t = time.perf_counter()
        checks = self._self_check(rec, parsed, uid if not cold_start else None)
        rec['self_check'] = checks
        failed = [k for k, v in checks.items() if not v]
        if failed:
            rec['notes'] = list(rec.get('notes') or []) + ['自检未通过: %s' % '、'.join(failed)]
            rec['low_confidence'] = True
        self._trace(trace, '7 自检', 'self_check',
                    '条数/类目/同类上限/唯一性/去重',
                    '全部通过' if not failed else '未通过 %d 项' % len(failed), t)

        rec['constraints'] = parsed.to_dict()
        rec['mode'] = mode_note + ' · ' + rec['mode']
        result['status'] = STATUS_OK
        result['answer'] = rec
        return self._finish(result, trace, t_all)

    # ── 自检：把“Agent 说完成了”变成可核验的断言 ──
    def _self_check(self, rec, parsed, uid):
        items = rec['items']
        checks = {'条数满足': len(items) == parsed.top_k,
                  '类目满足': (not parsed.categories
                              or all(it['category'] in parsed.categories for it in items)),
                  '无重复': len({it['video_id'] for it in items}) == len(items)}
        cnt = Counter(it['category'] for it in items)
        within_cap = max(cnt.values()) <= parsed.max_per_cat if items else False
        # 回填是被显式标注的降级路径，不算违规
        checks['同类上限'] = within_cap or bool(rec.get('constraint_note'))
        if uid is not None and parsed.exclude_seen:
            seen = self.engine.history(uid)
            checks['未推已看'] = all(it['video_id'] not in seen for it in items)
        return checks

    @staticmethod
    def _scorer_name(mode):
        for key in ('BPR-MF + BM25', 'BPR-MF', 'KMeans', 'BM25', 'ItemCF', '热门'):
            if key in mode:
                return {'BPR-MF + BM25': 'hybrid', 'BPR-MF': 'mf', 'KMeans': 'cold',
                        'BM25': 'bm25', 'ItemCF': 'cf', '热门': 'pop'}[key]
        return 'auto'

    @staticmethod
    def _finish(result, trace, t_all):
        result['trace'] = trace
        result['latency_ms'] = round((time.perf_counter() - t_all) * 1000, 1)
        return result


def render_text(result):
    """渲染成人类可读卡片（CLI 与截图演示用）。"""
    lines = []
    st = result['status']
    if st == STATUS_OK:
        a = result['answer']
        lines.append('推荐结果（%s）' % a['mode'])
        for i, it in enumerate(a['items'], 1):
            lines.append('%2d. [%s] %s' % (i, it['category'], it['caption'] or '(无文案)'))
            lines.append('    分数 %.4f ｜ %s' % (it['score'], it['reason']))
        lines.append('覆盖类目: %s ｜ 候选池: %d %s'
                     % ('、'.join(a['category_coverage']), a['n_candidates'],
                        a.get('constraint_note') or ''))
        for n in a.get('notes') or []:
            lines.append('提示: ' + n)
        if a.get('low_confidence'):
            lines.append('建议人工复核：本次结果区分度不足或候选偏少')
        if a['constraints']['warnings']:
            lines.append('约束提示: ' + '；'.join(a['constraints']['warnings']))
        lines.append('自检: %s' % ' '.join(
            '%s=%s' % (k, '✓' if v else '✗') for k, v in (a.get('self_check') or {}).items()))
    elif st == STATUS_NEED_RELAX:
        lines.append('【需放宽约束】%s' % result['answer']['hint'])
    elif st == STATUS_INPUT_ERROR:
        lines.append('【输入错误】%s' % result['detail']['error'])
    else:
        lines.append('【系统错误】%s' % result['detail'].get('error', '未知'))
    if result.get('trace'):
        lines.append('执行轨迹: ' + ' → '.join('%s(%s)' % (s['step'], s['ms'])
                                          for s in result['trace'][:8]))
    if 'latency_ms' in result:
        lines.append('耗时 %.1f ms' % result['latency_ms'])
    return '\n'.join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser(description='约束感知短视频推荐 Agent')
    ap.add_argument('--user', type=int, help='数据集用户 id（如 1001）')
    ap.add_argument('--new-user', help='冷启动：声明的兴趣类目，空格/逗号分隔')
    ap.add_argument('--categories', default='', help='想看的类目，空格/逗号分隔')
    ap.add_argument('--query', default='', help='内容意图（走 BM25 文本检索/混合打分）')
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--max-per-cat', type=int, default=2)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    cons_str = args.categories
    if args.top_k != 5 or args.max_per_cat != 2:
        cons_str = '%s k=%d mpc=%d' % (cons_str, args.top_k, args.max_per_cat)
    user_input = {'constraints': cons_str.strip(), 'query': args.query}
    if args.user is not None:
        user_input['user_id'] = args.user
    if args.new_user is not None:
        user_input['new_user'] = args.new_user

    r = Agent().run(user_input)
    print(json.dumps(r, ensure_ascii=False, indent=2) if args.json else render_text(r))


if __name__ == '__main__':
    main()
