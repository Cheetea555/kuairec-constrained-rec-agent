# -*- coding: utf-8 -*-
"""Web 演示：表单提交约束，展示推荐结果与 Agent 执行轨迹。

运行（项目根目录）：python web/app.py  ->  http://127.0.0.1:5000
GET  /       表单
POST /       提交约束
GET  /demo   预填参数的示例（用于截图/演示复现），如
             /demo?user=1001&categories=美食&k=5&mpc=2&query=做饭
"""
import os
import sys

# 脚本方式启动时把项目根加入模块搜索路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask, render_template, request  # noqa: E402

from agent import constraints as cons_mod  # noqa: E402
from agent.agent import Agent  # noqa: E402

app = Flask(__name__)
AGENT = Agent()


def category_options():
    """按视频数从多到少排列类目，界面更好用（UNKNOWN 放最后）。"""
    from collections import Counter
    cnt = Counter(AGENT.engine.cats.values())
    names = [c for c, _ in cnt.most_common() if c != 'UNKNOWN']
    if 'UNKNOWN' in cnt:
        names.append('UNKNOWN')
    return names


def _int_or(name, default, src):
    try:
        return int(src.get(name) or default)
    except ValueError:
        return default


def build_input(src, multi=False):
    """把表单/查询参数归一成 Agent 的输入。"""
    user_input = {}
    uid = (src.get('user_id') or '').strip()
    new_user = src.getlist('new_user') if multi else (src.get('new_user') or '').split()
    cats = src.getlist('categories') if multi else (src.get('categories') or '').split()
    k = _int_or('top_k', 5, src)
    mpc = _int_or('max_per_cat', 2, src)
    query = (src.get('query') or '').strip()
    user_input['constraints'] = ' '.join(list(cats) + ['k=%d' % k, 'mpc=%d' % mpc])
    if query:
        user_input['query'] = query
    if uid:
        user_input['user_id'] = uid
    elif new_user:
        user_input['new_user'] = new_user
    return user_input


def render(user_input=None, result=None, form=None):
    return render_template('index.html', categories=category_options(),
                           result=result, form=form or {})


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    form = {'k': 5, 'mpc': 2, 'user_id': '', 'query': ''}
    if request.method == 'POST':
        user_input = build_input(request.form, multi=True)
        form = {'k': _int_or('top_k', 5, request.form),
                'mpc': _int_or('max_per_cat', 2, request.form),
                'user_id': request.form.get('user_id', ''),
                'query': request.form.get('query', '')}
        result = AGENT.run(user_input)
    return render(form=form, result=result)


@app.route('/demo')
def demo():
    """用查询参数直接渲染一次真实运行结果，便于录屏与截图复现。"""
    args = request.args
    form = {'user_id': args.get('user', ''), 'k': args.get('k', '5'),
            'mpc': args.get('mpc', '2'), 'query': args.get('query', '')}
    src = {'user_id': args.get('user', ''),
           'categories': args.get('categories', '').replace(',', ' '),
           'top_k': args.get('k', '5'), 'max_per_cat': args.get('mpc', '2'),
           'query': args.get('query', ''),
           'new_user': args.get('new_user', '')}
    user_input = build_input(src, multi=False)
    return render(user_input=user_input, result=AGENT.run(user_input), form=form)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
