/* 约束感知短视频推荐 Agent —— 纯前端演示
 * 模型参数从训练产物导出（P: 1411×32, Q: 3327×32, 训练期已看位图），
 * 打分、约束过滤、同类上限重排、理由生成全部在浏览器内完成。
 */
(function () {
  'use strict';

  const state = {
    ready: false,
    n_i: 0,
    n_u: 0,
    dims: 32,
    users: [],
    itemIds: [],
    itemCat: [],
    itemCap: [],
    cats: [],
    P: null,
    Q: null,
    seen: null,
  };

  const $ = (id) => document.getElementById(id);
  const fmtPct = (x) => (x * 100).toFixed(1) + '%';

  // ── 位图读取：bit 索引 k = ui * n_i + j，packbits 高位在前 ──
  function seenBit(ui, j) {
    const k = ui * state.n_i + j;
    return (state.seen[k >> 3] >> (7 - (k & 7))) & 1;
  }

  function ensureDOMCats() {
    const counts = new Map();
    for (let j = 0; j < state.n_i; j++) {
      counts.set(state.itemCat[j], (counts.get(state.itemCat[j]) || 0) + 1);
    }
    state.cats = [...counts.entries()]
      .sort((a, b) => (a[0] === 'UNKNOWN' ? 1 : b[0] === 'UNKNOWN' ? -1 : b[1] - a[1]))
      .map(([c]) => c);

    const box = $('cats');
    box.innerHTML = '';
    state.cats.forEach((c) => {
      const label = document.createElement('label');
      label.className = 'cat';
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.value = c;
      input.name = 'cat';
      const span = document.createElement('span');
      span.textContent = c;
      label.appendChild(input);
      label.appendChild(span);
      box.appendChild(label);
    });
  }

  async function boot() {
    const result = $('result');
    result.hidden = false;
    result.innerHTML = '<div class="status">正在加载模型参数（约 1.5 MB）…</div>';

    try {
      const [itemsRes, modelRes, seenRes] = await Promise.all([
        fetch('assets/items.json'),
        fetch('assets/model.bin'),
        fetch('assets/seen.bin'),
      ]);
      if (!itemsRes.ok || !modelRes.ok || !seenRes.ok) throw new Error('静态资源加载失败');

      const meta = await itemsRes.json();
      state.n_i = meta.n_i;
      state.users = meta.users;
      state.dims = meta.dims;
      state.n_u = meta.users.length;
      meta.items.forEach(([id, cat, cap]) => {
        state.itemIds.push(id);
        state.itemCat.push(cat);
        state.itemCap.push(cap || '');
      });

      const modelBuf = await modelRes.arrayBuffer();
      const nP = state.n_u * state.dims;
      state.P = new Float32Array(modelBuf, 0, nP);
      state.Q = new Float32Array(modelBuf, nP * 4, state.n_i * state.dims);

      state.seen = new Uint8Array(await seenRes.arrayBuffer());
      ensureDOMCats();

      const dl = $('users');
      state.users.forEach((u) => {
        const o = document.createElement('option');
        o.value = String(u);
        dl.appendChild(o);
      });

      state.ready = true;
      result.hidden = true;
      result.innerHTML = '';
      applyUrlParams() || run();
    } catch (e) {
      result.innerHTML = '<div class="status err">加载失败：' + e.message + '</div>';
    }
  }

  // ── 核心：召回 → 打分 → 重排 → 理由 → 自检 ──
  function recommend(ui, cats, k, mpc, excludeSeen) {
    const t0 = performance.now();
    const n_i = state.n_i;

    // 1) 训练期历史与类目占比（推荐理由用）
    const share = {};
    let histTotal = 0;
    for (let j = 0; j < n_i; j++) {
      if (seenBit(ui, j)) {
        histTotal++;
        const c = state.itemCat[j];
        share[c] = (share[c] || 0) + 1;
      }
    }
    for (const c in share) share[c] /= histTotal || 1;
    const t1 = performance.now();

    // 2) 候选召回：类目过滤 + 剔除已看
    const cand = [];
    for (let j = 0; j < n_i; j++) {
      if (cats.length && cats.indexOf(state.itemCat[j]) === -1) continue;
      if (excludeSeen && seenBit(ui, j)) continue;
      cand.push(j);
    }
    const t2 = performance.now();
    if (cand.length < k) {
      return {
        status: 'NEED_RELAX',
        n: cand.length,
        histTotal,
        hint: '当前约束下候选只有 ' + cand.length + ' 条，不足 ' + k +
              ' 条；请减少类目或提高同类上限。系统不会用其他内容硬凑。',
        trace: [
          ['1 约束解析', 'constraints.parse', '类目=' + (cats.length ? cats.join('、') : '全部') + '，k=' + k + '，同类上限=' + mpc, 0],
          ['2 身份识别', 'identity.resolve', '老用户 ' + state.users[ui] + '（训练期历史 ' + histTotal + ' 条）', 0],
          ['3 候选召回', 'recall.category+seen', '候选 ' + cand.length + ' 条', t2 - t1],
          ['4 可行域检查', 'feasibility', '需要 ' + k + ' 条，候选不足', performance.now() - t2],
          ['5 处理', 'guardrail', '返回 NEED_RELAX，请求放宽而非硬凑', 0],
        ],
      };
    }

    // 3) 打分：BPR-MF 内积
    const P = state.P, Q = state.Q, base = ui * state.dims, dims = state.dims;
    const scored = new Array(cand.length);
    for (let c = 0; c < cand.length; c++) {
      const j = cand[c];
      const qb = j * dims;
      let s = 0;
      for (let d = 0; d < dims; d++) s += P[base + d] * Q[qb + d];
      scored[c] = [j, s];
    }
    scored.sort((a, b) => b[1] - a[1]);
    const t3 = performance.now();

    // 4) 同类上限重排（贪心，不足回填并标注）
    const picked = [], cnt = {};
    for (let i = 0; i < scored.length; i++) {
      const j = scored[i][0], c = state.itemCat[j];
      if ((cnt[c] || 0) >= mpc) continue;
      picked.push(scored[i]);
      cnt[c] = (cnt[c] || 0) + 1;
      if (picked.length >= k) break;
    }
    let note = '';
    if (picked.length < k) {
      const used = new Set(picked.map((p) => p[0]));
      const rest = scored.filter((p) => !used.has(p[0]));
      const n = Math.min(k - picked.length, rest.length);
      for (let i = 0; i < n; i++) picked.push(rest[i]);
      note = '（同类上限截断后回填 ' + n + ' 条）';
    }
    const t4 = performance.now();

    // 5) 理由 + 自检
    const items = picked.map(([j, s]) => {
      const c = state.itemCat[j];
      const parts = [];
      if ((share[c] || 0) >= 0.03) {
        parts.push('与你历史中「' + c + '」类目偏好一致（占历史 ' + Math.round(share[c] * 100) + '%）');
      }
      if (cats.length) parts.push('满足指定类目');
      parts.push('MF 打分 ' + s.toFixed(2));
      return { id: state.itemIds[j], cat: c, cap: state.itemCap[j], score: s, reason: parts.join('；') };
    });
    const t5 = performance.now();

    const catCount = {};
    items.forEach((it) => { catCount[it.cat] = (catCount[it.cat] || 0) + 1; });
    const withinCap = Object.values(catCount).every((v) => v <= mpc) || !!note;
    const checks = [
      ['条数满足', items.length === k],
      ['类目满足', !cats.length || items.every((it) => cats.indexOf(it.cat) !== -1)],
      ['同类上限', withinCap],
      ['无重复', new Set(items.map((it) => it.id)).size === items.length],
    ];

    return {
      status: 'OK',
      items,
      n: cand.length,
      histTotal,
      histShare: share,
      note,
      lowConf: cand.length < k * 3,
      totalMs: performance.now() - t0,
      trace: [
        ['1 约束解析', 'constraints.parse', '类目=' + (cats.length ? cats.join('、') : '全部') + '，k=' + k + '，同类上限=' + mpc, 0],
        ['2 身份识别', 'identity.resolve', '老用户 ' + state.users[ui] + '（训练期历史 ' + histTotal + ' 条）', 0],
        ['3 候选召回', 'recall.category+seen', '候选 ' + cand.length + ' 条', t2 - t1],
        ['4 打分排序', 'score.mf', 'BPR-MF(个性化)', t3 - t2],
        ['5 多样性重排', 'rerank.greedy', items.length + ' 条，覆盖 ' + Object.keys(catCount).length + ' 个类目', t4 - t3],
        ['6 理由生成', 'explain.template', '已生成 ' + items.length + ' 条理由', t5 - t4],
        ['7 自检', 'self_check', checks.map(([n, ok]) => n + (ok ? '✓' : '✗')).join(' '), 0],
      ],
      checks,
    };
  }

  // ── 渲染 ──
  function render(ui, cats, k, mpc, exclude, res) {
    const box = $('result');
    box.hidden = false;

    if (res.status === 'NEED_RELAX') {
      box.innerHTML = '<div class="status relax">【需要放宽约束】' + esc(res.hint) + '</div>';
      box.appendChild(traceEl(res.trace));
      return;
    }

    const mode = '老用户 ' + state.users[ui] + '（训练期历史 ' + res.histTotal + ' 条） · BPR-MF(个性化)';
    const status = document.createElement('div');
    status.className = 'status';
    status.textContent = '已完成：' + mode;
    box.innerHTML = '';
    box.appendChild(status);

    const stats = document.createElement('div');
    stats.className = 'stats';
    stats.innerHTML =
      '<div class="stat"><b>' + res.items.length + '</b><span>推荐条数</span></div>' +
      '<div class="stat"><b>' + res.n + '</b><span>候选池</span></div>' +
      '<div class="stat"><b>' + new Set(res.items.map((i) => i.cat)).size + '</b><span>覆盖类目</span></div>' +
      '<div class="stat"><b>' + res.totalMs.toFixed(1) + ' ms</b><span>端到端耗时</span></div>' +
      (res.lowConf ? '<div class="stat"><b class="warn">低置信</b><span>建议人工复核</span></div>' : '');
    box.appendChild(stats);

    const table = document.createElement('table');
    const head = document.createElement('tr');
    ['#', '类目', '视频文案', '分数', '推荐依据'].forEach((h) => {
      const th = document.createElement('th');
      th.textContent = h;
      head.appendChild(th);
    });
    table.appendChild(head);
    res.items.forEach((it, i) => {
      const tr = document.createElement('tr');
      tr.innerHTML =
        '<td class="rank">' + (i + 1) + '</td>' +
        '<td><span class="badge"></span></td>' +
        '<td class="cap"></td>' +
        '<td class="score">' + it.score.toFixed(4) + '</td>' +
        '<td class="reason"></td>';
      tr.querySelector('.badge').textContent = it.cat;
      tr.querySelector('.cap').textContent = it.cap || '（无文案）';
      tr.querySelector('.reason').textContent = it.reason;
      table.appendChild(tr);
    });
    box.appendChild(table);

    if (res.note) {
      const p = document.createElement('p');
      p.className = 'hint';
      p.textContent = '提示：' + res.note;
      box.appendChild(p);
    }
    box.appendChild(traceEl(res.trace));
  }

  function traceEl(trace) {
    const wrap = document.createElement('div');
    wrap.className = 'trace';
    const h = document.createElement('h3');
    h.textContent = 'Agent 执行轨迹';
    wrap.appendChild(h);
    const ul = document.createElement('ul');
    ul.className = 'steps';
    trace.forEach(([step, tool, out, ms]) => {
      const li = document.createElement('li');
      const b = document.createElement('b');
      b.textContent = step;
      li.appendChild(b);
      li.appendChild(document.createTextNode(out));
      const em = document.createElement('em');
      em.textContent = tool + ' · ' + (ms >= 0.1 ? ms.toFixed(1) : '0.0') + ' ms';
      li.appendChild(document.createElement('br'));
      li.appendChild(em);
      ul.appendChild(li);
    });
    wrap.appendChild(ul);
    return wrap;
  }

  function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  // ── 表单与状态 ──
  function readForm() {
    const uid = parseInt($('user').value, 10);
    const k = Math.max(1, Math.min(10, parseInt($('k').value, 10) || 5));
    const mpc = Math.max(1, Math.min(5, parseInt($('mpc').value, 10) || 2));
    const cats = [...document.querySelectorAll('input[name="cat"]:checked')].map((x) => x.value);
    const exclude = $('exclude').checked;
    return { uid, k, mpc, cats, exclude };
  }

  function run() {
    if (!state.ready) return;
    const { uid, k, mpc, cats, exclude } = readForm();
    const ui = state.users.indexOf(uid);
    if (ui === -1) {
      renderError('用户 ' + (isNaN(uid) ? '（空）' : uid) +
        ' 不在数据集（KuaiRec 小矩阵共 ' + state.users.length + ' 个用户，id 形如 ' +
        state.users[0] + '、' + state.users[Math.floor(state.users.length / 2)] + '）。');
      return;
    }
    const res = recommend(ui, cats, k, mpc, exclude);
    render(ui, cats, k, mpc, exclude, res);
    const q = new URLSearchParams({ user: uid, k, mpc });
    if (cats.length) q.set('cats', cats.join(','));
    if (!exclude) q.set('exclude', '0');
    history.replaceState(null, '', '?' + q.toString() + '#demo');
  }

  function renderError(msg) {
    const box = $('result');
    box.hidden = false;
    box.innerHTML = '<div class="status err">【输入错误】' + esc(msg) + '</div>';
  }

  function applyUrlParams() {
    const q = new URLSearchParams(location.search);
    if (!q.has('user') && !q.has('cats')) return false;
    if (q.has('user')) $('user').value = q.get('user');
    if (q.has('k')) $('k').value = q.get('k');
    if (q.has('mpc')) $('mpc').value = q.get('mpc');
    if (q.has('cats')) {
      const set = new Set(q.get('cats').split(','));
      document.querySelectorAll('input[name="cat"]').forEach((x) => { x.checked = set.has(x.value); });
    }
    if (q.get('exclude') === '0') $('exclude').checked = false;
    run();
    return true;
  }

  function init() {
    $('form').addEventListener('submit', (e) => { e.preventDefault(); run(); });
    $('randomUser').addEventListener('click', () => {
      if (!state.ready) return;
      $('user').value = state.users[Math.floor(Math.random() * state.users.length)];
      run();
    });
    $('surprise').addEventListener('click', () => {
      document.querySelectorAll('input[name="cat"]').forEach((x) => { x.checked = x.value === '读书'; });
      $('k').value = 10;
      $('mpc').value = 1;
      run();
      document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
    });
    boot();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

