const $ = s => document.querySelector(s);
const esc = t => t.replace(/[&<>]/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;' }[c]));
const md = t => window.marked ? marked.parse(t) : '<p>' + esc(t) + '</p>';

let ws = null, sid = '', sessions = [], streaming = false;
let cur = null, curText = '', lastTok = 0;
const rate = new Array(80).fill(0);

/* ---------- WebSocket ---------- */
function connect() {
  ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
  ws.onopen = () => { $('#dot').className = 'on'; };
  ws.onclose = () => { $('#dot').className = ''; setTimeout(connect, 1500); };
  ws.onmessage = e => on(JSON.parse(e.data));
}
const sendWs = o => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); };

function on(m) {
  if (m.ev === 'tok') {
    lastTok = Date.now();
    curText += m.t;
    paint();
    autoScroll();
  } else if (m.ev === 'start') {
    if (m.session_id) sid = m.session_id;
  } else if (m.ev === 'tele') {
    telemetry(m);
  }
}

/* ---------- 渲染 ---------- */
function paint() {
  if (!cur) return;
  cur.classList.add('caret');
  cur.innerHTML = md(curText);
  if (window.hljs) cur.querySelectorAll('pre code').forEach(b => {
    if (!b.dataset.hl) { hljs.highlightElement(b); b.dataset.hl = 1; }
  });
}
function addMsg(role, text, stream) {
  $('#welcome') && ($('#welcome').style.display = 'none');
  const m = document.createElement('div');
  m.className = 'msg ' + role;
  if (role === 'assistant') {
    const who = document.createElement('div'); who.className = 'who'; who.textContent = 'N';
    m.appendChild(who);
  }
  const b = document.createElement('div'); b.className = 'bubble';
  const c = document.createElement('div'); c.className = 'md';
  if (text) c.innerHTML = md(text);
  if (stream) { c.classList.add('caret'); cur = c; curText = ''; }
  b.appendChild(c); m.appendChild(b);
  $('#thread').appendChild(m);
  autoScroll();
  return c;
}
function autoScroll() {
  const s = $('#scroll');
  if (s.scrollHeight - s.scrollTop - s.clientHeight < 220) s.scrollTop = s.scrollHeight;
}

/* ---------- 会话 ---------- */
async function refreshSessions() {
  const j = await (await fetch('/api/sessions')).json();
  sessions = j.sessions || [];
  renderHist();
}
function renderHist() {
  const h = $('#hist'); h.innerHTML = '';
  sessions.slice().reverse().forEach(s => {
    const d = document.createElement('div');
    d.className = 'item' + (s.id === sid ? ' on' : '');
    d.innerHTML = '<span class="t">' + esc(s.title || s.id) + '</span><span class="n">' + (s.turns || 0) + '</span>';
    d.onclick = () => openSession(s.id);
    h.appendChild(d);
  });
}
async function openSession(id) {
  if (streaming) return;
  sid = id;
  const j = await (await fetch('/api/history?session_id=' + encodeURIComponent(id))).json();
  $('#thread').innerHTML = '';
  cur = null;
  (j.messages || []).forEach(m => addMsg(m.role === 'user' ? 'user' : 'assistant', m.text));
  renderHist();
  $('#scroll').scrollTop = $('#scroll').scrollHeight;
}
$('#newChat').onclick = async () => {
  const j = await (await fetch('/api/new', { method: 'POST' })).json();
  sid = j.id;
  $('#thread').innerHTML = '';
  cur = null;
  await refreshSessions();
  $('#input').focus();
};

/* ---------- 发送 ---------- */
async function submit() {
  const t = $('#input').value.trim();
  if (!t || streaming) return;
  if (!sid) { const j = await (await fetch('/api/new', { method: 'POST' })).json(); sid = j.id; }

  $('#input').value = ''; autoGrow();
  addMsg('user', t);
  streaming = true; $('#send').disabled = true;
  cur = null;
  const c = addMsg('assistant', '', true);
  cur = c; curText = ''; lastTok = Date.now();
  sendWs({ cmd: 'chat', text: t, session_id: sid });

  const iv = setInterval(() => {
    if (Date.now() - lastTok > 1400 && curText.length > 0) {
      clearInterval(iv);
      if (cur) cur.classList.remove('caret');
      streaming = false; $('#send').disabled = false; cur = null;
      refreshSessions();
    }
  }, 300);
}
$('#send').onclick = submit;
$('#input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
});
function autoGrow() { const el = $('#input'); el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 200) + 'px'; }
$('#input').addEventListener('input', autoGrow);
document.querySelectorAll('.card').forEach(c => c.onclick = () => {
  $('#input').value = c.querySelector('span').textContent; autoGrow(); submit();
});

/* ---------- 模型加载 ---------- */
$('#load').onclick = async () => {
  const p = $('#gguf').value.trim();
  if (!p) return;
  $('#stat').textContent = '加载中…（约 25s）';
  try {
    const r = await fetch('/api/load', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: p }) });
    const j = await r.json();
    $('#stat').textContent = j.ok ? '已加载' : ('失败：' + j.err);
    $('#stat').style.color = j.ok ? '#34d399' : '#f87171';
    if (j.ok) { $('#model').textContent = p.split(/[\\/]/).pop(); await refreshSessions(); }
  } catch (e) { $('#stat').textContent = '请求失败'; }
};

/* ---------- 遥测 ---------- */
function telemetry(m) {
  $('#tps').textContent = (m.tok_s || 0).toFixed(1);
  $('#vram').textContent = (m.vram_used || 0).toFixed(2) + '/' + (m.vram_cap || 0).toFixed(2);
  $('#npast').textContent = m.n_past || 0;
  $('#blocks').textContent = ((m.blocks_total || 0) - (m.blocks_free || 0)) + '/' + (m.blocks_total || 0);
  $('#shared').textContent = m.shared || 0;
  $('#cow').textContent = m.cow || 0;
  rate.push(m.tok_s || 0); rate.shift();
  drawChart();
}

/* ---------- 折线图（固定尺寸，只用 ResizeObserver） ---------- */
const cv = $('#chart');
let cw = 0, ch = 0;
function sizeChart() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(cv.clientWidth)), h = Math.max(1, Math.round(cv.clientHeight));
  if (w === cw && h === ch) return;
  cw = w; ch = h;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  drawChart();
}
function drawChart() {
  if (!cw || !ch) return;
  const g = cv.getContext('2d');
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, cw, ch);
  const max = Math.max(12, ...rate), n = rate.length, pad = 3;
  const y = v => ch - pad - (v / max) * (ch - pad * 2);
  // 网格
  g.strokeStyle = 'rgba(255,255,255,.05)'; g.lineWidth = 1;
  for (let i = 1; i <= 3; i++) { const yy = (ch / 4) * i; g.beginPath(); g.moveTo(0, yy); g.lineTo(cw, yy); g.stroke(); }
  // 面积
  g.beginPath();
  rate.forEach((v, i) => { const x = (i / (n - 1)) * cw; i ? g.lineTo(x, y(v)) : g.moveTo(x, y(v)); });
  g.lineTo(cw, ch); g.lineTo(0, ch); g.closePath();
  const grad = g.createLinearGradient(0, 0, 0, ch);
  grad.addColorStop(0, 'rgba(59,130,246,.28)');
  grad.addColorStop(1, 'rgba(59,130,246,0)');
  g.fillStyle = grad; g.fill();
  // 线
  g.beginPath();
  rate.forEach((v, i) => { const x = (i / (n - 1)) * cw; i ? g.lineTo(x, y(v)) : g.moveTo(x, y(v)); });
  g.strokeStyle = '#3b82f6'; g.lineWidth = 1.6; g.lineJoin = 'round'; g.stroke();
}
new ResizeObserver(sizeChart).observe(cv);

/* 轮询遥测（WS 未连接也能显示） */
setInterval(async () => {
  try { const r = await fetch('/api/telemetry?session_id=' + encodeURIComponent(sid)); telemetry(await r.json()); } catch (e) {}
}, 800);

connect(); sizeChart(); refreshSessions();