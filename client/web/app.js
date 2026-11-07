const $ = s => document.querySelector(s);
const md = t => window.marked ? marked.parse(t) : t.replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));

let ws = null, stream = null, buf = '', hist = new Array(96).fill(0), busy = false;

/* ---------- 连接 ---------- */
function connect() {
  ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
  ws.onopen = () => { $('#dot').className = 'on'; };
  ws.onclose = () => { $('#dot').className = 'err'; setTimeout(connect, 1200); };
  ws.onmessage = e => on(JSON.parse(e.data));
}
const send = o => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); };

function on(m) {
  if (m.ev === 'tok') {
    buf += m.t;
    if (stream) { stream.innerHTML = md(buf).replace(/<\/p>$/, '') + '<span class="cursor"></span>'; tweak(); }
    scroll();
  } else if (m.ev === 'start') {
    buf = '';
  } else if (m.ev === 'tele') {
    tele(m);
  }
}

/* 代码块高亮 + 复制按钮 */
function tweak() {
  if (!window.hljs || !stream) return;
  stream.querySelectorAll('pre code').forEach(b => { if (!b.dataset.hl) { hljs.highlightElement(b); b.dataset.hl = 1; } });
}

/* ---------- 消息 ---------- */
function turn(role, text) {
  const t = document.createElement('div');
  t.className = 'turn ' + role;
  const av = document.createElement('div'); av.className = 'avatar'; av.textContent = role === 'user' ? '你' : 'N';
  const body = document.createElement('div'); body.className = 'body';
  const name = document.createElement('div'); name.className = 'name';
  name.textContent = role === 'user' ? '你' : 'Conflux Nova';
  if (role === 'bot') {
    const f = document.createElement('button'); f.className = 'fork'; f.textContent = 'Fork';
    f.onclick = () => send({ cmd: 'fork' });
    name.appendChild(f);
  }
  const c = document.createElement('div'); c.className = 'content';
  c.innerHTML = md(text || '');
  body.append(name, c); t.append(av, body); $('#chat').appendChild(t);
  $('#hero') && ($('#hero').style.display = 'none');
  scroll();
  return c;
}
function scroll() { const c = $('#chat'); c.scrollTop = c.scrollHeight; }

/* ---------- 发送 ---------- */
function submit() {
  const t = $('#input').value.trim();
  if (!t || busy) return;
  busy = true; $('#send').disabled = true;
  turn('user', t);
  $('#input').value = ''; autosize();
  buf = '';
  stream = turn('bot', '');
  stream.innerHTML = '<span class="cursor"></span>';
  send({ cmd: 'chat', text: t });
  waitIdle();
}
function waitIdle() {
  const iv = setInterval(() => {
    if (ws && ws.readyState !== 1) return;
    if (buf.length > 0) { clearInterval(iv); busy = false; $('#send').disabled = false; }
  }, 400);
  setTimeout(() => { clearInterval(iv); busy = false; $('#send').disabled = false; }, 180000);
}

$('#send').onclick = submit;
$('#input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
});
function autosize() { const el = $('#input'); el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 180) + 'px'; }
$('#input').addEventListener('input', autosize);
document.querySelectorAll('.chip').forEach(c => c.onclick = () => { $('#input').value = c.textContent; autosize(); $('#input').focus(); });

/* ---------- 模型 ---------- */
$('#ignite').onclick = async () => {
  const p = $('#gguf').value.trim();
  if (!p) return;
  $('#stat').textContent = '加载中…'; $('#stat').style.color = '#8a8a93'; $('#dot').className = '';
  try {
    const r = await fetch('/api/load', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: p }) });
    const j = await r.json();
    if (j.ok) { $('#stat').textContent = '已加载'; $('#stat').style.color = '#7bd88f'; $('#dot').className = 'on'; }
    else { $('#stat').textContent = '失败: ' + j.err; $('#stat').style.color = '#f87171'; $('#dot').className = 'err'; }
  } catch (e) { $('#stat').textContent = '请求失败'; $('#dot').className = 'err'; }
};

/* ---------- 遥测 ---------- */
function tele(m) {
  $('#tps').textContent = (m.tok_s || 0).toFixed(1);
  $('#vram').textContent = (m.vram_used || 0).toFixed(2) + ' / ' + (m.vram_cap || 0).toFixed(2);
  $('#vbar').style.width = Math.min(100, 100 * (m.vram_used || 0) / (m.vram_cap || 1)) + '%';
  $('#blocks').textContent = ((m.blocks_total || 0) - (m.blocks_free || 0)) + ' / ' + (m.blocks_total || 0);
  $('#shared').textContent = m.shared || 0;
  $('#cow').textContent = m.cow || 0;
  $('#temp').textContent = (m.temp || 0).toFixed(2);
  $('#topk').textContent = m.topk || 0;
  $('#npast').textContent = m.n_past || 0;
  $('#meta').textContent = (m.n_past ? m.n_past + ' tokens · ' : '') + (m.tok_s || 0).toFixed(1) + ' tok/s';
  hist.push(m.tok_s || 0); hist.shift(); spark();
}

function spark() {
  const c = $('#spark'), g = c.getContext('2d');
  const W = c.width = c.clientWidth * devicePixelRatio, H = c.height = 64 * devicePixelRatio;
  g.clearRect(0, 0, W, H);
  const max = Math.max(10, ...hist), n = hist.length;
  const pts = hist.map((v, i) => [i * W / (n - 1), H - (v / max) * (H - 8) - 4]);
  g.beginPath(); pts.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y));
  g.strokeStyle = '#d97757'; g.lineWidth = 1.6 * devicePixelRatio; g.stroke();
  g.lineTo(W, H); g.lineTo(0, H); g.closePath();
  const grad = g.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, 'rgba(217,119,87,.30)'); grad.addColorStop(1, 'rgba(217,119,87,0)');
  g.fillStyle = grad; g.fill();
}

/* 轮询兜底（WS 没连上也能看到数字） */
setInterval(async () => {
  try { const r = await fetch('/api/telemetry'); tele(await r.json()); } catch (e) {}
}, 700);

$('#new').onclick = () => { $('#chat').innerHTML = ''; location.reload(); };
$('#toggle').onclick = () => { const t = $('#tele'); t.style.display = t.style.display === 'none' ? 'block' : 'none'; };

connect(); spark();