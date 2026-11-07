const $ = s => document.querySelector(s);
let ws, last = '', chart, data = [], cur = null;

const md = t => (window.marked ? marked.parse(t) : t.replace(/</g,'&lt;'));

function connect(){
  ws = new WebSocket('ws://127.0.0.1:17530');
  ws.onopen = () => send({cmd:'hello'});
  ws.onmessage = e => on(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connect, 1000);
}
function send(o){ if(ws && ws.readyState===1) ws.send(JSON.stringify(o)); }

function on(m){
  if(m.ev==='tok'){ last += m.t; if(cur) cur.innerHTML = md(last); scroll(); }
  else if(m.ev==='tele') tele(m);
  else if(m.ev==='loaded'){
    if(m.ok){ $('#stat').textContent = '已加载 · '+(m.path||'').split(/[\\/]/).pop(); $('#stat').style.color='#3ddc84'; }
    else { $('#stat').textContent = '失败: ' + (m.err||'未知'); $('#stat').style.color='#ff6b5e'; }
  }
  else if(m.ev==='picked'){ if(m.path){ $('#gguf').value = m.path; $('#stat').textContent = '已选择模型'; } }
  else if(m.ev==='start'){ last=''; }
}

function bubble(role, html){
  const d=document.createElement('div'); d.className='msg '+role; d.innerHTML=html;
  $('#chat').appendChild(d); scroll(); return d;
}
function scroll(){ const c=$('#chat'); c.scrollTop=c.scrollHeight; }

$('#send').onclick = () => {
  const t=$('#input').value.trim(); if(!t) return;
  bubble('user', md(t)); $('#input').value='';
  last=''; cur = bubble('bot','');
  const f=document.createElement('button'); f.className='forkBtn'; f.textContent='Fork';
  f.onclick=()=>send({cmd:'fork'}); cur.appendChild(f);
  send({cmd:'chat', text:t});
};
$('#input').addEventListener('keydown', e=>{ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();$('#send').click();} });
$('#browse').onclick = ()=> send({cmd:'pick'});
$('#ignite').onclick = ()=>{
  const p = $('#gguf').value.trim();
  if(!p){ $('#stat').textContent='先选模型'; send({cmd:'pick'}); return; }
  $('#stat').textContent = '加载中…';
  send({cmd:'load', path:p});
};
$('#new').onclick = ()=>{ $('#chat').innerHTML=''; last=''; };
$('#toggle').onclick = ()=>{ const r=$('#right'); r.style.display = r.style.display==='none'?'block':'none'; };

function tele(m){
  $('#tps').textContent=(m.tok_s||0).toFixed(1);
  $('#vram').textContent=(m.vram_used||0).toFixed(2)+' / '+(m.vram_cap||0).toFixed(2);
  $('#vbar').style.width=Math.min(100,100*(m.vram_used||0)/(m.vram_cap||1))+'%';
  $('#blocks').textContent=((m.blocks_total||0)-(m.blocks_free||0))+' / '+(m.blocks_total||0);
  $('#shared').textContent=m.shared||0;
  $('#cow').textContent=m.cow||0;
  $('#temp').textContent=(m.temp||0).toFixed(2);
  $('#topk').textContent=m.topk||0;
  $('#npast').textContent=m.n_past||0;
  data.push([Date.now(), m.tok_s||0]); if(data.length>120) data.shift(); drawChart();
}

function drawChart(){
  if(chart){
    chart.setOption({series:[{data:data}]});
    return;
  }
  if(window.echarts){
    chart = echarts.init($('#chart'), null, {renderer:'canvas'});
    chart.setOption({
      grid:{left:2,right:2,top:6,bottom:2},
      xAxis:{type:'time',show:false},
      yAxis:{type:'value',show:false,min:0},
      series:[{type:'line',showSymbol:false,smooth:true,lineStyle:{width:2,color:'#4da3ff'},
                areaStyle:{color:new echarts.graphic.LinearGradient(0,0,0,1,[
                  {offset:0,color:'rgba(77,163,255,.45)'},{offset:1,color:'rgba(77,163,255,0)'}])},
                data:data}]
    });
    window.addEventListener('resize',()=>chart.resize());
  }
}

connect();