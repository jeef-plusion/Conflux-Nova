// nova_v2/tools/novascope_check.js — 用 node 校验仪表盘的 JS 与统计口径
//
// 两件事：
//   1. 把 novascope.html 里的 <script> 抠出来做语法检查（没有浏览器时唯一的自动检查）；
//   2. 用同一套 summarize() 逻辑读 build\nova_trace.json，与 C++ 侧打印的四个 KPI 对拍
//      （两份独立实现跑同一份 trace，数字必须一致）。
//
// 用法: node tools\novascope_check.js [trace.json]
"use strict";

const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
let failed = 0;
function check(ok, what, detail) {
  console.log(`  [${ok ? "PASS" : "FAIL"}] ${what}${detail ? "  " + detail : ""}`);
  if (!ok) failed++;
}

// ---- 1. 语法检查 ----
const html = fs.readFileSync(path.join(root, "tools", "novascope.html"), "utf8");
const m = html.match(/<script>([\s\S]*?)<\/script>/);
check(!!m, "novascope.html 里有 <script> 块");
const js = m ? m[1] : "";

let syntaxOk = true, syntaxErr = "";
try {
  // 只做语法解析，不执行（页面里的 DOM 调用在没有浏览器时不能跑）
  new Function(js);
} catch (e) {
  syntaxOk = false;
  syntaxErr = e.message;
}
check(syntaxOk, "仪表盘 JS 语法可解析", syntaxErr);

// 关键函数是否都在
for (const fn of ["function summarize", "function loadTrace", "function render(",
                  "function animate", "function renderTimeline", "function demoTrace"]) {
  check(js.includes(fn), `定义了 ${fn.replace("function ", "").replace("(", "")}`);
}
// 三条泳道 + 粒子 + 红绿灯的判据都在
check(js.includes("LANE.BUS"), "有 Bus 泳道");
check(js.includes("particles"), "有粒子流");
check(js.includes("copyRatio > .15"), "有拷贝拥堵阈值 15%");
check(js.includes("busyRatio < .7"), "有算力空转阈值 70%");

// ---- 2. 与 C++ 对拍（同一份 trace） ----
const tracePath = process.argv[2] || path.join(root, "build", "nova_trace.json");
if (!fs.existsSync(tracePath)) {
  console.log(`\n  [SKIP] 没有 ${tracePath}（先跑 build\\scope_host_check.exe）`);
} else {
  // 从页面里抠出 summarize 的实现来执行 —— 保证测的就是页面用的那份逻辑
  const start = js.indexOf("function summarize");
  const end = js.indexOf("function loadTrace");
  const src = js.slice(start, end);
  const summarize = new Function(src + "; return summarize;")();

  const raw = JSON.parse(fs.readFileSync(tracePath, "utf8"));
  const evs = raw.traceEvents || raw;
  const s = summarize(evs);

  console.log(`\n--- 2. 与 C++ 对拍（${path.basename(tracePath)}）---`);
  console.log(`  事件 ${evs.length}，X 切片 ${evs.filter(e => e.ph === "X").length}，` +
              `wall ${(s.wall / 1000).toFixed(3)} ms，忙比 ${(100 * s.busyRatio).toFixed(2)}%，` +
              `拷贝 ${s.copies.length} 次 / ${(s.copyB / 1024).toFixed(0)} KiB，` +
              `有效带宽 ${s.gbs.toFixed(2)} GB/s`);

  // C++ 侧（scope_host_check 的合成流）的期望值：见 tests/scope_host_check.cpp 的手算
  //   墙钟 16.636 ms、忙 14.182 ms、拷贝 412 µs、4 次/token、2 token
  const near = (a, b, tol) => Math.abs(a - b) <= tol;
  check(near(s.wall / 1000, 16.636, 0.01), "墙钟与 C++ 一致（16.636 ms）", (s.wall / 1000).toFixed(3));
  check(near(s.busy / 1000, 13.770, 0.01), "GPU 忙与 C++ 一致（13.770 ms，不含拷贝）", (s.busy / 1000).toFixed(3));
  check(near(s.copyT / 1000, 0.412, 0.001), "拷贝耗与 C++ 一致（412 µs）", (s.copyT / 1000).toFixed(3));
  check(s.copies.length === 8, "拷贝次数 = 8（2 token × 4）", String(s.copies.length));
  check(near(s.copyPerTok, 4, 1e-9), "拷贝触发率 = 4 次/token", s.copyPerTok.toFixed(2));
  check(near(100 * s.busyRatio, 82.79, 0.05), "忙比 = 82.79%", (100 * s.busyRatio).toFixed(2));
  check(s.ops.length > 0 && s.ops[0].name === "gemv.q", "Top1 = gemv.q", s.ops[0] && s.ops[0].name);
}

console.log(`\n=== 结论：失败项 = ${failed} ===`);
process.exit(failed);
