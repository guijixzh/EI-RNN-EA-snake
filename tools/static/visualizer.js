"use strict";
/* ============================================================
   E-I 皮质柱脑 · 贪吃蛇实时可视化 —— 前端逻辑
   后端：brain_visualizer.py（独立进程，SSE 推送）
   注意：必须通过 http://127.0.0.1:8000 访问；直接双击 HTML 无效
============================================================ */
const $ = (id) => document.getElementById(id);
const MAX_HIST = 150;

const S = {
  layout: null, meta: null,
  paused: false, speed: 1,
  histObs: [], histE: [], histTau: [], histEx: [], histInH: [], histLogits: [],
  histActions: [], histSwitch: [],
  histLen: 0,
  cur: null,
  episode: 0,
  topoStatic: null,
  topoCtx: null,
  colPos: null, inPos: null, outPos: null,
  nodeColors: null,
  order: null,
  commBd: [],
  heatTab: "obs",
  heatRowH: 4,           // 热力图每行像素高度（滑块实时调节）
  heatCv: null, heatCtx: null,
  connState: "connecting",
  buffers: [],
};

/* ---------------- 色标 LUT ---------------- */
function makeLUT(stops) {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let lo = stops[0], hi = stops[stops.length - 1];
    for (let k = 0; k < stops.length - 1; k++) {
      if (t >= stops[k][0] && t <= stops[k + 1][0]) { lo = stops[k]; hi = stops[k + 1]; break; }
    }
    const span = (hi[0] - lo[0]) || 1e-9;
    const f = (t - lo[0]) / span;
    lut[i * 3]     = lo[1] + (hi[1] - lo[1]) * f;
    lut[i * 3 + 1] = lo[2] + (hi[2] - lo[2]) * f;
    lut[i * 3 + 2] = lo[3] + (hi[3] - lo[3]) * f;
  }
  return lut;
}
const LUT_VIRIDIS = makeLUT([[0,68,1,84],[0.25,59,82,139],[0.5,33,145,140],[0.75,94,201,98],[1,253,231,37]]);
const LUT_PLASMA  = makeLUT([[0,13,8,135],[0.25,126,3,168],[0.5,204,71,120],[0.75,248,149,64],[1,240,249,33]]);
const LUT_REDS    = makeLUT([[0,15,15,15],[0.5,155,30,30],[1,255,80,60]]);
const LUT_BLUES   = makeLUT([[0,15,15,15],[0.5,20,80,180],[1,90,200,255]]);

function idx2color(lut, t) {
  let i = (t <= 0) ? 0 : (t >= 1 ? 255 : (t * 255) | 0);
  return [lut[i * 3], lut[i * 3 + 1], lut[i * 3 + 2]];
}
function rgba(c, a) { return `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`; }

/* ---------------- 连接与协议检测 ---------------- */
function setConn(state, text) {
  S.connState = state;
  const dot = $("conn-dot"), txt = $("conn-text");
  dot.className = state;
  txt.textContent = text;
}
function setHint(t) { $("hint").textContent = t; }

function protocolError() {
  setConn("reconnecting", "✗ 无法连接后端");
  setHint("请改用 http://127.0.0.1:8000 访问（先运行 python brain_visualizer.py）");
  $("btnNew").disabled = true;
  $("btnPause").disabled = true;
  $("modelSelect").disabled = true;
}

/* ---------------- 输入/输出面板 DOM ---------------- */
const RAY_NAMES = ["左", "左前", "前", "右前", "右"];
const SELF_NAMES = ["前", "左前", "左", "左后", "后", "右后", "右", "右前"];

// test7a 32proj 观测通道（头/尾方向 one-hot ×4，食物 8 扇区投影，自体 8 扇区，障碍 8 扇区）
const OBS32_LABELS = [
  "头·右", "头·下", "头·左", "头·上",
  "尾·右", "尾·下", "尾·左", "尾·上",
  "食·前", "食·左前", "食·左", "食·左后", "食·后", "食·右后", "食·右", "食·右前",
  "身·前", "身·左前", "身·左", "身·左后", "身·后", "身·右后", "身·右", "身·右前",
  "障·前", "障·左前", "障·左", "障·左后", "障·后·√(len/G)", "障·右后", "障·右", "障·右前",
];
const OBS32_GROUPS = [
  { name: "蛇头方向 one-hot", from: 0, to: 4, color: "#3fb950" },
  { name: "蛇尾方向 one-hot", from: 4, to: 8, color: "#3fe0c8" },
  { name: "食物 8 扇区投影 (×8)", from: 8, to: 16, color: "#ffd23d" },
  { name: "自体 8 扇区 (×8)", from: 16, to: 24, color: "#d07ff5" },
  { name: "障碍 8 扇区 (×8)", from: 24, to: 32, color: "#4aa8ff" },
];
// 32proj 中 8..32 通道含 ×8 幅度缩放，条形显示按 /8 归一
const obs32Norm = (r) => (r >= 8 ? v => v / 8 : v => v);
const obs32ColorOf = (r) => {
  for (const g of OBS32_GROUPS) if (r >= g.from && r < g.to) return g.color;
  return "#8b98ab";
};
const hexRgb = (hex) => [
  parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];

function makeBarRow(container, label, idPrefix) {
  const row = document.createElement("div");
  row.className = "bar-row";
  row.innerHTML = `<div class="b-label">${label}</div>
    <div class="bar-track"><div class="bar-fill" id="${idPrefix}-fill"></div></div>
    <div class="b-val" id="${idPrefix}-val">0.00</div>`;
  container.appendChild(row);
  return { fill: $(idPrefix + "-fill"), val: $(idPrefix + "-val") };
}

function buildIOPanels() {
  // 24 维（test5a 射线观测）专用面板：仅在 OBS==24 模式下构建
  if (S.obsMode !== 32) {
    const cF = $("bars-fdir"), cR = $("bars-ray"), cS = $("bars-self"),
          cT = $("bars-tail"), cD = $("bars-fdist");
    S.barsF = [];
    S.barsF.push(makeBarRow(cF, "前方", "if0"));
    S.barsF.push(makeBarRow(cF, "左前", "if1"));
    S.barsF.push(makeBarRow(cF, "右前", "if2"));
    S.barsDist = makeBarRow(cD, "欧氏距离", "if3");
    S.barsR = [];
    for (let i = 0; i < 5; i++) {
      S.barsR.push(makeBarRow(cR, RAY_NAMES[i] + " 路径", "ir" + i + "p"));
      S.barsR.push(makeBarRow(cR, RAY_NAMES[i] + " 食物", "ir" + i + "f"));
    }
    S.barsS = [];
    for (let i = 0; i < 8; i++) S.barsS.push(makeBarRow(cS, SELF_NAMES[i], "is" + i));
    S.barsT = [];
    S.barsT.push(makeBarRow(cT, "尾·前向", "it0"));
    S.barsT.push(makeBarRow(cT, "尾·左向", "it1"));
    buildActBars();
    return;
  }
  // 32 维（test7a 32proj 观测）：隐藏旧 24 维分组，生成通用分组条形
  const body = $("input-body");
  for (const el of body.querySelectorAll(".io-group")) el.style.display = "none";
  let g = $("io-generic32");
  if (g) g.remove();
  g = document.createElement("div");
  g.className = "io-group";
  g.id = "io-generic32";
  body.appendChild(g);
  S.bars32 = [];
  for (const grp of OBS32_GROUPS) {
    const grpDiv = document.createElement("div");
    grpDiv.innerHTML = `<div class="g-label"><span>${grp.name}</span></div>`;
    const barsDiv = document.createElement("div");
    grpDiv.appendChild(barsDiv);
    g.appendChild(grpDiv);
    for (let i = grp.from; i < grp.to; i++) {
      S.bars32.push(makeBarRow(barsDiv, OBS32_LABELS[i], "i32_" + i));
    }
  }
  buildActBars();
}

function buildActBars() {
  // 动作条（两种模式共用；独立出来避免重复构建）
  const cA = $("act-bars");
  if (cA.children.length) return;
  S.actBars = [];
  const actNames = ["前进 Fwd", "左转 Left", "右转 Right"];
  for (let i = 0; i < 3; i++) {
    const row = document.createElement("div");
    row.className = "act-row";
    row.innerHTML = `<div class="act-label"><span><span class="tag" id="atag-${i}">×</span> ${actNames[i]}</span><span id="alog-${i}">0.00</span></div>
      <div class="act-bar-track" id="atrack-${i}"><div class="act-fill" id="afill-${i}"></div></div>`;
    cA.appendChild(row);
    S.actBars.push({ track: $(`atrack-${i}`), fill: $(`afill-${i}`), log: $(`alog-${i}`), tag: $(`atag-${i}`) });
  }
}

function styleFill(fill, color, v) {
  fill.style.background = color;
  const pct = Math.max(0, Math.min(100, v * 100));
  fill.style.width = pct + "%";
  fill.style.opacity = 0.25 + 0.75 * Math.min(1, v);
}

/* ---------------- 画布尺寸自适应 ---------------- */
function resizeHeatCanvas() {
  // heatCanvas：实际像素高度 = 行数 × 每行像素高
  // obs 24 行 / 柱体 N 行 → 按 S.heatRowH 定高，超容器则出现滚动条
  // logits 视图 → 撑满可视区高度
  const cv = $("heatCanvas"), wrap = $("heat-wrap");
  if (!cv || !wrap) return;
  const dpr = window.devicePixelRatio || 1;
  const wrapW = wrap.clientWidth, wrapH = wrap.clientHeight;
  if (wrapW <= 0 || wrapH <= 0) return;
  const rows = S.heatTab === "obs" ? (S.meta ? S.meta.OBS : 24)
             : (S.heatTab === "logits" ? null : (S.meta ? S.meta.N : 256));
  let pxH;
  if (rows === null) pxH = wrapH;                         // logits 撑满可视
  else pxH = Math.max(1, rows * S.heatRowH);              // 严格按行高，不钳制下限 → 可调窄
  cv.style.width = "100%";
  cv.style.height = pxH + "px";
  cv.width = Math.round(wrapW * dpr);
  cv.height = Math.round(pxH * dpr);
}

function resizeCanvases() {
  const dpr = window.devicePixelRatio || 1;
  // game / topo 画布：按容器撑满
  for (const id of ["gameCanvas", "topoCanvas"]) {
    const cv = $(id);
    if (!cv) continue;
    const rect = cv.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) continue;
    cv.width = Math.round(rect.width * dpr);
    cv.height = Math.round(rect.height * dpr);
  }
  resizeHeatCanvas();
  if (S.layout) {
    prepareTopoLayout();
    prepareHeat();
  }
}

/* ---------------- 游戏区：蛇头视线（5 条射线）---------------- */
function rayDirsFromDir(dir) {
  // 返回 [左, 左前, 前, 右前, 右] 五个单位方向（与后端 ray_dirs 合成方式一致）
  const [dy, dx] = dir; // dir = [row, col]
  const left = [-dx, dy];   // 左：row=-dx, col=dy
  const right = [dx, -dy];  // 右：row=dx, col=-dy
  // 斜前方向 = dir + 侧向（与后端 left_dir/right_dir 合成一致）
  return [left,
          [dir[0] + left[0], dir[1] + left[1]],
          [dx, dy],
          [dir[0] + right[0], dir[1] + right[1]],
          right];
}

function drawVision(ctx, fr, G, cell) {
  if (!fr || !fr.obs) return;
  if (S.obsMode === 32) return;   // 5 射线视线仅适用于 24 维射线观测模式
  const [dy, dx] = fr.dir;
  const hx = (fr.body[0][1] + .5) * cell;
  const hy = (fr.body[0][0] + .5) * cell;
  const raydirs = rayDirsFromDir(fr.dir);
  for (let i = 0; i < 5; i++) {
    // obs[4 + i*2] = 自由路径比, obs[4 + i*2 + 1] = 食物信号
    const freeR = fr.obs[4 + i * 2];
    const foodSig = fr.obs[4 + i * 2 + 1];
    const [rdy, rdx] = raydirs[i];
    // 射线长度：自由路径比 × grid；ray 可看到障碍前一格
    const len = Math.max(0.5, Math.min(G, (freeR > 0 ? freeR : 1 / G) * G));
    const ex = hx + rdx * len * cell;
    const ey = hy + rdy * len * cell;

    // 射线主体：半透明虚线
    ctx.strokeStyle = i === 2 ? "rgba(120,220,255,0.55)" : "rgba(120,170,220,0.30)";
    ctx.lineWidth = i === 2 ? 2 : 1.2;
    ctx.setLineDash([4, 5]);
    ctx.beginPath();
    ctx.moveTo(hx, hy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    ctx.setLineDash([]);

    // 终点小圆点：有食物更亮
    ctx.fillStyle = foodSig > 0 ? "rgba(255,210,61,0.95)" : "rgba(160,190,220,0.45)";
    ctx.beginPath();
    ctx.arc(ex, ey, foodSig > 0 ? cell * .14 : cell * .08, 0, Math.PI * 2);
    ctx.fill();

    // 若该射线能看到食物：沿射线标出食物格黄色圆环
    if (foodSig > 0) {
      // foodSig = 1 - food_dist/G；反推 food_dist 格数
      const foodLen = Math.max(1, Math.min(G, Math.round((1 - foodSig) * G)));
      const [fy, fx] = fr.food;
      const fx_ = hx + rdx * foodLen * cell;
      const fy_ = hy + rdy * foodLen * cell;
      ctx.strokeStyle = "rgba(255,210,61,0.9)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(fx_, fy_, cell * .22, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}

/* ---------------- 游戏盘绘制（固定 300×300 内部坐标，依赖宽高比 1:1） ---------------- */
function drawGame() {
  const cv = $("gameCanvas");
  const ctx = cv.getContext("2d");
  const fr = S.cur;
  const G = S.meta ? S.meta.GRID : 10;
  const w = cv.width, h = cv.height, cell = w / G;
  ctx.clearRect(0, 0, w, h);
  for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) {
    ctx.fillStyle = ((r + c) % 2 === 0) ? "#11151d" : "#171c27";
    ctx.fillRect(c * cell, r * cell, cell, cell);
  }
  if (!fr) return;
  ctx.fillStyle = "#ffd23d";
  ctx.beginPath();
  ctx.arc((fr.food[1] + .5) * cell, (fr.food[0] + .5) * cell, cell * .32, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#fff3b0";
  ctx.beginPath();
  ctx.arc((fr.food[1] + .5) * cell, (fr.food[0] + .5) * cell, cell * .1, 0, Math.PI * 2);
  ctx.fill();
  const body = fr.body;
  for (let i = body.length - 1; i >= 0; i--) {
    const seg = body[i];
    const t = i / Math.max(1, body.length - 1);
    const col = i === 0 ? "#e13a3a"
      : `rgb(${Math.round(40 + 90 * t)},${Math.round(120 + 80 * t)},${Math.round(200 + 40 * t)})`;
    const pad = cell * 0.06;
    ctx.fillStyle = col;
    ctx.fillRect(seg[1] * cell + pad, seg[0] * cell + pad, cell - pad * 2, cell - pad * 2);
    ctx.strokeStyle = "#00000044"; ctx.lineWidth = 1;
    ctx.strokeRect(seg[1] * cell + pad, seg[0] * cell + pad, cell - pad * 2, cell - pad * 2);
  }
  const head = body[0];
  const dx = fr.dir[1], dy = fr.dir[0];
  const hx = (head[1] + .5) * cell, hy = (head[0] + .5) * cell;
  ctx.save();
  ctx.translate(hx, hy);
  ctx.rotate(Math.atan2(dy, dx));
  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.moveTo(cell * .32, 0); ctx.lineTo(-cell * .1, -cell * .22); ctx.lineTo(-cell * .1, cell * .22);
  ctx.closePath(); ctx.fill();
  ctx.restore();
  // 蛇的 5 射线视线示意（绘制在最上层）
  drawVision(ctx, fr, G, cell);
}

/* ---------------- 输入/输出面板更新 ---------------- */
function drawIO() {
  const fr = S.cur;
  // 32proj 模式：通用分组条形（8..32 通道 ×8 幅度，按 /8 归一显示）
  if (S.obsMode === 32) {
    const obs = fr ? fr.obs : new Array(32).fill(0);
    for (let i = 0; i < 32; i++) {
      styleFill(S.bars32[i].fill, obs32ColorOf(i), obs32Norm(i)(obs[i]));
      S.bars32[i].val.textContent = obs[i].toFixed(2);
    }
  } else {
    const obs = fr ? fr.obs : new Array(24).fill(0);
    const colorMap = { f: "#3fb950", d: "#f0883e", rp: "#4aa8ff", rf: "#ffd23d", s: "#d07ff5", t: "#3fe0c8" };
    for (let i = 0; i < 3; i++) styleFill(S.barsF[i].fill, colorMap.f, obs[i]);
    styleFill(S.barsDist.fill, colorMap.d, obs[3]);
    for (let i = 0; i < 5; i++) {
      styleFill(S.barsR[i * 2].fill, colorMap.rp, obs[4 + i * 2]);
      styleFill(S.barsR[i * 2 + 1].fill, colorMap.rf, obs[4 + i * 2 + 1]);
    }
    for (let i = 0; i < 8; i++) styleFill(S.barsS[i].fill, colorMap.s, obs[14 + i]);
    styleFill(S.barsT[0].fill, colorMap.t, Math.abs(obs[22]) * 2);
    styleFill(S.barsT[1].fill, colorMap.t, Math.abs(obs[23]) * 2);
    const valEls = [
      S.barsF[0].val, S.barsF[1].val, S.barsF[2].val, S.barsDist.val,
      ...S.barsR.map(b => b.val), ...S.barsS.map(b => b.val), ...S.barsT.map(b => b.val),
    ];
    for (let i = 0; i < 24; i++) valEls[i].textContent = obs[i].toFixed(2);
    if (fr) {
      const fd = [obs[0], obs[1], obs[2]];
      const dirs = ["前方", "左前", "右前"];
      const hits = dirs.filter((_, k) => fd[k] > 0);
      $("g-fdir").textContent = hits.length ? hits.join("/") : "无";
      $("g-fdist").textContent = (obs[3] * S.meta.GRID * Math.SQRT2).toFixed(1) + " 格";
    }
  }
  if (fr) {
    for (let i = 0; i < 3; i++) {
      const lg = fr.logits[i];
      const active = fr.action === i;
      S.actBars[i].log.textContent = lg.toFixed(3);
      S.actBars[i].track.classList.toggle("active", active);
      S.actBars[i].tag.textContent = active ? "▶ " + ["Fwd", "Left", "Right"][i] : ["Fwd", "Left", "Right"][i];
      S.actBars[i].tag.style.background = active ? "#ffd33d" : "transparent";
      S.actBars[i].tag.style.color = active ? "#111" : "#8b98ab";
      const norm = Math.max(-3, Math.min(3, lg)) / 3;
      S.actBars[i].fill.style.background = norm >= 0
        ? `rgba(248,81,73,${Math.min(1, norm + .08)})`
        : `rgba(74,168,255,${Math.min(1, -norm + .08)})`;
      S.actBars[i].fill.style.left = (norm >= 0 ? 50 : 50 + norm * 50) + "%";
      S.actBars[i].fill.style.width = Math.abs(norm) * 50 + "%";
    }
    $("cur-act").textContent = ["前进 (Fwd)", "左转 (Left)", "右转 (Right)"][fr.action];
    $("f-0").textContent = fr.fatigue[0].toFixed(1);
    $("f-1").textContent = fr.fatigue[1].toFixed(1);
    $("f-2").textContent = fr.fatigue[2].toFixed(1);
  }
}

/* ---------------- 拓扑网络 ---------------- */
function prepareTopoLayout() {
  const L = S.layout;
  const cv = $("topoCanvas");
  const ctx = cv.getContext("2d");
  const tauMin = S.meta.tau_min, tauMax = S.meta.tau_max;
  const nodeColors = new Array(L.col_x.length);
  for (let i = 0; i < nodeColors.length; i++) {
    const t = (L.tau[i] - tauMin) / (tauMax - tauMin);
    nodeColors[i] = idx2color(LUT_PLASMA, t);
  }
  const pad = 14;   // 缩小边距 → 拓扑占比放大
  // 收紧后端 range 的空白边距：网络本体占满卡片大部分面积
  // 后端 range: x0=-1.1, x1=OUT_X+0.9, y0=-0.8, y1=total_height+0.8
  const OUT_X = L.range.x1 - 0.9;              // 输出层原始 x 坐标
  const outPad = 0.45;                          // 输出层右侧留白
  const yExtra = 0.45;                          // 上下留白
  const rng = {
    x0: -0.70,                                  // 输入层左侧留白（供标签）
    y0: -yExtra + 0.30,                         // ≈ -0.15，轻微上移让顶部不裁
    x1: OUT_X + outPad,                         // = OUT_X + 0.45
    y1: (L.range.y1 - 0.8) + yExtra,            // = total_height + 0.45
  };
  const outX = OUT_X;                           // 保持原输出层坐标
  const ZOOM = 1.13;                            // 放大系数：网络本体占满卡片大部分面积
  const dw = rng.x1 - rng.x0, dh = rng.y1 - rng.y0;
  const s_ = Math.min((cv.width - pad * 2) / dw, (cv.height - pad * 2) / dh) * ZOOM;
  const offX = (cv.width - dw * s_) / 2 - rng.x0 * s_;
  const offY = (cv.height - dh * s_) / 2 - rng.y0 * s_;
  const P = (x, y) => [offX + x * s_, offY + y * s_];

  const inPos = L.in_y.map(y => P(0, y));
  const outPos = L.out_y.map(y => P(outX, y));
  const colPos = L.col_x.map((x, i) => P(x, L.col_y[i]));

  // ---- 静态层 ----
  const st = document.createElement("canvas");
  st.width = cv.width; st.height = cv.height;
  const sctx = st.getContext("2d");
  sctx.clearRect(0, 0, st.width, st.height);

  const commColors = ["#f14c4c", "#2e8de6", "#35c26b", "#f0a12f", "#b05ce0",
                      "#36c5c0", "#e06db0", "#9aa82e", "#5b7dff", "#d9a83e"];
  for (const c of L.communities) {
    const [ax, ay] = P(c.x, c.y);
    sctx.font = "bold 10px Segoe UI, Microsoft YaHei";
    sctx.textAlign = "center";
    sctx.fillStyle = commColors[c.color % commColors.length];
    sctx.fillText("C" + c.id, ax, ay - 14);
  }
  const obsLabels = S.obsMode === 32 ? OBS32_LABELS : ["食·前", "食·左前", "食·右前", "食·距离",
    "射·左·径", "射·左·食", "射·左前·径", "射·左前·食", "射·前·径", "射·前·食",
    "射·右前·径", "射·右前·食", "射·右·径", "射·右·食",
    "身·前", "身·左前", "身·左", "身·左后", "身·后", "身·右后", "身·右", "身·右前",
    "尾·前", "尾·左"];
  const inR = 6;
  sctx.font = "9px Segoe UI, Microsoft YaHei";
  for (let j = 0; j < L.in_y.length; j++) {
    const [x, y] = inPos[j];
    sctx.fillStyle = "#3fb950";
    sctx.fillRect(x - inR, y - inR, inR * 2, inR * 2);
    sctx.strokeStyle = "#0a0d13"; sctx.lineWidth = 1;
    sctx.strokeRect(x - inR, y - inR, inR * 2, inR * 2);
    sctx.textAlign = "right"; sctx.fillStyle = "#8b98ab";
    sctx.fillText(j + "·" + obsLabels[j], x - inR - 3, y + 3);
  }
  const actNames = ["Fwd", "Left", "Right"];
  for (let i = 0; i < L.out_y.length; i++) {
    const [x, y] = outPos[i];
    sctx.save(); sctx.translate(x, y); sctx.rotate(Math.PI / 4);
    sctx.fillStyle = "#f0883e";
    sctx.fillRect(-inR - 1, -inR - 1, inR * 2 + 2, inR * 2 + 2);
    sctx.fillStyle = "#f0a12f";
    sctx.fillRect(-inR, -inR, inR * 2, inR * 2);
    sctx.restore();
    sctx.textAlign = "left"; sctx.fillStyle = "#8b98ab";
    sctx.fillText(i + "·" + actNames[i], x + inR + 3, y + 3);
  }
  // 输入边
  for (const [col, inIdx, nw] of L.in_edges) {
    const [x0, y0] = inPos[inIdx], [x1, y1] = colPos[col];
    sctx.strokeStyle = `rgba(63,185,80,${(0.10 + 0.55 * nw).toFixed(3)})`;
    sctx.lineWidth = 0.4 + 1.4 * nw;
    sctx.beginPath(); sctx.moveTo(x0, y0); sctx.lineTo(x1, y1); sctx.stroke();
  }
  // 输出边
  for (const [outIdx, col, nw] of L.out_edges) {
    const [x0, y0] = colPos[col], [x1, y1] = outPos[outIdx];
    sctx.strokeStyle = `rgba(230,80,230,${(0.10 + 0.55 * nw).toFixed(3)})`;
    sctx.lineWidth = 0.4 + 1.4 * nw;
    sctx.beginPath(); sctx.moveTo(x0, y0); sctx.lineTo(x1, y1); sctx.stroke();
  }
  // 递归边
  for (const [src, tgt, w, alpha] of L.rec_edges) {
    const [x0, y0] = colPos[src], [x1, y1] = colPos[tgt];
    const dist = Math.hypot(x1 - x0, y1 - y0);
    const curve = (0.10 + 0.08 * Math.min(3, dist)) * 30;
    const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
    sctx.strokeStyle = w >= 0
      ? `rgba(248,81,73,${(0.08 + 0.5 * alpha).toFixed(3)})`
      : `rgba(74,120,255,${(0.08 + 0.5 * alpha).toFixed(3)})`;
    sctx.lineWidth = 0.3 + 1.3 * alpha;
    sctx.beginPath();
    sctx.moveTo(x0, y0);
    sctx.quadraticCurveTo(mx + curve * 0.15, my - curve, x1, y1);
    sctx.stroke();
  }

  S.topoStatic = st;
  S.topoCtx = ctx;
  S.colPos = colPos;
  S.inPos = inPos;
  S.outPos = outPos;
  S.nodeColors = nodeColors;
}

function drawTopo() {
  const ctx = S.topoCtx;
  if (!ctx || !S.topoStatic) return;
  ctx.clearRect(0, 0, S.topoStatic.width, S.topoStatic.height);
  ctx.drawImage(S.topoStatic, 0, 0);
  const E = S.histE[S.histLen - 1];
  if (!E) return;
  const N = S.layout.col_x.length;
  // 本帧 E 动态 min-max 归一化：每帧都用满亮度范围，激活模式清晰可辨
  let eMin = 1, eMax = 0;
  for (let i = 0; i < N; i++) {
    const v = E[i];
    if (v < eMin) eMin = v;
    if (v > eMax) eMax = v;
  }
  const eSpan = (eMax - eMin) || 1e-9;
  // 柱节点：固定半径纯填色圆，激活度(E 归一化)直接映射为整圆亮度
  // 最暗 ≈ 黑，最亮 = 全亮 tau 色（alpha 始终 1，仅亮度变化，无中央白点）
  for (let i = 0; i < N; i++) {
    const [x, y] = S.colPos[i];
    const act = Math.max(0, Math.min(1, (E[i] - eMin) / eSpan));
    const base = S.nodeColors[i];
    const r = 5.2;   // 柱节点略放大
    const bright = 0.10 + 0.90 * act;   // 亮度系数：0.10(近黑) ~ 1.0(全亮)
    ctx.fillStyle = rgba([
      base[0] * bright,
      base[1] * bright,
      base[2] * bright,
    ], 1);
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#0a0d13"; ctx.lineWidth = 1; ctx.stroke();
  }
  const obs = S.histObs[S.histLen - 1];
  if (obs) {
    for (let j = 0; j < S.inPos.length; j++) {
      const [x, y] = S.inPos[j];
      const v = Math.max(0, Math.min(1, obs[j]));
      if (v > 0.05) {
        const g = ctx.createRadialGradient(x, y, 0, x, y, 11);
        g.addColorStop(0, `rgba(70,255,130,${0.3 + 0.4 * v})`);
        g.addColorStop(1, "rgba(70,255,130,0)");
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(x, y, 11, 0, Math.PI * 2); ctx.fill();
      }
    }
  }
  const lg = S.histLogits[S.histLen - 1];
  if (lg) {
    const curAction = S.histActions[S.histLen - 1];
    for (let i = 0; i < S.outPos.length; i++) {
      const [x, y] = S.outPos[i];
      const v = lg[i];
      // 亮度映射：|logits|/3 → 0.15(几乎暗) ~ 1.0(全亮)，正向红、负向蓝
      const act_col = v >= 0 ? [248, 120, 90] : [90, 160, 255];
      const bright = 0.15 + 0.85 * Math.min(1, Math.abs(v) / 3);
      ctx.fillStyle = rgba([act_col[0] * bright, act_col[1] * bright, act_col[2] * bright], 1);
      ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
      if (i === curAction) {
        // 当前选中的动作：黄色高亮描边
        ctx.strokeStyle = "#ffd23d";
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, y, 9.5, 0, Math.PI * 2); ctx.stroke();
      } else {
        ctx.strokeStyle = "#0a0d13"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.stroke();
      }
    }
  }
}

/* ---------------- 时序热图（tab 子选项独占整卡高度） ---------------- */
const OBS_ROW_LABELS = ["F·前", "F·左前", "F·右前", "F·距离",
  "R·左·径", "R·左·食", "R·左前·径", "R·左前·食", "R·前·径", "R·前·食",
  "R·右前·径", "R·右前·食", "R·右·径", "R·右·食",
  "S·前", "S·左前", "S·左", "S·左后", "S·后", "S·右后", "S·右", "S·右前", "尾·前", "尾·左"];
const OBS_ROW_COLORS = [
  ["#3fb950", "#3fb950", "#3fb950"],
  ["#f0883e"],
  ["#4aa8ff", "#ffd23d", "#4aa8ff", "#ffd23d", "#4aa8ff", "#ffd23d", "#4aa8ff", "#ffd23d", "#4aa8ff", "#ffd23d"],
  ["#d07ff5", "#d07ff5", "#d07ff5", "#d07ff5", "#d07ff5", "#d07ff5", "#d07ff5", "#d07ff5"],
  ["#3fe0c8", "#3fe0c8"],
];
function obsRowColorOf(r) {
  if (r < 3) return OBS_ROW_COLORS[0][r];
  if (r === 3) return OBS_ROW_COLORS[1][0];
  if (r < 14) return OBS_ROW_COLORS[2][r - 4];
  if (r < 22) return OBS_ROW_COLORS[3][r - 14];
  return OBS_ROW_COLORS[4][r - 22];
}

function prepareHeat() {
  const cv = $("heatCanvas");
  S.heatCv = cv;
  S.heatCtx = cv.getContext("2d");
  S.brainLUTs = {
    E:   { key: "histE",   lut: LUT_VIRIDIS, vmin: 0,              vmax: 1,              label: "柱活动 E" },
    tau: { key: "histTau", lut: LUT_PLASMA,  vmin: S.meta.tau_min, vmax: S.meta.tau_max, label: "有效 τ_eff" },
    ex:  { key: "histEx",  lut: LUT_REDS,    vmin: 0,              vmax: 0.06,           label: "兴奋激素" },
    inh: { key: "histInH", lut: LUT_BLUES,   vmin: 0,              vmax: 0.06,           label: "抑制激素" },
  };
}

function drawHeat() {
  const ctx = S.heatCtx;
  if (!ctx) return;
  const cv = S.heatCv;
  const W = cv.width, H = cv.height;
  if (W <= 0 || H <= 0) return;
  const colW = W / MAX_HIST;
  const startCol = MAX_HIST - S.histLen;
  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, W, H);

    // ---- 输入观测视图 ----
  if (S.heatTab === "obs") {
    const rows = S.meta ? S.meta.OBS : 24;
    const mode32 = S.obsMode === 32;
    const rowH = H / rows;
    const lines = (mode32 ? [0, 4, 8, 16, 24] : [0, 3, 4, 14, 22]).map(v => (v + 0.5) * rowH);
    const img = ctx.createImageData(W, H);
    const d = img.data;
    for (let p = 0; p < W * H; p++) { d[p * 4] = 13; d[p * 4 + 1] = 17; d[p * 4 + 2] = 23; d[p * 4 + 3] = 255; }
    for (let c = 0; c < S.histLen; c++) {
      const obs = S.histObs[c];
      if (!obs) continue;
      const px0 = Math.round((startCol + c) * colW);
      const px1 = Math.round((startCol + c + 1) * colW);
      // 连续行填充：上一行终点 = 下一行起点，行间零空隙
      for (let r = 0; r < rows; r++) {
        const y0 = Math.round(r * rowH);
        const y1 = Math.round((r + 1) * rowH);
        const rgb = mode32 ? hexRgb(obs32ColorOf(r)) : obsRowColorOf(r);
        const v0 = Math.max(0, Math.min(1, obs[r]));
        const v = mode32 ? obs32Norm(r)(v0) : v0;
        const col = [rgb[0] * (0.12 + 0.88 * v), rgb[1] * (0.12 + 0.88 * v), rgb[2] * (0.12 + 0.88 * v)];
        for (let yy = y0; yy < y1; yy++) {
          for (let px = px0; px < px1; px++) {
            const idx = (yy * W + px) * 4;
            d[idx] = col[0]; d[idx + 1] = col[1]; d[idx + 2] = col[2]; d[idx + 3] = 255;
          }
        }
      }
    }
    ctx.putImageData(img, 0, 0);
    ctx.strokeStyle = "rgba(255,255,255,0.50)";
    ctx.lineWidth = 1;
    for (const y of lines) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
    ctx.font = "9px Consolas, monospace"; ctx.textAlign = "right";
    if (rowH > 14) {
      for (let r = 0; r < rows; r++) {
        ctx.fillStyle = "#aebccd";
        ctx.fillText(mode32 ? OBS32_LABELS[r] : OBS_ROW_LABELS[r], W - 4, (r + 0.35) * rowH + 8);
      }
    }
    // 动作切换绿虚线 + 动作星标
    drawActionOverlay(ctx, startCol, colW, H);
    return;
  }

  // ---- logits 折线视图 ----
  if (S.heatTab === "logits") {
    const lgMarg = 30, lgTop = 16, lgBottom = 26;
    const plotH = H - lgTop - lgBottom;
    const plotW = W - lgMarg * 2;
    const colW2 = plotW / MAX_HIST;
    const y0 = lgTop + plotH / 2;
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.beginPath(); ctx.moveTo(lgMarg, y0); ctx.lineTo(W - lgMarg, y0); ctx.stroke();
    const actionColors = ["#ffd23d", "#3fb950", "#f85149"];
    const actNames = ["Fwd", "Left", "Right"];
    for (let a = 0; a < 3; a++) {
      ctx.strokeStyle = actionColors[a];
      ctx.lineWidth = 2;
      ctx.beginPath();
      let started = false;
      for (let c = 0; c < S.histLen; c++) {
        const lg = S.histLogits[c];
        const x = lgMarg + (startCol + c + 0.5) * colW2;
        if (!lg) { started = false; continue; }
        const norm = Math.max(-1, Math.min(1, lg[a] / 3));
        const yTop = y0 - norm * (plotH * 0.38);
        if (!started) { ctx.moveTo(x, yTop); started = true; } else ctx.lineTo(x, yTop);
      }
      ctx.stroke();
      // 右侧图例
      ctx.fillStyle = actionColors[a];
      ctx.fillRect(W - lgMarg + 4, lgTop + 6 + a * 14, 8, 8);
      ctx.fillText(actNames[a], W - lgMarg + 16, lgTop + 14 + a * 14);
    }
    drawActionOverlay(ctx, lgMarg + startCol * colW2, colW2, H);
    return;
  }

  // ---- 柱体热力图视图（E / tau / ex / inh） ----
  const spec = S.brainLUTs[S.heatTab];
  if (!spec) return;
  const N = S.meta.N;
  const data = S[spec.key];
  const rowH = H / N;
  const vspan = (spec.vmax - spec.vmin) || 1e-9;
  const img = ctx.createImageData(W, H);
  const d = img.data;
  for (let p = 0; p < W * H; p++) { d[p * 4] = 13; d[p * 4 + 1] = 17; d[p * 4 + 2] = 23; d[p * 4 + 3] = 255; }
  for (let c = 0; c < S.histLen; c++) {
    const arr = data[c];
    if (!arr) continue;
    const px0 = Math.round((startCol + c) * colW);
    const px1 = Math.round((startCol + c + 1) * colW);
    // 连续行填充：上一行终点 = 下一行起点，行间零空隙
    for (let r = 0; r < N; r++) {
      const v = arr[r];
      let t = (v - spec.vmin) / vspan;
      if (t < 0) t = 0; else if (t > 1) t = 1;
      const col = idx2color(spec.lut, t);
      const y0 = Math.round(r * rowH);
      const y1 = Math.round((r + 1) * rowH);
      for (let yy = y0; yy < y1; yy++) {
        for (let px = px0; px < px1; px++) {
          const idx = (yy * W + px) * 4;
          d[idx] = col[0]; d[idx + 1] = col[1]; d[idx + 2] = col[2]; d[idx + 3] = 255;
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  // 左标签（与连续行填充的 y0=Math.round(r*rowH) 对齐）
  ctx.font = "8px Consolas, monospace";
  ctx.textAlign = "left";
  ctx.fillStyle = "#77869c";
  for (let n = 0; n < N; n += 64) {
    if (n === N) continue;
    ctx.fillText(String(n), 6, Math.round(n * rowH) + 7);
  }
  // 社区分隔线（画在两行边界，与连续行填充的边界对齐）
  ctx.strokeStyle = "rgba(255,255,255,0.45)";
  ctx.lineWidth = 1;
  for (const bd of S.commBd) {
    const y = Math.round(bd * rowH);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  drawActionOverlay(ctx, startCol, colW, H);
}

/* 动作切换绿虚线 + 实际动作星标（叠加在热图上） */
function drawActionOverlay(ctx, startCol, colW, H) {
  for (let c = 0; c < S.histLen; c++) {
    const sw = S.histSwitch[c];
    const act = S.histActions[c];
    if (act === undefined || act === null) continue;
    const x = (startCol + c) * colW + colW / 2;
    if (sw) {
      ctx.strokeStyle = "rgba(80,255,160,0.55)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    ctx.fillStyle = "#000";
    ctx.strokeStyle = "#ffe14d"; ctx.lineWidth = 1;
    star(ctx, x, 9, 3);
  }
}

function star(ctx, x, y, r) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const ang = -Math.PI / 2 + i * Math.PI / 5;
    const rr = i % 2 === 0 ? r : r * 0.45;
    const px = x + Math.cos(ang) * rr, py = y + Math.sin(ang) * rr;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

/* ---------------- 数据处理 ---------------- */
function pushFrame(fr) {
  const order = S.order;
  S.histObs.push(fr.obs);
  const E = new Array(S.meta.N), tau = new Array(S.meta.N),
        ex = new Array(S.meta.N), inh = new Array(S.meta.N);
  for (let i = 0; i < S.meta.N; i++) {
    E[i] = fr.E[order[i]]; tau[i] = fr.tau[order[i]];
    ex[i] = fr.hormone_ex[order[i]]; inh[i] = fr.hormone_in[order[i]];
  }
  const lastE = S.histE[S.histLen - 1];
  if (lastE) for (let i = 0; i < S.meta.N; i++) E[i] = 0.5 * E[i] + 0.5 * lastE[i];
  S.histE.push(E);
  S.histTau.push(tau);
  S.histEx.push(ex);
  S.histInH.push(inh);
  S.histLogits.push(fr.logits);
  S.histActions.push(fr.action);
  const prevA = S.histActions[S.histLen - 2];
  S.histSwitch.push(prevA !== undefined && prevA !== fr.action);
  if (S.histObs.length > MAX_HIST) {
    S.histObs.shift(); S.histE.shift(); S.histTau.shift();
    S.histEx.shift(); S.histInH.shift(); S.histLogits.shift();
    S.histActions.shift(); S.histSwitch.shift();
  }
  S.histLen = S.histObs.length;
}

/* ---------------- 主渲染循环（含连接超时看门狗） ---------------- */
let rafId = null;
let lastFrameAt = 0;
function renderLoop() {
  // 超过 3 秒无新帧 → 提示可能后端断开
  if (S.cur && Date.now() - lastFrameAt > 3000 && S.connState === "connected") {
    setConn("reconnecting", "⚠ 数据流中断（后端可能已退出）");
    setHint("请重新运行 python brain_visualizer.py");
  }
  if (S.cur) {
    drawGame();
    drawIO();
    drawTopo();
    if (S.histLen > 0) drawHeat();
  }
  rafId = requestAnimationFrame(renderLoop);
}

/* ---------------- SSE ---------------- */
let es = null;
function connectSSE() {
  es = new EventSource("/stream");
  es.onopen = () => {
    setConn("connected", "已连接后端");
    setHint("蛇由进化出的最优模型自动控制 · 死亡自动重开");
  };
  es.onmessage = ev => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "init") {
      handleInit(msg);
      setConn("connected", "已连接后端");
    } else if (msg.type === "frame") {
      lastFrameAt = Date.now();
      handleFrame(msg);
    }
  };
  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) {
      setConn("reconnecting", "✗ 无法连接后端");
      setHint("请先运行 python brain_visualizer.py，再刷新页面");
    } else {
      setConn("reconnecting", "⚠ 连接中断，正在重连…");
    }
  };
}

function clearHistory() {
  S.histObs = []; S.histE = []; S.histTau = []; S.histEx = []; S.histInH = [];
  S.histLogits = []; S.histActions = []; S.histSwitch = []; S.histLen = 0;
}

function handleInit(msg) {
  S.layout = msg.layout;
  S.meta = msg.meta;
  S.order = msg.layout.col_order;
  S.episode = msg.episode;
  S.obsMode = msg.meta.OBS === 24 ? 24 : 32;

  // 模型下拉框：用后端枚举的模型列表重建，并选中当前模型
  const sel = $("modelSelect");
  if (msg.models && msg.models.length) {
    sel.innerHTML = "";
    for (const [k, label] of msg.models) {
      const o = document.createElement("option");
      o.value = k;
      o.textContent = label;
      sel.appendChild(o);
    }
  }
  const wantKey = S.urlModel || msg.meta.model_key;
  if (wantKey && [...sel.options].some(o => o.value === wantKey)) sel.value = wantKey;

  // 面板文案随观测维度/引擎自适应（7g 为曼哈顿度量观测）
  $("input-sub").textContent = msg.meta.OBS + " 维" + (msg.meta.engine === "7g" ? " · 曼哈顿" : "");
  const hto = $("heatTabObs");
  if (hto) hto.textContent = "输入观测 " + msg.meta.OBS + " 行";
  buildIOPanels();
  S.commBd = [];
  let acc = 0;
  for (const c of msg.layout.communities) {
    acc += c.size;
    if (acc < msg.meta.N) S.commBd.push(acc);
  }
  clearHistory();
  S.cur = null;

  $("m-model").textContent = msg.meta.engine && msg.meta.engine !== "einbrain"
    ? msg.meta.engine + " · " + msg.meta.model : msg.meta.model;
  $("m-n").textContent = msg.meta.N;
  $("m-k").textContent = "K=" + msg.meta.FRAME_RATE + "·衰减" + msg.meta.INPUT_DECAY;
  $("m-food").textContent = msg.meta.food;
  $("m-steps").textContent = msg.meta.steps;
  $("st-ep").textContent = msg.episode;

  resizeCanvases();
  prepareTopoLayout();
  prepareHeat();
}

function handleFrame(fr) {
  if (fr.episode !== S.episode) {
    clearHistory();
    S.episode = fr.episode;
    $("st-ep").textContent = fr.episode;
  }
  S.cur = fr;
  $("st-score").textContent = fr.score;
  $("st-steps").textContent = fr.steps;
  $("game-status").classList.toggle("show", fr.done);
  pushFrame(fr);
}

/* ---------------- 控制（带即时反馈） ---------------- */
function flash(btn) {
  const old = btn.textContent;
  btn.dataset.old = old;
  btn.textContent = "✓ 已发送";
  setTimeout(() => {
    if (btn.dataset.old) btn.textContent = btn.dataset.old;
  }, 800);
}

function doControl(qs, btn) {
  if (btn) flash(btn);
  fetch("/api/control?" + qs)
    .then(r => r.json())
    .catch(() => {
      setConn("reconnecting", "✗ 控制失败（后端未运行）");
      setHint("请先运行 python brain_visualizer.py");
    });
}

window.addEventListener("DOMContentLoaded", () => {
  // 协议检测：禁止 file:// 直接打开
  if (location.protocol === "file:") {
    protocolError();
    return;
  }

  buildIOPanels();

  // 热力图 tab 子选项切换
  $("heat-tabs").addEventListener("click", e => {
    const btn = e.target.closest("button[data-heat]");
    if (!btn) return;
    S.heatTab = btn.dataset.heat;
    document.querySelectorAll("#heat-tabs button").forEach(b =>
      b.classList.toggle("active", b === btn));
    resizeCanvases();   // 切换视图后重算画布高度（行数不同）
  });

  // 热图行高实时调节（滑块）→ 只重算热图画布，不动拓扑与游戏盘
  $("heatHeightSlider").addEventListener("input", e => {
    S.heatRowH = parseInt(e.target.value, 10);
    $("heatHVal").textContent = S.heatRowH + "px";
    resizeHeatCanvas();
  });
  $("heatHVal").textContent = S.heatRowH + "px";

  $("btnNew").onclick = () => {
    doControl("action=new", $("btnNew"));
    setHint("已请求新局…");
  };
  $("btnPause").onclick = () => {
    S.paused = !S.paused;
    const btn = $("btnPause");
    btn.textContent = S.paused ? "▶ 继续" : "⏸ 暂停";
    btn.classList.toggle("pressed", S.paused);
    doControl("action=" + (S.paused ? "pause" : "resume"), null);
  };
  $("speedSlider").oninput = e => {
    S.speed = parseFloat(e.target.value);
    fetch("/api/control?speed=" + S.speed).catch(() => {});
  };
  $("modelSelect").onchange = e => {
    const v = e.target.value;
    $("modelSelect").disabled = true;
    setHint("正在加载模型，请稍候…");
    fetch("/api/init?model=" + v)
      .then(r => r.json())
      .then(msg => {
        handleInit(msg);
        $("modelSelect").disabled = false;
        fetch("/api/control?action=new").catch(() => {});
        setHint("模型已切换，新局开始");
      })
      .catch(() => {
        $("modelSelect").disabled = false;
        protocolError();
      });
  };
  // URL 预选模型（兼容旧值 64/256 → 5a；init 到达后再真正选中）
  const m = new URLSearchParams(location.search).get("model");
  if (m === "64" || m === "256") S.urlModel = "5a";
  else if (m) S.urlModel = m;

  // 窗口尺寸变化 → 重设画布
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resizeCanvases, 150);
  });

  setConn("connecting", "正在连接后端…");
  connectSSE();
  renderLoop();
});