"use strict";
/* Thin view layer: controls are generated from the kernel schema and every
   image / trace / repair comes from the stochastic-medium JSON API. */

const state = {
  schema: [],
  config: {},
  cycleId: 0,
  accTimer: null,
  lastRender: null,
  validationLoaded: false,
};

const $ = (id) => document.getElementById(id);

async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  let data = null;
  try { data = await resp.json(); } catch (_e) { /* non-json */ }
  if (!resp.ok) {
    const err = new Error((data && data.error && data.error.message) ||
                          `HTTP ${resp.status}`);
    err.body = data;
    throw err;
  }
  return data;
}
async function getJSON(url) {
  const resp = await fetch(url);
  return resp.json();
}

function setDot(kind, text) {
  const dot = $("kernelDot");
  dot.className = "dot dot-" + kind;
  if (text) $("kernelStatus").textContent = text;
}

/* ---------------------------------------------------------------- schema */
function buildForm(schema) {
  state.schema = schema;
  const form = $("controlForm");
  form.innerHTML = "";
  for (const ctl of schema) {
    state.config[ctl.id] = ctl.default;
    const row = document.createElement("div");
    row.className = "control-item";
    const label = document.createElement("label");
    label.htmlFor = "ctl-" + ctl.id;

    const name = document.createElement("span");
    name.textContent = ctl.label;
    label.appendChild(name);

    if (ctl.type === "range") {
      const val = document.createElement("span");
      val.className = "val";
      val.id = "val-" + ctl.id;
      val.textContent = formatRangeValue(ctl, ctl.default);
      label.appendChild(val);
      row.appendChild(label);

      const input = document.createElement("input");
      input.type = "range";
      input.id = "ctl-" + ctl.id;
      input.min = ctl.min; input.max = ctl.max; input.step = ctl.step;
      input.value = ctl.default;
      input.addEventListener("input", () => {
        const v = parseFloat(input.value);
        state.config[ctl.id] = ctl.step >= 1 ? Math.round(v) : v;
        val.textContent = formatRangeValue(ctl, state.config[ctl.id]);
        scheduleRender(260);
      });
      row.appendChild(input);
      if (ctl.hint || ctl.unit) {
        const h = document.createElement("div");
        h.className = "hint-line";
        h.textContent = [ctl.unit ? "单位 " + ctl.unit : "", ctl.hint || ""]
          .filter(Boolean).join(" · ");
        row.appendChild(h);
      }
    } else if (ctl.type === "select") {
      row.appendChild(label);
      const sel = document.createElement("select");
      sel.id = "ctl-" + ctl.id;
      for (const opt of ctl.options) {
        const o = document.createElement("option");
        o.value = opt.value; o.textContent = opt.label;
        if (opt.value === ctl.default) o.selected = true;
        sel.appendChild(o);
      }
      sel.addEventListener("change", () => {
        state.config[ctl.id] = sel.value;
        scheduleRender(0);
      });
      row.appendChild(sel);
    }
    form.appendChild(row);
  }
}

function formatRangeValue(ctl, v) {
  const numeric = ctl.step >= 1 ? String(Math.round(v)) : Number(v).toFixed(2);
  return ctl.unit ? `${numeric} ${ctl.unit}` : numeric;
}

function applyConfigToControls(cfg) {
  Object.assign(state.config, cfg);
  for (const ctl of state.schema) {
    const el = $("ctl-" + ctl.id);
    if (!el || cfg[ctl.id] === undefined) continue;
    el.value = cfg[ctl.id];
    const val = $("val-" + ctl.id);
    if (val && ctl.type === "range") {
      val.textContent = formatRangeValue(ctl, cfg[ctl.id]);
    }
  }
}

/* ----------------------------------------------------------- render loop */
function scheduleRender(delay) {
  clearTimeout(state.accTimer);
  state.cycleId++;
  $("controlNote").textContent = "参数已变更，将由随机介质内核重新追踪…";
  const myCycle = state.cycleId;
  state.accTimer = setTimeout(() => {
    if (myCycle === state.cycleId) renderFirstPass();
  }, delay);
}

async function renderFirstPass() {
  clearTimeout(state.accTimer);
  state.cycleId++;
  const myCycle = state.cycleId;
  $("errorPanel").classList.add("hidden");
  $("renderBtn").disabled = true;
  $("renderBusy").classList.remove("hidden");
  setDot("busy", "内核追踪中…");
  try {
    const payload = Object.assign({}, state.config, { accumulate: false });
    const r = await postJSON("/api/render", payload);
    if (myCycle !== state.cycleId) return;
    applyRender(r);
    setDot("ok", "内核正常 · 已连接");
    continueAccumulation(myCycle);
  } catch (err) {
    if (myCycle !== state.cycleId) return;
    handleKernelError(err);
  } finally {
    if (myCycle === state.cycleId) $("renderBtn").disabled = false;
  }
}

async function continueAccumulation(myCycle) {
  if (!$("accToggle").checked || !state.lastRender) {
    $("renderBusy").classList.add("hidden");
    return;
  }
  const r0 = state.lastRender;
  const cap = r0.config.mode === "multi" ? 6 : 128;
  if (r0.total_spp >= cap) {
    $("renderBusy").classList.add("hidden");
    appendStatNote(`已达到累积上限 ${cap} 样本；取消再勾选或改参数可继续。`);
    return;
  }
  try {
    const r = await postJSON("/api/render",
      Object.assign({}, state.config, { accumulate: true }));
    if (myCycle !== state.cycleId) return;
    applyRender(r);
    state.accTimer = setTimeout(() => continueAccumulation(myCycle), 40);
  } catch (err) {
    if (myCycle !== state.cycleId) return;
    handleKernelError(err);
  }
}

function appendStatNote(text) {
  $("controlNote").textContent = text;
}

function applyRender(r) {
  state.lastRender = r;
  $("renderImage").src = r.image;
  $("weightsImage").src = r.weights_image;
  $("candidatesImage").src = r.candidates_image;
  $("thumbWeights").src = r.weights_image;
  $("thumbCand").src = r.candidates_image;
  $("renderBusy").classList.add("hidden");
  $("controlNote").textContent = "";
  renderStats(r);
  const wh = $("weightHint");
  wh.textContent = `透射权重范围 [${r.stats.weight_min.toFixed(4)}, ` +
    `${r.stats.weight_max.toFixed(4)}]；候选热图 99 分位上限 ` +
    `${r.stats.candidate_heatmap_vmax.toFixed(2)}。` +
    `当前累计 ${r.total_spp} 样本 / 像素（${r.accumulated_passes} 轮）。`;
  refreshDiagnostics(true);
}

function chip(label, value, cls) {
  return `<span class="stat">${label} <b class="${cls || ""}">${value}</b></span>`;
}

function renderStats(r) {
  const s = r.stats;
  const strips = [];
  strips.push(chip("累计样本/像素", r.total_spp));
  strips.push(chip("上一轮耗时", s.elapsed_seconds_last_pass + " s"));
  strips.push(chip("平均候选/光线", Number(s.mean_candidates).toFixed(2)));
  strips.push(chip("最大候选", s.max_candidates));
  strips.push(chip("波前迭代轮数", s.iterations));
  if (s.real_events !== null && s.real_events !== undefined) {
    strips.push(chip("真实事件", s.real_events));
    strips.push(chip("空事件", s.null_events));
  }
  if (s.pmax) {
    const cls = s.pmax > 1 + 1e-6 ? "bad" : "good";
    strips.push(chip("max p", Number(s.pmax).toFixed(4), cls));
  }
  if (s.tolerance_clamps) {
    strips.push(chip("容差内夹紧", s.tolerance_clamps, "warn"));
  }
  if (s.scalar_multi_bounce) {
    strips.push(chip("模式", "标量有限反弹预览 " +
      s.preview_width + "×" + s.preview_height, "warn"));
  }
  if (s.repairs && s.repairs.length) {
    strips.push(chip("内核修复", s.repairs.length, "warn"));
  }
  $("statsStrip").innerHTML = strips.join("");
}

/* ---------------------------------------------------------- error feedback */
const ACTION_LABELS = {
  repartition: "重分区后重试",
  discard_resample: "丢弃该段并按安全主界重采样",
  clamp_demo: "反面演示：夹紧 p（有偏！）",
  analytic_handoff: "解析反演精确交接尾部",
  lower_density: "降低密度 κ 40% 后重试",
  raise_budget: "候选预算/迭代上限加倍",
};

function handleKernelError(err) {
  setDot("err", "内核报告失败");
  $("renderBusy").classList.add("hidden");
  $("renderBtn").disabled = false;
  const e = (err.body && err.body.error) || { type: "request_error", message: err.message };
  showError(e);
  refreshDiagnostics(true);
}

function showError(e) {
  $("errorPanel").classList.remove("hidden");
  $("errorType").textContent = "内核失败：" + (e.type || "unknown");
  $("errorMessage").textContent = e.message || "";
  const ctx = Object.assign({}, e);
  delete ctx.message; delete ctx.allowed_actions;
  $("errorContext").textContent = JSON.stringify(ctx, null, 2);
  const box = $("feedbackButtons");
  box.innerHTML = "";
  const actions = e.allowed_actions ||
    ["repartition", "discard_resample", "clamp_demo"];
  for (const action of actions) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "btn " + (action === "clamp_demo" ? "btn-danger"
      : action === "repartition" ? "btn-ok" : "btn-warn");
    b.textContent = ACTION_LABELS[action] || action;
    b.addEventListener("click", () => submitFeedback(action, b));
    box.appendChild(b);
  }
  switchTab("image");
}

async function submitFeedback(action, btn) {
  if (btn) btn.disabled = true;
  setDot("busy", "反馈已进入内核，执行修复…");
  try {
    const r = await postJSON("/api/feedback", {
      action, config: state.config,
    });
    if (r.probe_only) {
      setDot("ok", "内核解析交接完成");
      alert("内核探针结果：\n" + JSON.stringify(r.kernel_probe, null, 2));
      refreshDiagnostics(true);
      return;
    }
    if (r.error) { showError(r.error); return; }
    $("errorPanel").classList.add("hidden");
    if (r.config) applyConfigToControls(r.config);
    state.cycleId++;
    applyRender(r);
    setDot("ok", "内核已按反馈修复并重新追踪");
    const myCycle = state.cycleId;
    continueAccumulation(myCycle);
    refreshDiagnostics(true);
  } catch (err) {
    handleKernelError(err);
  }
}

/* ------------------------------------------------------------------ trace */
$("renderImage").addEventListener("click", async (ev) => {
  if (!state.lastRender) return;
  const rect = ev.target.getBoundingClientRect();
  const fx = (ev.clientX - rect.left) / rect.width;
  const fy = (ev.clientY - rect.top) / rect.height;
  const marker = $("pickMarker");
  marker.classList.remove("hidden");
  marker.style.left = (fx * 100).toFixed(2) + "%";
  marker.style.top = (fy * 100).toFixed(2) + "%";
  switchTab("trace");
  $("traceEmpty").textContent = "内核追踪该光线中…";
  $("traceEmpty").classList.remove("hidden");
  $("traceContent").classList.add("hidden");
  try {
    const r = await postJSON("/api/trace",
      Object.assign({}, state.config, { fx, fy }));
    renderTrace(r);
  } catch (err) {
    const e = (err.body && err.body.error) || { message: err.message };
    $("traceEmpty").textContent = "该光线追踪失败：" + e.message;
    showError(e);
  }
});

function fmt(v, d = 4) {
  if (v === null || v === undefined) return "—";
  if (typeof v !== "number") return String(v);
  if (!Number.isFinite(v)) return v > 0 ? "∞" : "–";
  if (Math.abs(v) >= 1000 || (Math.abs(v) < 1e-4 && v !== 0)) {
    return v.toExponential(2);
  }
  return v.toFixed(d);
}

function renderTrace(r) {
  if (r.mode === "single") {
    state._lastEvents = r.result.events || [];
  } else {
    state._lastEvents =
      ((r.trace.bounces.find((b) => b.candidates && b.candidates.length)
        || { candidates: [] }).candidates);
  }
  $("traceEmpty").classList.add("hidden");
  $("traceContent").classList.remove("hidden");
  const p = r.pixel;
  let html = "";
  html += chip("像素", `(${p.ix}, ${p.iy})`);
  html += chip("世界坐标", `x=${p.x.toFixed(2)}, z=${p.z.toFixed(2)}`);
  html += chip("光线长度", fmt(r.length, 3) + " wu");
  html += chip("解析 τ", fmt(r.tau_analytic, 4));
  html += chip("解析 T", fmt(r.transmittance_analytic, 5), "good");

  if (r.mode === "single") {
    const res = r.result;
    html += chip("估计 T", fmt(res.estimate, 5));
    html += chip("候选", res.n_candidates);
    html += chip("真实/空", `${res.n_real} / ${res.n_null}`);
    html += chip("尝试次数", res.attempts);
    if (res.repairs && res.repairs.length) {
      html += chip("修复记录", res.repairs.length, "warn");
    }
    $("traceSummary").innerHTML = html;
    fillEventTable(res.events || []);
    drawWeightPlot(res.events || []);
    $("multiBounces").innerHTML = "";
  } else {
    const t = r.trace;
    html += chip("反弹上限", t.max_bounces);
    $("traceSummary").innerHTML = html;
    const firstEvents = (t.bounces.find((b) => b.candidates && b.candidates.length)
      || { candidates: [] }).candidates;
    fillEventTable(firstEvents);
    drawWeightPlot(firstEvents);
    $("multiBounces").innerHTML = (t.bounces || []).map(renderBounce).join("");
  }
}

function renderBounce(b) {
  if (b.type === "background") {
    return `<div class="bounce-card"><h4>第 ${b.bounce} 段：逸出到背景</h4>
      <div class="kv">候选 ${b.n_candidates} · 二元透射 ${fmt(b.estimate, 4)}
      · 贡献 RGB [${(b.contribution || []).map((x) => fmt(x, 3)).join(", ")}]</div></div>`;
  }
  return `<div class="bounce-card"><h4>第 ${b.bounce} 次：真实散射事件</h4>
    <div class="kv">位置 [${(b.position || []).map((x) => fmt(x, 2)).join(", ")}]
      · cosθ=${fmt(b.cos_theta, 3)} · HG phase=${fmt(b.phase, 4)}
      · 阴影透射=${fmt(b.shadow_transmittance, 4)}
      · 候选 ${b.n_candidates}（真实 ${b.n_real}）
      · 直射贡献 [${(b.direct_contribution || []).map((x) => fmt(x, 3)).join(", ")}]
      ${b.terminated ? "· 终止：" + b.terminated : ""}</div></div>`;
}

function fillEventTable(events) {
  const tb = $("eventTable").querySelector("tbody");
  const MAX_ROWS = 400;
  const shown = events.slice(0, MAX_ROWS);
  tb.innerHTML = shown.map((e, i) => {
    const kind = e.kind || "null";
    return `<tr class="kind-${kind}">
      <td>${i}</td>
      <td>${fmt(e.t, 3)}</td>
      <td><span class="kind-badge">${kind}</span></td>
      <td>${e.tag || e.segment || "—"}</td>
      <td>${fmt(e.density, 4)}</td>
      <td>${fmt(e.sigma_t, 4)}</td>
      <td>${fmt(e.majorant, 4)}</td>
      <td>${fmt(e.p_raw, 4)}</td>
      <td>${fmt(e.p_applied, 4)}</td>
      <td>${fmt(e.weight_before, 5)}</td>
      <td>${fmt(e.weight_factor, 4)}</td>
      <td>${fmt(e.weight_after, 5)}</td>
    </tr>`;
  }).join("");
  if (events.length > MAX_ROWS) {
    tb.innerHTML += `<tr><td colspan="12">… 仅显示前 ${MAX_ROWS} / ${events.length} 个事件</td></tr>`;
  }
}

function drawWeightPlot(events) {
  const canvas = $("weightPlot");
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 900;
  const H = 220;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  const ml = 46, mr = 12, mt = 12, mb = 24;
  const pw = W - ml - mr, ph = H - mt - mb;

  ctx.strokeStyle = "#2b3a52"; ctx.fillStyle = "#8da2c0";
  ctx.font = "11px sans-serif"; ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = mt + ph * g / 4;
    ctx.beginPath(); ctx.moveTo(ml, y); ctx.lineTo(W - mr, y); ctx.stroke();
    const val = 1 - g / 4;
    ctx.fillText(val.toFixed(2), 8, y + 4);
  }
  if (!events.length) {
    ctx.fillStyle = "#8da2c0";
    ctx.fillText("该光线没有候选事件（真空快路径或直接逸出）", ml, mt + 20);
    return;
  }
  const n = events.length;
  const bw = Math.max(1, pw / n);
  const colorFor = (k) => k === "real" ? "#ff7a59"
    : k === "analytic" ? "#c792ea" : "#5b6b86";
  events.forEach((e, i) => {
    const p = Math.min(1, Math.max(0, e.p_applied || 0));
    const h = ph * p;
    ctx.fillStyle = colorFor(e.kind);
    ctx.globalAlpha = 0.85;
    ctx.fillRect(ml + i * bw, mt + ph - h, Math.max(1, bw - 0.5), h);
  });
  ctx.globalAlpha = 1;
  // cumulative weight_after line
  ctx.strokeStyle = "#5aa9ff"; ctx.lineWidth = 2;
  ctx.beginPath();
  events.forEach((e, i) => {
    const w = Math.min(1, Math.max(0, e.weight_after !== null &&
      e.weight_after !== undefined ? e.weight_after : 1));
    const x = ml + (i + 0.5) * bw;
    const y = mt + ph * (1 - w);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#8da2c0";
  ctx.fillText("柱：应用概率 p（橙=真实散射，灰=空事件，紫=解析）；蓝线：累积权重 P",
    ml, H - 6);
}

/* ------------------------------------------------------------- diagnostics */
async function refreshDiagnostics(silent) {
  try {
    const d = await getJSON("/api/diagnostics");
    const view = $("diagnosticsView");
    const cards = [
      ["渲染次数", d.renders], ["光线检查", d.traces],
      ["运行秒数", d.uptime_seconds], ["已加载场景", (d.scenes_loaded || []).join(",") || "—"],
      ["统计验证", d.validation_pass === null || d.validation_pass === undefined
        ? "未运行" : (d.validation_pass ? "全部通过" : "存在失败")],
    ].map(([k, v]) => `<div class="stat">${k} <b>${v}</b></div>`).join("");
    view.innerHTML = `<div class="diag-cards">${cards}</div>` +
      `<div>累积轮数：${d.accumulated_passes}　反馈条数：${(d.feedback_log || []).length}</div>` +
      `<div style="margin-top:8px">${JSON.stringify(d, null, 2)
        .replace(/</g, "&lt;")}</div>`;
    if (!silent) switchTab("diagnostics");
  } catch (e) {
    if (!silent) $("diagnosticsView").textContent = "诊断获取失败：" + e.message;
  }
}

$("diagBtn").addEventListener("click", () => {
  switchTab("diagnostics");
  refreshDiagnostics(false);
});

/* -------------------------------------------------------------- validation */
async function loadValidation() {
  if (state.validationLoaded) return;
  const box = $("validationSummary");
  try {
    const r = await getJSON("/api/validation");
    if (!r.available) {
      box.textContent = "尚未生成报告。请在 /workspace 运行：" +
        "python3 -m stochastic_medium.cli validate";
      return;
    }
    state.validationLoaded = true;
    box.classList.remove("placeholder");
    box.innerHTML = `<div class="diag-cards">
      <div class="stat">统计检查 <b class="${r.all_statistical_checks_pass ? "good" : "bad"}">` +
      (r.all_statistical_checks_pass ? "全部通过" : "有失败") + `</b></div>
      <div class="stat">验证耗时 <b>${Math.round(r.elapsed_seconds)} s</b></div>
      <div class="stat">生成时间 <b>${r.generated_at}</b></div></div>
      <p class="hint">常密度均值/方差/事件数、分层密度、稀疏体积、主界修复与夹紧偏差的完整结果见
      outputs/validation_report.json。</p>`;
    if (r.figure) {
      const img = $("validationFigure");
      img.src = r.figure;
      img.classList.remove("hidden");
    }
  } catch (e) {
    box.textContent = "验证报告读取失败：" + e.message;
  }
}

/* ------------------------------------------------------------------ tabs */
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-pane").forEach((p) =>
    p.classList.toggle("active", p.id === "tab-" + name));
  if (name === "validation") loadValidation();
  if (name === "diagnostics") refreshDiagnostics(true);
  if (name === "trace") { /* plot needs visible width */
    const evs = state._lastEvents;
    if (evs) requestAnimationFrame(() => drawWeightPlot(evs));
  }
}
document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => switchTab(t.dataset.tab)));
document.querySelectorAll(".thumb[data-goto]").forEach((f) =>
  f.addEventListener("click", () => switchTab(f.dataset.goto)));

/* ------------------------------------------------------------- top buttons */
$("renderBtn").addEventListener("click", renderFirstPass);
$("accToggle").addEventListener("change", () => {
  if ($("accToggle").checked) {
    renderFirstPass();
  } else {
    state.cycleId++;
    clearTimeout(state.accTimer);
    $("renderBusy").classList.add("hidden");
  }
});
$("errorClose").addEventListener("click", () =>
  $("errorPanel").classList.add("hidden"));
$("resetBtn").addEventListener("click", async () => {
  state.cycleId++;
  clearTimeout(state.accTimer);
  await postJSON("/api/reset", {});
  $("errorPanel").classList.add("hidden");
  setDot("ok", "内核会话已重置");
  renderFirstPass();
});

/* redraw plot on resize if trace present */
window.addEventListener("resize", () => {
  if (state._lastEvents && $("tab-trace").classList.contains("active")) {
    drawWeightPlot(state._lastEvents);
  }
});

/* ------------------------------------------------------------------ boot */
(async function boot() {
  try {
    const meta = await getJSON("/api/schema");
    buildForm(meta.schema);
    setDot("ok", "内核 schema 已加载");
    renderFirstPass();
  } catch (e) {
    setDot("err", "无法连接内核服务");
    $("kernelStatus").textContent = "请通过 run_ui.sh 或 python3 -m stochastic_medium.cli serve 启动";
  }
})();
