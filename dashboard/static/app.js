/* TuftView front end. Plain JavaScript, no build step, no outside files. */
(() => {
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const STATE = { 0: "learning", 1: "steady", 2: "fluttering", 3: "stalled" };
const STATE_VAR = { 0: "--s0", 1: "--s1", 2: "--s2", 3: "--s3" };
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

const G = {            // app state
  jobs: [], cur: null, tab: "video", setup: null, results: null, overview: null, summary: null, units: "px",
  sel: null, tuftData: null, sort: { key: "tuft_id", dir: 1 }, filter: "all", q: "", dataView: "tufts",
  hPage: 0, roi: null, view: "live", layers: { arrows: true, ids: true, trail: true, grid: false },
  loadedId: null, loadedVersion: 0,
};

/* ---------------------------------------------------------------- helpers */
async function api(path, opt = {}) {
  const o = { ...opt };
  if (o.json !== undefined) { o.method = o.method || "POST"; o.headers = { "Content-Type": "application/json" }; o.body = JSON.stringify(o.json); delete o.json; }
  const r = await fetch("/api" + path, o);
  let data = null;
  try { data = await r.json(); } catch (_) { /* not json */ }
  if (!r.ok) throw new Error((data && data.error) || `Something went wrong (${r.status}).`);
  return data;
}
let toastT;
function toast(msg, err = false) {
  const t = $("#toast"); t.textContent = msg; t.classList.toggle("err", err); t.hidden = false;
  clearTimeout(toastT); toastT = setTimeout(() => (t.hidden = true), err ? 6000 : 3200);
}
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: 0 }));
function clock(s) { s = Math.max(0, s || 0); const m = Math.floor(s / 60); return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`; }
function eta(s) { if (s == null || !isFinite(s)) return "—"; if (s < 60) return `${Math.ceil(s)} s`; return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`; }
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const mul3 = H => H;
function inv3(m) {
  const [a, b, c, d, e, f, g, h, i] = m.flat();
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-14) return null;
  return [[A / det, -(b * i - c * h) / det, (b * f - c * e) / det], [B / det, (a * i - c * g) / det, -(a * f - c * d) / det], [C / det, -(a * h - b * g) / det, (a * e - b * d) / det]];
}
function applyH(m, x, y) { const w = m[2][0] * x + m[2][1] * y + m[2][2]; return [(m[0][0] * x + m[0][1] * y + m[0][2]) / w, (m[1][0] * x + m[1][1] * y + m[1][2]) / w]; }
function sizeCanvas(cv) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  return { w, h, dpr };
}
/** Where the picture sits inside an element that uses object-fit: contain. */
function fitRect(boxW, boxH, W, H) { const s = Math.min(boxW / W, boxH / H); return { s, x0: (boxW - W * s) / 2, y0: (boxH - H * s) / 2 }; }
const jobById = id => G.jobs.find(j => j.id === id);

/* ---------------------------------------------------------------- the soft light behind the home page */
const Field = (() => {
  const cv = $("#field"), ctx = cv.getContext("2d");
  let W = 0, H = 0, raf = 0, running = false, mouse = { x: -999, y: -999, sx: -999, sy: -999 }, energy = 0.35, target = 0.35, t0 = performance.now();
  // slow drifting pools of colour; each has its own orbit
  const orbs = [
    { c: "10,140,166", r: 0.46, ax: 0.30, ay: 0.34, fx: 0.07, fy: 0.05, p: 0.0, a: 0.34 },
    { c: "86,104,255", r: 0.40, ax: 0.72, ay: 0.28, fx: 0.05, fy: 0.08, p: 1.7, a: 0.22 },
    { c: "60,210,170", r: 0.36, ax: 0.62, ay: 0.78, fx: 0.06, fy: 0.045, p: 3.1, a: 0.24 },
    { c: "168,120,255", r: 0.32, ax: 0.20, ay: 0.80, fx: 0.04, fy: 0.07, p: 4.6, a: 0.16 },
  ];
  function size() { const dpr = window.devicePixelRatio || 1; W = cv.clientWidth; H = cv.clientHeight; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); }
  function frame(now) {
    const t = (now - t0) / 1000, dark = css("color-scheme") === "dark" || matchMedia("(prefers-color-scheme: dark)").matches && document.documentElement.dataset.theme !== "light";
    energy += (target - energy) * 0.05;
    mouse.sx += (mouse.x - mouse.sx) * 0.12; mouse.sy += (mouse.y - mouse.sy) * 0.12;
    ctx.clearRect(0, 0, W, H);
    const m = Math.max(W, H);
    ctx.globalCompositeOperation = "source-over";
    for (const o of orbs) {
      const sp = t * (0.6 + energy);
      const x = W * (o.ax + 0.16 * Math.sin(sp * o.fx * 6.28 + o.p)), y = H * (o.ay + 0.16 * Math.cos(sp * o.fy * 6.28 + o.p * 1.3));
      const r = m * o.r * (0.92 + 0.12 * Math.sin(sp * 0.3 + o.p)), al = o.a * (dark ? 0.8 : 1) * (0.8 + 0.5 * energy);
      const g = ctx.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0, `rgba(${o.c},${al})`); g.addColorStop(1, `rgba(${o.c},0)`);
      ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    }
    // fine dot grid that wakes up around the pointer
    const gap = 30, ink = dark ? "231,237,242" : "15,27,38";
    for (let y = gap / 2; y < H; y += gap) for (let x = gap / 2; x < W; x += gap) {
      const d = Math.hypot(x - mouse.sx, y - mouse.sy), k = Math.max(0, 1 - d / 190), w = 0.5 + 0.5 * Math.sin(t * 0.8 + x * 0.012 + y * 0.009);
      const r = 1 + 1.7 * k * k + 0.35 * w * energy, a = 0.09 + 0.5 * k * k;
      ctx.fillStyle = k > 0.02 ? `rgba(10,140,166,${a + 0.1})` : `rgba(${ink},${0.07 + 0.05 * w})`;
      ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill();
    }
    // a soft light that follows the pointer
    if (mouse.sx > -500) { const g = ctx.createRadialGradient(mouse.sx, mouse.sy, 0, mouse.sx, mouse.sy, 240); g.addColorStop(0, "rgba(10,140,166,.13)"); g.addColorStop(1, "rgba(10,140,166,0)"); ctx.fillStyle = g; ctx.fillRect(0, 0, W, H); }
    if (running) raf = requestAnimationFrame(frame);
  }
  function start() { if (running) return; running = true; size(); if (reduceMotion) { running = false; frame(performance.now()); } else raf = requestAnimationFrame(frame); }
  function stop() { running = false; cancelAnimationFrame(raf); }
  addEventListener("resize", () => { if (!$("#home").hidden) { size(); if (!running) frame(performance.now()); } });
  $("#home").addEventListener("pointermove", e => { const r = cv.getBoundingClientRect(); mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top; });
  $("#home").addEventListener("pointerleave", () => { mouse.x = mouse.y = -999; });
  return { start, stop, gust(on) { target = on ? 1.5 : 0.35; } };
})();

/* ---------------------------------------------------------------- uploads, drag and drop, the + menu */
const MEDIA_OK = /\.(mp4|m4v|mov|avi|mkv|webm|jpe?g|png|bmp|webp|tiff?|avif|heic)$/i;
function showHome() { setHash(""); }
function uploadFiles(files) {
  files = [...files];
  if (!files.length) return;
  if (G.cur) showHome();
  const first = [];
  for (const f of files) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${esc(f.name)}</span><span class="muted">Uploading…</span><div class="bar"><i></i></div>`;
    $("#uploads").prepend(li);
    if (!MEDIA_OK.test(f.name)) { li.querySelector(".muted").textContent = "Not a video or picture"; li.classList.add("err"); li.querySelector(".muted").classList.add("err"); continue; }
    const fd = new FormData(); fd.append("files", f);
    const x = new XMLHttpRequest();
    x.open("POST", "/api/upload");
    x.upload.onprogress = e => { if (e.lengthComputable) { const p = Math.round((e.loaded / e.total) * 100); li.querySelector("i").style.width = p + "%"; li.querySelector(".muted").textContent = p < 100 ? `Uploading ${p}%` : "Reading the file…"; } };
    x.onload = () => {
      let d = {}; try { d = JSON.parse(x.responseText); } catch (_) {}
      if (x.status === 200 && d.jobs && d.jobs.length) {
        li.remove(); d.jobs.forEach(j => { G.jobs.unshift(j); first.push(j); });
        renderRail(); if (first.length === 1 || !G.cur) openJob(d.jobs[0].id);
      } else {
        const msg = (d.errors && d.errors[0] && d.errors[0].error) || d.error || "Upload failed.";
        li.querySelector(".muted").textContent = msg; li.querySelector(".muted").classList.add("err");
      }
    };
    x.onerror = () => { li.querySelector(".muted").textContent = "Upload failed. Is TuftView still running?"; li.querySelector(".muted").classList.add("err"); };
    x.send(fd);
  }
}
let dragDepth = 0;
addEventListener("dragenter", e => { if (e.dataTransfer && [...e.dataTransfer.types].includes("Files")) { dragDepth++; $("#veil").hidden = false; Field.gust(true); } });
addEventListener("dragleave", () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) { $("#veil").hidden = true; Field.gust(false); } });
addEventListener("dragover", e => e.preventDefault());
addEventListener("drop", e => { e.preventDefault(); dragDepth = 0; $("#veil").hidden = true; Field.gust(false); $("#composer").classList.remove("over"); if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files); });
$("#composer").addEventListener("dragover", () => $("#composer").classList.add("over"));
$("#composer").addEventListener("dragleave", () => $("#composer").classList.remove("over"));
$("#fileIn").addEventListener("change", e => { uploadFiles(e.target.files); e.target.value = ""; });
function pickFiles() { $("#fileIn").click(); }
function closeMenus() { $$(".menu").forEach(m => (m.hidden = true)); $("#plusBtn").setAttribute("aria-expanded", "false"); $("#dlBtn").setAttribute("aria-expanded", "false"); }
$("#plusBtn").addEventListener("click", e => { e.stopPropagation(); const m = $("#plusMenu"), open = m.hidden; closeMenus(); m.hidden = !open; $("#plusBtn").setAttribute("aria-expanded", String(open)); if (open) $("#menuUpload").focus(); });
$("#menuUpload").addEventListener("click", () => { closeMenus(); pickFiles(); });
$("#menuDemo").addEventListener("click", async () => {
  closeMenus();
  try { toast("Making a sample clip…"); const j = await api("/demo", { method: "POST" }); G.jobs.unshift(j); renderRail(); openJob(j.id); }
  catch (e) { toast(e.message, true); }
});
$("#newBtn").addEventListener("click", () => { showHome(); $("#shell").classList.remove("rail-open"); });
$("#composer").addEventListener("click", e => { if (!e.target.closest(".plus-wrap")) pickFiles(); });
addEventListener("click", e => { if (!e.target.closest(".menu") && !e.target.closest("#plusBtn") && !e.target.closest("#dlBtn")) closeMenus(); });
addEventListener("keydown", e => { if (e.key === "Escape") closeMenus(); });
$("#railToggle").addEventListener("click", () => $("#shell").classList.toggle("rail-open"));

/* ---------------------------------------------------------------- the list of analyses */
function statusText(j) {
  if (j.status === "running") return `Analyzing ${Math.round(j.progress * 100)}%`;
  if (j.status === "queued") return "Waiting";
  if (j.status === "review") return "Check the tufts";
  if (j.status === "done") return `${j.tufts} tufts`;
  if (j.status === "cancelled") return "Cancelled";
  return "Needs attention";
}
function renderRail() {
  const ul = $("#runs"); ul.innerHTML = "";
  $("#railEmpty").hidden = G.jobs.length > 0;
  for (const j of G.jobs) {
    const li = document.createElement("li");
    const b = document.createElement("button"); b.type = "button"; b.className = "run-item";
    b.setAttribute("aria-current", String(G.cur === j.id));
    const run = j.status === "running" || j.status === "queued";
    b.innerHTML = `<img alt="" src="/api/jobs/${j.id}/first_frame.jpg" loading="lazy"><div><div class="nm">${esc(j.name)}</div><div class="sub"><i class="sdot ${run ? "running" : j.status}"></i>${statusText(j)}</div>${run ? `<div class="bar"><i style="width:${Math.round(j.progress * 100)}%"></i></div>` : ""}</div>`;
    b.addEventListener("click", () => { openJob(j.id); $("#shell").classList.remove("rail-open"); });
    const x = document.createElement("button"); x.type = "button"; x.className = "x"; x.setAttribute("aria-label", `Remove ${j.name}`);
    x.innerHTML = '<svg viewBox="0 0 20 20" width="12" height="12"><path d="M5 5l10 10M15 5L5 15" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
    x.addEventListener("click", async ev => {
      ev.stopPropagation();
      if (!confirm(`Remove “${j.name}” and its results?`)) return;
      try { await api(`/jobs/${j.id}`, { method: "DELETE" }); G.jobs = G.jobs.filter(k => k.id !== j.id); if (G.cur === j.id) showHome(); renderRail(); } catch (e) { toast(e.message, true); }
    });
    b.appendChild(x); li.appendChild(b); ul.appendChild(li);
  }
}
async function pollJobs() {
  let busy = false;
  try {
    const list = await api("/jobs");
    const prev = Object.fromEntries(G.jobs.map(j => [j.id, j]));
    G.jobs = list; renderRail();
    for (const j of list) {
      if (j.status === "running" || j.status === "queued") busy = true;
      if (j.id === G.cur) {
        const p = prev[j.id];
        if (!p || p.status !== j.status || p.version !== j.version) openJob(j.id, true);
        else if (j.status === "running" || j.status === "queued") updateWorking(j);
        else if (p.render === "running" && j.render === "done") { if (G.view === "export") setViewSource(); }
      }
    }
  } catch (_) { /* server may be restarting */ }
  setTimeout(pollJobs, busy ? 700 : 3500);
}

/* ---------------------------------------------------------------- routing */
function setHash(h) { if (location.hash.slice(1) !== h) location.hash = h; else route(); }
addEventListener("hashchange", route);
function route() {
  const m = location.hash.match(/^#\/run\/([a-z0-9]+)(?:\/(video|data|setup))?/);
  if (m) { if (m[2]) G.tab = m[2]; if (G.cur !== m[1] || G.loadedId !== m[1]) openJob(m[1], false, true); else showTab(G.tab, true); }
  else { G.cur = null; stopViewer(); $("#run").hidden = true; $("#home").hidden = false; Field.start(); renderRail(); }
}

/* ---------------------------------------------------------------- opening a run */
async function openJob(id, refresh = false, fromRoute = false) {
  const j = jobById(id) || (await api(`/jobs/${id}`).catch(() => null));
  if (!j) { toast("That analysis no longer exists.", true); setHash(""); return; }
  if (!jobById(id)) G.jobs.unshift(j);
  if (!fromRoute && !location.hash.startsWith(`#/run/${id}`)) { G.cur = id; location.hash = `#/run/${id}/${G.tab}`; return; }
  const switching = G.cur !== id || G.loadedId !== id;
  G.cur = id; $("#home").hidden = true; $("#run").hidden = false; Field.stop(); renderRail();
  if (switching) { G.sel = null; G.tuftData = null; G.roi = null; G.hPage = 0; G.view = "live"; $$("#viewSeg button").forEach(b => b.classList.toggle("on", b.dataset.view === "live")); }
  $("#runName").textContent = j.name;
  renderChips(j);
  const done = j.status === "done";
  $("#working").hidden = !(j.status === "running" || j.status === "queued");
  $("#failed").hidden = !(j.status === "error" || j.status === "cancelled");
  $("#reviewPanel").hidden = j.status !== "review";
  $("#tabs").classList.toggle("off", !done);
  $("#dlWrap").hidden = !done;
  $$(".tabpane").forEach(p => (p.hidden = true));
  if (j.status === "running" || j.status === "queued") { stopViewer(); updateWorking(j); return; }
  if (j.status === "review") { stopViewer(); showReviewPanel(j); return; }
  if (!done) {
    stopViewer(); $("#failMsg").textContent = j.status === "cancelled" ? "You cancelled this analysis." : (j.error || "Something went wrong while analyzing this file.");
    return;
  }
  try {
    if (switching || refresh || G.loadedVersion !== j.version) await loadResults(j);
  } catch (e) { toast(e.message, true); return; }
  G.loadedId = id; G.loadedVersion = j.version;
  buildDownloads(j);
  showTab(G.tab, true);
}
function un0(s) { return !!s.unet; }
function warnHtml(lines, fix, classic) {
  const ico = `<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 2.5 20h19L12 3z"/><path d="M12 10v4.5M12 17.6v.1"/></svg>`;
  const title = classic ? "Trained model not in use" : "Heads up";
  return `<span class="wi">${ico}</span><span class="wb"><b>${title}</b>${lines.map(l => `<span>${esc(l)}</span>`).join("")}` +
    (fix ? `<em>${esc(fix)}</em>` : "") + `</span>` + (classic ? `<span class="wp">Classic detector</span>` : "");
}
function renderChips(j) {
  const c = [];
  if (j.kind === "video") c.push(`<span class="chip"><b>${clock(j.info.duration)}</b> video</span>`);
  c.push(`<span class="chip"><b>${j.info.width}×${j.info.height}</b></span>`);
  if (j.status === "done") { c.push(`<span class="chip"><b>${j.tufts}</b> tufts</span>`); if (j.stats && j.stats.engine) { const un = /U-Net/.test(j.stats.engine), tip = ((j.stats.warnings || []).join(" ")); c.push(`<span class="chip det ${un ? "on" : ""}" title="${esc(tip)}">${un ? "Trained U-Net" : "Classical detector"}</span>`); } }
  $("#runChips").innerHTML = c.join("");
}
async function loadResults(j) {
  const [res, ov, sm, st] = await Promise.all([api(`/jobs/${j.id}/results.json`), api(`/jobs/${j.id}/overview.json`), api(`/jobs/${j.id}/summary.json`), api(`/jobs/${j.id}/setup`)]);
  G.results = res; G.overview = ov; G.summary = sm.rows; G.units = sm.units; G.setup = st.setup; G.residuals = st.residuals;
  G.liveFallback = false;
}
function buildDownloads(j) {
  const a = (h, t, s) => `<button type="button" role="menuitem" data-h="${h}"><svg viewBox="0 0 20 20" width="18" height="18"><path d="M5 2.5h6l4 4v11H5z M11 2.5v4h4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg><span>${t}<small>${s}</small></span></button>`;
  const base = `/api/jobs/${j.id}/download/`;
  const media = j.kind === "video" ? a(base + "annotated.mp4", "Annotated video", "MP4 with flow arrows drawn on it") : a(base + "annotated.jpg", "Annotated picture", "JPG with flow arrows drawn on it");
  $("#dlMenu").innerHTML = a(base + "all.zip", "Everything", "One ZIP with all of the files below") + "<hr>" + media +
    a(base + "vector_time_history.csv", "Vector time history", "CSV: every tuft at every moment") + a(base + "tuft_summary.csv", "Tuft summary", "CSV: one line per tuft") +
    a(base + "results.xlsx", "Excel workbook", "Summary and time history in one file") + a(base + "tuft_locations.csv", "Tuft positions", "CSV you can edit and import again");
  $$("#dlMenu button").forEach(b => b.addEventListener("click", () => { closeMenus(); location.href = b.dataset.h; }));
}
$("#dlBtn").addEventListener("click", e => { e.stopPropagation(); const m = $("#dlMenu"), open = m.hidden; closeMenus(); m.hidden = !open; $("#dlBtn").setAttribute("aria-expanded", String(open)); });

/* ---------------------------------------------------------------- working view */
function updateWorking(j) {
  $("#workImg").src = `/api/jobs/${j.id}/preview.jpg?t=${Date.now() % 1e7}`;
  const p = j.status === "queued" ? 0 : j.progress;
  $("#ringFg").style.strokeDashoffset = String(326.7 * (1 - p)); $("#ringTxt").textContent = Math.round(p * 100) + "%";
  $("#workTitle").textContent = j.status === "queued" ? "Waiting for its turn" : /^Frame \d/.test(j.message || "") ? "Processing every frame" : "Finding tufts";
  $("#workMsg").textContent = j.message || ""; $("#workTufts").textContent = j.tufts || 0; $("#workEta").textContent = eta(j.eta);
}
$("#cancelBtn").addEventListener("click", async () => { try { await api(`/jobs/${G.cur}/cancel`, { method: "POST" }); } catch (e) { toast(e.message, true); } });
$("#retryBtn").addEventListener("click", async () => { try { await api(`/jobs/${G.cur}/reanalyze`, { json: {} }); } catch (e) { toast(e.message, true); } });
$("#failDel").addEventListener("click", async () => { try { await api(`/jobs/${G.cur}`, { method: "DELETE" }); G.jobs = G.jobs.filter(k => k.id !== G.cur); showHome(); renderRail(); } catch (e) { toast(e.message, true); } });

/* ---------------------------------------------------------------- tabs */
$$("#tabs button").forEach(b => b.addEventListener("click", () => { G.tab = b.dataset.tab; location.hash = `#/run/${G.cur}/${G.tab}`; }));
function showTab(tab, force) {
  const j = jobById(G.cur); if (!j || j.status !== "done") return;
  G.tab = tab;
  $$("#tabs button").forEach(b => b.setAttribute("aria-selected", String(b.dataset.tab === tab)));
  $$(".tabpane").forEach(p => (p.hidden = p.id !== `pane-${tab}`));
  if (tab === "video") startViewer(j); else pauseViewer();
  if (tab === "data") renderData();
  if (tab === "setup") renderSetup(j);
}

/* ---------------------------------------------------------------- the video viewer */
const vid = $("#vid"), still = $("#stillImg"), ov = $("#ov"), octx = ov.getContext("2d");
const V = { times: [], frames: [], W: 1, H: 1, fps: 30, step: 1, idx: 0, hover: null, mouse: null, raf: 0, active: false, isVideo: true, dirty: true, lastT: -1, chart: null };
const mediaEl = () => (V.isVideo ? vid : still);
function frameAt(t) { const T = V.times; let lo = 0, hi = T.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (T[m] <= t + 1e-4) lo = m; else hi = m - 1; } return lo; }
function curTime() { return V.isVideo ? vid.currentTime : 0; }
function startViewer(j) {
  const r = G.results;
  V.times = r.times; V.frames = r.frames; V.W = r.width; V.H = r.height; V.fps = r.fps; V.step = r.step; V.isVideo = j.kind === "video";
  vid.hidden = !V.isVideo; still.hidden = V.isVideo; $("#transport").hidden = !V.isVideo;
  $("#viewSeg").hidden = false; $("#gridTog").hidden = !(G.setup && G.setup.coords && G.setup.coords.defined);
  $("#statCards").innerHTML = ""; renderStats();
  $("#layers [data-layer=trail]").closest("label").hidden = !V.isVideo; $(".legend").hidden = !V.isVideo;
  $("#viewSeg [data-view=export]").textContent = V.isVideo ? "Exported video" : "Exported picture";
  const wantSrc = viewUrl(j);
  if (V.isVideo) { if (vid.dataset.src !== wantSrc) { vid.dataset.src = wantSrc; vid.src = wantSrc; } } else if (still.dataset.src !== wantSrc) { still.dataset.src = wantSrc; still.src = wantSrc; }
  drawTimelineBackdrop(); renderTuftCard();
  V.active = true; V.dirty = true; cancelAnimationFrame(V.raf); V.raf = requestAnimationFrame(loop);
}
function viewUrl(j) {
  const live = G.view === "live";
  if (V.isVideo) return live ? `/api/jobs/${j.id}/original?v=${j.id}` : `/api/jobs/${j.id}/media?v=${j.version}.${j.render_ver || 0}`;
  return live ? `/api/jobs/${j.id}/first_frame.jpg` : `/api/jobs/${j.id}/media`;
}
function setViewSource() {
  const j = jobById(G.cur); if (!j) return; const t = vid.currentTime, was = !vid.paused;
  const url = viewUrl(j); vid.dataset.src = url; vid.src = url;
  if (V.isVideo) vid.addEventListener("loadedmetadata", () => { vid.currentTime = t; if (was) vid.play(); }, { once: true });
  else { still.dataset.src = url; still.src = url; }
  V.dirty = true;
}
function pauseViewer() { if (V.isVideo) vid.pause(); V.active = false; cancelAnimationFrame(V.raf); }
function stopViewer() { pauseViewer(); try { vid.removeAttribute("src"); vid.dataset.src = ""; vid.load(); } catch (_) {} }
vid.addEventListener("error", () => {
  if (G.view === "export" && G.liveFallback) { toast("This browser can’t play the video. Download the annotated video from the Download menu instead.", true); return; }
  if (G.view === "live") { G.liveFallback = true; G.view = "export"; $$("#viewSeg button").forEach(b => b.classList.toggle("on", b.dataset.view === "export")); toast("Your browser can’t play the original file, so the exported video is shown instead."); setViewSource(); }
});
vid.addEventListener("loadedmetadata", () => { V.dirty = true; });
vid.addEventListener("play", () => $("#transport").classList.add("playing"));
vid.addEventListener("pause", () => $("#transport").classList.remove("playing"));
vid.addEventListener("ended", () => $("#transport").classList.remove("playing"));
$("#playBtn").addEventListener("click", () => { if (vid.paused) { if (vid.ended) vid.currentTime = 0; vid.play(); } else vid.pause(); });
$("#speed").addEventListener("change", e => (vid.playbackRate = Number(e.target.value)));
$$("#viewSeg button").forEach(b => b.addEventListener("click", () => { G.view = b.dataset.view; $$("#viewSeg button").forEach(x => x.classList.toggle("on", x === b)); setViewSource(); }));
$$("#layers input").forEach(i => i.addEventListener("change", () => { G.layers[i.dataset.layer] = i.checked; V.dirty = true; }));
$("#stage").addEventListener("pointermove", e => { const r = ov.getBoundingClientRect(); V.mouse = { x: e.clientX - r.left, y: e.clientY - r.top }; V.dirty = true; });
$("#stage").addEventListener("pointerleave", () => { V.mouse = null; V.hover = null; $("#stageTip").hidden = true; V.dirty = true; ov.style.cursor = "default"; });
$("#stage").addEventListener("click", e => {
  const r = ov.getBoundingClientRect(), hit = pickAt(e.clientX - r.left, e.clientY - r.top);   // works for taps, which have no hover
  if (hit) selectTuft(hit[0]);
});
addEventListener("keydown", e => {
  if (!V.active || /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
  if (e.key === " " && V.isVideo) { e.preventDefault(); $("#playBtn").click(); }
  if ((e.key === "ArrowRight" || e.key === "ArrowLeft") && V.isVideo) { e.preventDefault(); vid.pause(); const i = Math.max(0, Math.min(V.times.length - 1, V.idx + (e.key === "ArrowRight" ? 1 : -1))); vid.currentTime = V.times[i] + 0.0005; }
});
// timeline: flow health over time, click or drag to seek
const tl = $("#timeline");
function seekFromEvent(e) { const r = tl.getBoundingClientRect(); const f = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)); if (vid.duration) vid.currentTime = f * vid.duration; }
let tlDrag = false;
tl.addEventListener("pointerdown", e => { tlDrag = true; tl.setPointerCapture(e.pointerId); seekFromEvent(e); });
tl.addEventListener("pointermove", e => { if (tlDrag) seekFromEvent(e); });
tl.addEventListener("pointerup", () => (tlDrag = false));
tl.addEventListener("keydown", e => { if (e.key === "ArrowRight") vid.currentTime += 1; if (e.key === "ArrowLeft") vid.currentTime -= 1; });
function drawTimelineBackdrop() {
  const cv = $("#tlCanvas"), { w, h, dpr } = sizeCanvas(cv), c = cv.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h);
  const S = G.overview.series, dur = Math.max(G.overview.duration, 0.001);
  if (!S.length || w < 4) return;
  const cols = [css("--s1"), css("--s2"), css("--s3")];
  for (let x = 0; x < w; x += 2) {
    const t = (x / w) * dur, i = frameAt2(S, t), r = S[i]; const known = r[3] + r[4] + r[5];
    if (!known) { c.fillStyle = css("--line"); c.globalAlpha = 0.6; c.fillRect(x, h - 4, 2, 4); c.globalAlpha = 1; continue; }
    let y = h; [r[3], r[4], r[5]].forEach((n, k) => { const hh = (n / known) * (h - 6); y -= hh; c.fillStyle = cols[k]; c.globalAlpha = 0.9; c.fillRect(x, y, 2, hh + 0.5); });
    c.globalAlpha = 1;
  }
}
function frameAt2(S, t) { let lo = 0, hi = S.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (S[m][0] <= t + 1e-4) lo = m; else hi = m - 1; } return lo; }
addEventListener("resize", () => { if (V.active) { drawTimelineBackdrop(); V.dirty = true; drawChart(); } });

function loop() {
  if (!V.active) return;
  const t = curTime(), idx = frameAt(t);
  if (idx !== V.idx) { V.idx = idx; V.dirty = true; }
  if (t !== V.lastT) {
    V.lastT = t; V.dirty = true;
    const dur = vid.duration || G.overview.duration || 0;
    $("#playhead").style.left = dur ? `calc(${(t / dur) * 100}% - 1px)` : "0";
    $("#clock").textContent = `${clock(t)} / ${clock(dur)}`; drawChart(true);
  }
  if (V.dirty) { V.dirty = false; drawOverlay(); }
  V.raf = requestAnimationFrame(loop);
}
function geom() {
  const el = mediaEl(), { w, h, dpr } = sizeCanvas(ov);
  const box = el.getBoundingClientRect();
  const f = fitRect(box.width, box.height, V.W, V.H);
  return { w, h, dpr, ...f, k: Math.max(V.W, V.H) / 1000 };
}
function arrowEnd(r, k) { const ln = Math.max(22 * k, 0.8 * r[8]); return [r[1] + r[5] * ln, r[2] + r[6] * ln]; }
function distSeg(px, py, ax, ay, bx, by) { const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy || 1; let t = ((px - ax) * dx + (py - ay) * dy) / L; t = Math.max(0, Math.min(1, t)); return Math.hypot(px - (ax + t * dx), py - (ay + t * dy)); }
function drawOverlay() {
  const g = geom(), c = octx; c.setTransform(g.dpr, 0, 0, g.dpr, 0, 0); c.clearRect(0, 0, g.w, g.h);
  c.save(); c.translate(g.x0, g.y0); c.scale(g.s, g.s);
  const k = g.k, recs = V.frames[V.idx] || [], live = G.view === "live", sel = G.sel;
  if (G.layers.grid && G.setup && G.setup.coords && G.setup.coords.defined) drawGrid(c, k);
  // motion trail of the selected tuft
  if (sel !== null && G.layers.trail) {
    const n = Math.max(3, Math.round((V.fps / V.step) * 1.2)); c.lineCap = "round"; c.lineJoin = "round";
    let prev = null;
    for (let i = Math.max(0, V.idx - n); i <= V.idx; i++) {
      const r = (V.frames[i] || []).find(q => q[0] === sel); if (!r) { prev = null; continue; }
      if (prev) { const a = (i - (V.idx - n)) / n; c.strokeStyle = `rgba(255,255,255,${0.15 + 0.7 * a})`; c.lineWidth = (2 + 3 * a) * k; c.beginPath(); c.moveTo(prev[0], prev[1]); c.lineTo(r[3], r[4]); c.stroke(); }
      prev = [r[3], r[4]];
    }
  }
  if (live && G.layers.arrows) {
    for (const r of recs) {
      const dim = sel !== null && r[0] !== sel, col = css(STATE_VAR[r[10]]), e = arrowEnd(r, k);
      c.globalAlpha = dim ? 0.38 : 1; drawArrow(c, r[1], r[2], e[0], e[1], col, k);
    }
    c.globalAlpha = 1;
  }
  if (live && G.layers.ids) {
    c.font = `600 ${Math.max(11, 12.5 * k)}px ${getComputedStyle(document.body).fontFamily}`; c.textBaseline = "alphabetic"; c.lineJoin = "round";
    for (const r of recs) { const dim = sel !== null && r[0] !== sel; c.globalAlpha = dim ? 0.45 : 1; c.lineWidth = 3.2 * k; c.strokeStyle = "rgba(10,16,22,.85)"; c.fillStyle = "#fff"; const tx = r[1] + 5 * k, ty = r[2] - 6 * k; c.strokeText(r[0], tx, ty); c.fillText(r[0], tx, ty); }
    c.globalAlpha = 1;
  }
  // hover + selection rings
  const hover = V.mouse ? pickAt(V.mouse.x, V.mouse.y, g) : null;
  V.hover = hover ? { id: hover[0] } : null;
  const ring = (r, col, w) => { const e = arrowEnd(r, k), cx = (r[1] + e[0]) / 2, cy = (r[2] + e[1]) / 2, rad = Math.max(16 * k, Math.hypot(e[0] - r[1], e[1] - r[2]) / 2 + 9 * k); c.strokeStyle = col; c.lineWidth = w * k; c.beginPath(); c.arc(cx, cy, rad, 0, 6.2832); c.stroke(); };
  if (hover && hover[0] !== sel) ring(hover, "rgba(255,255,255,.8)", 1.8);
  if (sel !== null) { const r = recs.find(q => q[0] === sel); if (r) { const pulse = reduceMotion ? 0 : 0.5 + 0.5 * Math.sin(performance.now() / 260); ring(r, "#fff", 2.6); c.globalAlpha = 0.25 + 0.3 * pulse; ring(r, css("--accent"), 7); c.globalAlpha = 1; V.dirty = !reduceMotion; } }
  c.restore();
  const tip = $("#stageTip");
  if (hover && V.mouse) { tip.hidden = false; tip.textContent = `Tuft ${hover[0]} · ${STATE[hover[10]]}${hover[0] === sel ? "" : " · click for details"}`; const flip = V.mouse.x > g.w - 230; tip.style.transform = flip ? "translate(calc(-100% - 12px), -110%)" : "translate(12px, -110%)"; tip.style.left = V.mouse.x + "px"; tip.style.top = Math.max(V.mouse.y, 34) + "px"; ov.style.cursor = "pointer"; }
  else { tip.hidden = true; ov.style.cursor = "default"; }
}
/** The tuft under a point (canvas pixels), or null. */
function pickAt(cx, cy, g) {
  g = g || geom(); const k = g.k, recs = V.frames[V.idx] || [], mx = (cx - g.x0) / g.s, my = (cy - g.y0) / g.s;
  let best = 18 / g.s, hit = null;
  for (const r of recs) { const e = arrowEnd(r, k), d = distSeg(mx, my, r[1], r[2], e[0], e[1]); if (d < best) { best = d; hit = r; } }
  return hit;
}
function drawArrow(c, x1, y1, x2, y2, col, k) {
  const a = Math.atan2(y2 - y1, x2 - x1), hl = Math.max(7 * k, Math.hypot(x2 - x1, y2 - y1) * 0.3), hw = 0.42;
  const path = () => { c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2 - Math.cos(a) * hl * 0.6, y2 - Math.sin(a) * hl * 0.6); };
  c.lineCap = "round"; c.lineJoin = "round";
  path(); c.strokeStyle = "rgba(14,20,26,.9)"; c.lineWidth = 4.4 * k; c.stroke();
  c.beginPath(); c.moveTo(x2, y2); c.lineTo(x2 - Math.cos(a - hw) * hl, y2 - Math.sin(a - hw) * hl); c.lineTo(x2 - Math.cos(a + hw) * hl, y2 - Math.sin(a + hw) * hl); c.closePath(); c.fillStyle = "rgba(14,20,26,.9)"; c.strokeStyle = "rgba(14,20,26,.9)"; c.lineWidth = 2.6 * k; c.stroke(); c.fill();
  path(); c.strokeStyle = col; c.lineWidth = 2.2 * k; c.stroke();
  c.beginPath(); c.moveTo(x2, y2); c.lineTo(x2 - Math.cos(a - hw) * hl, y2 - Math.sin(a - hw) * hl); c.lineTo(x2 - Math.cos(a + hw) * hl, y2 - Math.sin(a + hw) * hl); c.closePath(); c.fillStyle = col; c.fill();
  c.beginPath(); c.arc(x1, y1, 3.2 * k, 0, 6.2832); c.fillStyle = "#fff"; c.fill();
}
function drawGrid(c, k) {
  const Hm = G.setup.coords.H, inv = inv3(Hm); if (!inv) return;
  const corners = [[0, 0], [V.W, 0], [V.W, V.H], [0, V.H]].map(p => applyH(Hm, p[0], p[1]));
  const xs = corners.map(p => p[0]), ys = corners.map(p => p[1]);
  const raw = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) / 8, pw = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / pw, step = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * pw;
  c.lineWidth = 1 * k; c.strokeStyle = "rgba(255,255,255,.28)"; c.fillStyle = "rgba(255,255,255,.85)"; c.font = `${11 * k + 3}px ${getComputedStyle(document.body).fontFamily}`;
  const x0 = Math.floor(Math.min(...xs) / step) * step, x1 = Math.max(...xs), y0 = Math.floor(Math.min(...ys) / step) * step, y1 = Math.max(...ys);
  for (let x = x0; x <= x1; x += step) { const a = applyH(inv, x, y0), b = applyH(inv, x, y1); c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke(); }
  for (let y = y0; y <= y1; y += step) { const a = applyH(inv, x0, y), b = applyH(inv, x1, y); c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke(); }
  const o = applyH(inv, 0, 0), ex = applyH(inv, step, 0), ey = applyH(inv, 0, step);
  c.lineWidth = 3 * k; c.strokeStyle = "#ff6b6b"; c.beginPath(); c.moveTo(o[0], o[1]); c.lineTo(ex[0], ex[1]); c.stroke();
  c.strokeStyle = "#5fd68a"; c.beginPath(); c.moveTo(o[0], o[1]); c.lineTo(ey[0], ey[1]); c.stroke();
  c.fillStyle = "#fff"; c.beginPath(); c.arc(o[0], o[1], 5 * k, 0, 6.3); c.fill();
}

/* stat cards */
function renderStats() {
  const o = G.overview, card = (cls, v, l) => `<div class="stat ${cls}"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  const pc = v => (v === null || v === undefined ? "—" : `${fmt(v, 0)}%`);
  const j = jobById(G.cur), still = j && j.kind === "image";
  $("#statCards").innerHTML = (still ? card("s0 wide-card", o.tufts, "tufts found") + `<div class="stat wide"><div><div class="v">${fmt(o.mean_angle_deg, 0)}°</div><div class="l">average flow direction</div></div>${compass(o.mean_angle_deg, 34)}</div>`.replace("stat wide", "stat wide") :
    card("s1", pc(o.steady_pct), "steady") + card("s2", pc(o.fluttering_pct), "fluttering") + card("s3", pc(o.stalled_pct), "stalled") +
    `<div class="stat wide"><div><div class="v">${o.tufts} tufts</div><div class="l">average flow ${fmt(o.mean_angle_deg, 0)}° · ${o.frames} frames</div></div>${compass(o.mean_angle_deg, 34)}</div>`);
  if (still) $("#statCards .stat").classList.add("wide");
}
function compass(deg, sz = 18) {
  return `<svg class="compass" style="width:${sz}px;height:${sz}px" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="10.5" fill="none" stroke="currentColor" stroke-opacity=".3" stroke-width="1.5"/><g transform="rotate(${-(deg || 0)} 12 12)"><path d="M5 12h12M13 7l5 5-5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></g></svg>`;
}

/* selected tuft card */
async function selectTuft(id) {
  G.sel = id; V.dirty = true; G.tuftData = null; renderTuftCard();
  try { G.tuftData = await api(`/jobs/${G.cur}/tuft/${id}.json`); } catch (e) { toast(e.message, true); G.sel = null; }
  renderTuftCard(); renderDataSel();
}
function posInputs(s) {
  const loc = (G.setup && G.setup.locations && G.setup.locations[String(s.tuft_id)]) || [null, null, null];
  const f = (v, i, n) => `<label>${n}<input inputmode="decimal" data-ax="${i}" class="${loc[i] !== null ? "mine" : ""}" value="${s[n.toLowerCase()] ?? ""}" aria-label="${n} position"></label>`;
  return `<div class="xyz">${f(s.x, 0, "X")}${f(s.y, 1, "Y")}${f(s.z, 2, "Z")}<button class="btn solid" type="button" id="posSave" style="height:38px">Save</button></div>`;
}
function renderTuftCard() {
  const el = $("#tuftCard");
  if (G.sel === null) {
    el.innerHTML = `<div class="tc-empty"><svg viewBox="0 0 32 32" width="30" height="30"><path d="M8 22 C14 22 17 13 25 12" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"/><rect x="5" y="20" width="5" height="5" rx="1" fill="currentColor"/></svg><h3>Click any tuft</h3><p>Pick a tuft in the video to see where it is, which way it points, and how steady it is. You can also type its measured position.</p></div>`;
    return;
  }
  const d = G.tuftData;
  if (!d) { el.innerHTML = `<div class="tc-head"><h3>Tuft ${G.sel}</h3></div><div class="tc-empty"><p class="muted">Loading…</p></div>`; return; }
  const s = d.summary, u = G.units, rec = (V.frames[V.idx] || []).find(q => q[0] === G.sel), st = rec ? STATE[rec[10]] : s.main_state;
  const src = s.position_source;
  el.innerHTML = `<div class="tc-head"><h3>Tuft ${s.tuft_id}</h3><span class="pill ${st}"><i></i>${st}</span><span class="grow"></span><button class="tc-x" type="button" id="tcClose" aria-label="Clear selection"><svg viewBox="0 0 20 20" width="14" height="14"><path d="M5 5l10 10M15 5L5 15" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg></button></div>
  <div class="tc-chart"><canvas id="tcCanvas"></canvas></div><div class="tc-cap"><span>Direction over time (degrees)</span><span>${clock(s.first_seen_s)}–${clock(s.last_seen_s)}</span></div>
  <dl class="tc-grid"><div><dt>Average direction</dt><dd>${compass(s.mean_angle_deg)}${fmt(s.mean_angle_deg, 0)}°</dd></div><div><dt>Wobble</dt><dd>±${fmt(s.angle_spread_deg, 0)}°</dd></div><div><dt>Time steady</dt><dd>${fmt(s.steady_pct, 0)}%</dd></div><div><dt>Average length</dt><dd>${fmt(s.mean_length, 1)} ${u}</dd></div></dl>
  <div class="tc-pos"><h4>Position <span>${u} · ${src}</span></h4>${posInputs(s)}<p class="msg" id="posMsg"></p></div>`;
  $("#tcClose").onclick = () => { G.sel = null; G.tuftData = null; renderTuftCard(); V.dirty = true; renderDataSel(); };
  $("#posSave").onclick = savePos;
  $$(".xyz input", el).forEach(i => i.addEventListener("keydown", e => { if (e.key === "Enter") savePos(); }));
  requestAnimationFrame(() => { drawChart(); });
}
async function savePos() {
  const v = $$("#tuftCard .xyz input").map(i => i.value.trim()), msg = $("#posMsg");
  try {
    await api(`/jobs/${G.cur}/locations/${G.sel}`, { method: "PUT", json: { x: v[0] || null, y: v[1] || null, z: v[2] || null } });
    await refreshTables(); G.tuftData = await api(`/jobs/${G.cur}/tuft/${G.sel}.json`); renderTuftCard(); const m = $("#posMsg"); if (m) { m.textContent = "Saved. Your numbers are used in every table and download."; m.className = "msg ok"; }
  } catch (e) { msg.textContent = e.message; msg.className = "msg err"; }
}
async function refreshTables() {
  const [ov, sm, st] = await Promise.all([api(`/jobs/${G.cur}/overview.json`), api(`/jobs/${G.cur}/summary.json`), api(`/jobs/${G.cur}/setup`)]);
  G.overview = ov; G.summary = sm.rows; G.units = sm.units; G.setup = st.setup; G.residuals = st.residuals; renderStats(); if (G.tab === "data") renderData();
}
function drawChart() {
  const cv = $("#tcCanvas"); if (!cv || !G.tuftData) return;
  const { w, h, dpr } = sizeCanvas(cv), c = cv.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h);
  const rows = G.tuftData.rows; if (!rows.length) return;
  const mean = G.tuftData.summary.mean_angle_deg, wrap = a => ((((a - mean) % 360) + 540) % 360) - 180;
  const ys = rows.map(r => wrap(r.angle_deg)), dev = Math.max(25, ...ys.map(Math.abs)) * 1.12;
  const t0 = rows[0].time_s, t1 = rows[rows.length - 1].time_s || 1, span = Math.max(t1 - t0, 1e-6), padL = 34, padB = 16, pw = w - padL - 4, ph = h - padB - 4;
  const X = t => padL + ((t - t0) / span) * pw, Y = a => 4 + ph / 2 - (a / dev) * (ph / 2);
  c.font = "11px " + getComputedStyle(document.body).fontFamily; c.fillStyle = css("--ink-3"); c.strokeStyle = css("--line-2"); c.lineWidth = 1; c.textAlign = "right"; c.textBaseline = "middle";
  const lo = mean - dev, hi = mean + dev, raw = (hi - lo) / 4, pw10 = Math.pow(10, Math.floor(Math.log10(raw))), mm = raw / pw10, stp = (mm < 1.5 ? 1 : mm < 3.5 ? 2 : mm < 7.5 ? 5 : 10) * pw10;
  for (let v = Math.ceil(lo / stp) * stp; v <= hi + 1e-9; v += stp) { const y = Y(v - mean); c.beginPath(); c.moveTo(padL, y); c.lineTo(w - 4, y); c.strokeStyle = css("--line-2"); c.stroke(); c.fillText(Math.round(v) + "°", padL - 5, y); }
  c.lineWidth = 2.2; c.lineJoin = "round"; c.lineCap = "round";
  for (let i = 1; i < rows.length; i++) {
    if (Math.abs(ys[i] - ys[i - 1]) > 200) continue;
    const col = css(STATE_VAR[{ learning: 0, steady: 1, fluttering: 2, stalled: 3 }[rows[i].flow_state]]);
    c.strokeStyle = col; c.beginPath(); c.moveTo(X(rows[i - 1].time_s), Y(ys[i - 1])); c.lineTo(X(rows[i].time_s), Y(ys[i])); c.stroke();
  }
  if (rows.length === 1) { c.fillStyle = css("--accent"); c.beginPath(); c.arc(w / 2, Y(ys[0]), 4, 0, 6.3); c.fill(); }
  if (V.isVideo) { const t = Math.min(Math.max(curTime(), t0), t1), x = X(t); c.strokeStyle = css("--ink"); c.lineWidth = 1.6; c.beginPath(); c.moveTo(x, 2); c.lineTo(x, h - padB); c.stroke(); }
  cv._map = { padL, pw, t0, span };
  if (!cv._bound) { cv._bound = true; cv.addEventListener("click", e => { const r = cv.getBoundingClientRect(), m = cv._map, t = m.t0 + ((e.clientX - r.left - m.padL) / m.pw) * m.span; if (V.isVideo) vid.currentTime = Math.max(0, t); }); }
}

/* ---------------------------------------------------------------- data tab */
const COLS = [
  { k: "tuft_id", t: () => "Tuft", num: true, id: true }, { k: "x", t: u => `X (${u})`, num: true, edit: 0 }, { k: "y", t: u => `Y (${u})`, num: true, edit: 1 }, { k: "z", t: u => `Z (${u})`, num: true, edit: 2 },
  { k: "main_state", t: () => "Flow" }, { k: "mean_angle_deg", t: () => "Direction", num: true }, { k: "angle_spread_deg", t: () => "Wobble", num: true }, { k: "steady_pct", t: () => "Steady · fluttering · stalled" }, { k: "seen_pct", t: () => "Seen", num: true },
];
$$("#dataSeg button").forEach(b => b.addEventListener("click", () => { G.dataView = b.dataset.dv; $$("#dataSeg button").forEach(x => x.classList.toggle("on", x === b)); G.hPage = 0; renderData(); }));
$$("#stateFilter button").forEach(b => b.addEventListener("click", () => { G.filter = b.dataset.f; $$("#stateFilter button").forEach(x => x.classList.toggle("on", x === b)); renderData(); }));
$("#search").addEventListener("input", e => { G.q = e.target.value.trim(); G.hPage = 0; renderData(); });
$("#csvBtn").addEventListener("click", () => { location.href = `/api/jobs/${G.cur}/download/${G.dataView === "tufts" ? "tuft_summary.csv" : "vector_time_history.csv"}`; });
function renderData() {
  const tufts = G.dataView === "tufts";
  $("#tuftTableWrap").hidden = !tufts; $("#histWrap").hidden = tufts; $("#stateFilter").hidden = !tufts;
  $("#csvBtn").lastChild.textContent = tufts ? " Download tuft summary" : " Download time history";
  if (tufts) renderTufts(); else renderHistory();
}
function renderTufts() {
  const u = G.units, t = $("#tuftTable");
  let rows = G.summary.filter(r => (G.filter === "all" || r.main_state === G.filter) && (!G.q || String(r.tuft_id).startsWith(G.q)));
  const { key, dir } = G.sort; rows = [...rows].sort((a, b) => ((a[key] ?? 0) > (b[key] ?? 0) ? 1 : -1) * dir);
  const th = COLS.map(c => `<th class="${c.num ? "num " : ""}${c.k !== "steady_pct" ? "sort" : ""}" data-k="${c.k}">${c.t(u)}${G.sort.key === c.k ? `<span class="ar">${G.sort.dir > 0 ? "↑" : "↓"}</span>` : ""}</th>`).join("");
  const body = rows.map(r => {
    const loc = (G.setup.locations && G.setup.locations[String(r.tuft_id)]) || [null, null, null];
    const cell = (v, i) => `<td class="num"><span class="cell-edit ${loc[i] !== null ? "mine" : "auto"}" data-id="${r.tuft_id}" data-ax="${i}" title="Click to enter your own number">${fmt(v, 3)}</span></td>`;
    const k = (r.steady_pct ?? 0) + (r.fluttering_pct ?? 0) + (r.stalled_pct ?? 0) || 1;
    return `<tr data-id="${r.tuft_id}" class="${G.sel === r.tuft_id ? "sel" : ""}"><td class="num id">${r.tuft_id}</td>${cell(r.x, 0)}${cell(r.y, 1)}${cell(r.z, 2)}<td><span class="pill ${r.main_state}"><i></i>${r.main_state}</span></td><td class="num">${compass(r.mean_angle_deg)}${fmt(r.mean_angle_deg, 0)}°</td><td class="num">±${fmt(r.angle_spread_deg, 0)}°</td><td><span class="mix" title="${fmt(r.steady_pct, 0)}% steady, ${fmt(r.fluttering_pct, 0)}% fluttering, ${fmt(r.stalled_pct, 0)}% stalled"><i class="a" style="width:${(r.steady_pct / k) * 100}%"></i><i class="b" style="width:${(r.fluttering_pct / k) * 100}%"></i><i class="c" style="width:${(r.stalled_pct / k) * 100}%"></i></span></td><td class="num">${fmt(r.seen_pct, 0)}%</td></tr>`;
  }).join("");
  t.innerHTML = `<thead><tr>${th}</tr></thead><tbody>${body || `<tr class="empty-row"><td colspan="9">No tufts match.</td></tr>`}</tbody>`;
  $$("th.sort", t).forEach(h => h.addEventListener("click", () => { const k = h.dataset.k; G.sort = { key: k, dir: G.sort.key === k ? -G.sort.dir : 1 }; renderTufts(); }));
  $$("tbody tr[data-id]", t).forEach(tr => tr.addEventListener("click", e => { if (e.target.closest(".cell-edit, .cell-in")) return; const id = Number(tr.dataset.id); selectTuft(id); G.tab = "video"; location.hash = `#/run/${G.cur}/video`; }));
  $$(".cell-edit", t).forEach(sp => sp.addEventListener("click", e => { e.stopPropagation(); editCell(sp); }));
  $("#dataNote").textContent = `${rows.length} of ${G.summary.length} tufts. Grey numbers were worked out from the picture. Blue numbers are ones you entered. Click a number to change it, or a row to see that tuft in the video.`;
}
function editCell(sp) {
  const id = Number(sp.dataset.id), ax = Number(sp.dataset.ax), cur = sp.textContent.replace(/[^\d.\-eE]/g, "");
  const inp = document.createElement("input"); inp.className = "cell-in"; inp.value = cur; inp.inputMode = "decimal"; sp.replaceWith(inp); inp.focus(); inp.select();
  const done = async save => {
    if (!inp.isConnected) return;
    if (!save) { renderTufts(); return; }
    const loc = ((G.setup.locations || {})[String(id)] || [null, null, null]).slice(); loc[ax] = inp.value.trim() === "" ? null : inp.value.trim();
    try { await api(`/jobs/${G.cur}/locations/${id}`, { method: "PUT", json: { x: loc[0], y: loc[1], z: loc[2] } }); await refreshTables(); if (G.sel === id) G.tuftData = await api(`/jobs/${G.cur}/tuft/${id}.json`); }
    catch (e) { toast(e.message, true); renderTufts(); }
  };
  inp.addEventListener("keydown", e => { if (e.key === "Enter") done(true); if (e.key === "Escape") done(false); });
  inp.addEventListener("blur", () => done(inp.value.trim() !== cur));
}
function renderDataSel() { $$("#tuftTable tbody tr").forEach(tr => tr.classList.toggle("sel", Number(tr.dataset.id) === G.sel)); }
async function renderHistory() {
  const t = $("#histTable"), lim = 100;
  const qs = new URLSearchParams({ offset: G.hPage * lim, limit: lim }); if (G.q) qs.set("tuft", G.q);
  let d; try { d = await api(`/jobs/${G.cur}/history.json?${qs}`); } catch (e) { toast(e.message, true); return; }
  const u = G.units, H = ["Time (s)", "Tuft", `X (${u})`, `Y (${u})`, `Z (${u})`, "Vector X", "Vector Y", "Vector Z", "Direction", "Flow"];
  t.innerHTML = `<thead><tr>${H.map((h, i) => `<th class="${i ? "num" : ""}">${h}</th>`).join("")}</tr></thead><tbody>${d.rows.map(r => `<tr style="cursor:default"><td class="num">${fmt(r.time_s, 3)}</td><td class="num id">${r.tuft_id}</td><td class="num">${fmt(r.x, 3)}</td><td class="num">${fmt(r.y, 3)}</td><td class="num">${fmt(r.z, 3)}</td><td class="num">${fmt(r.vector_x, 4)}</td><td class="num">${fmt(r.vector_y, 4)}</td><td class="num">${fmt(r.vector_z, 4)}</td><td class="num">${fmt(r.angle_deg, 1)}°</td><td><span class="pill ${r.flow_state}"><i></i>${r.flow_state}</span></td></tr>`).join("") || `<tr class="empty-row"><td colspan="10">No rows.</td></tr>`}</tbody>`;
  const pages = Math.max(1, Math.ceil(d.total / lim));
  $("#hInfo").textContent = `Rows ${d.total ? G.hPage * lim + 1 : 0}–${Math.min(d.total, (G.hPage + 1) * lim)} of ${d.total.toLocaleString()}`;
  $("#hPrev").disabled = G.hPage === 0; $("#hNext").disabled = G.hPage >= pages - 1;
  $("#dataNote").textContent = "Vector X, Y and Z make a unit vector (length 1) pointing the way the air pushes the tuft, in your coordinate axes.";
}
$("#hPrev").addEventListener("click", () => { G.hPage = Math.max(0, G.hPage - 1); renderHistory(); });
$("#hNext").addEventListener("click", () => { G.hPage++; renderHistory(); });

/* ---------------------------------------------------------------- setup tab */
const C = { mode: "two", pts: [], dist: "", drag: -1, loaded: null };
const cstage = { img: $("#coordImg"), cv: $("#coordCv") };
function renderSetup(j) {
  cstage.img.src = `/api/jobs/${j.id}/first_frame.jpg`;
  if (C.loaded !== j.id + ":" + j.version) { loadCoordFromSetup(); C.loaded = j.id + ":" + j.version; }
  $("#unitSel").value = (G.setup.coords && G.setup.coords.units && G.setup.coords.units !== "px") ? G.setup.coords.units : "mm";
  $("#zPlane").value = (G.setup.coords && G.setup.coords.z_plane) || 0; $("#liftOn").checked = !!G.setup.estimate_lift;
  $("#yUp").checked = G.setup.coords && G.setup.coords.y_up !== false;
  const o = j.options || {}; $("#optPol").value = o.polarity || "auto"; $("#optSize").value = o.size || "auto"; $("#optStep").value = String(o.step || 0); $("#optEngine").value = o.engine || "auto";
  $("#optTrack").value = o.tracking || "map"; $("#optWeights").value = o.weights || "";
  $("#optTrackWrap").hidden = $("#optMapWrap").hidden = j.kind !== "video";
  fillMapNames(o.map_name || ""); loadMapCard(j);
  $("#optStepWrap").hidden = j.kind !== "video"; if (G.roi === null && o.roi) G.roi = o.roi; $("#roiTxt").textContent = G.roi ? `${G.roi[2]}×${G.roi[3]} px area chosen` : "Whole picture";
  syncCoordUi(); requestAnimationFrame(drawCoord);
}
function loadCoordFromSetup() {
  const c = G.setup.coords; C.pts = []; C.dist = "";
  if (c && c.defined && c.points && c.points.length) {
    if (c.method === "origin_axis" && c.points.length === 2) { C.mode = "two"; C.pts = [{ px: c.points[0][0], py: c.points[0][1] }, { px: c.points[1][0], py: c.points[1][1] }]; C.dist = String(c.points[1][2]); }
    else { C.mode = "many"; C.pts = c.points.map(p => ({ px: p[0], py: p[1], x: String(p[2]), y: String(p[3]) })); }
  } else C.mode = "two";
  $$("#coordMode button").forEach(b => b.classList.toggle("on", b.dataset.m === C.mode));
}
$$("#coordMode button").forEach(b => b.addEventListener("click", () => { C.mode = b.dataset.m; $$("#coordMode button").forEach(x => x.classList.toggle("on", x === b)); if (C.mode === "two") C.pts = C.pts.slice(0, 2); syncCoordUi(); drawCoord(); }));
function syncCoordUi() {
  const n = C.pts.length, two = C.mode === "two";
  $("#coordHint").textContent = two ? (n === 0 ? "Click the origin: the point that should be (0, 0)." : n === 1 ? "Now click a second point along the X axis." : "Type the real distance between the two points, then save.")
    : (n < 2 ? "Click a point whose real position you know." : n < 3 ? "Add a third point if you can. With 4 or more, TuftView also corrects for a tilted camera." : "Type each point’s real X and Y. Drag a point to adjust it.");
  const box = $("#pointsBox"); const u = $("#unitSel").value;
  if (!n) { box.innerHTML = `<p class="muted">No points yet. Click on the picture.</p>`; }
  else if (two) {
    box.innerHTML = `<div class="pt-row"><span class="n">1</span><span>Origin (0, 0)</span><span></span><button class="rm" data-rm="0" type="button" aria-label="Remove point 1">×</button></div>` +
      (n > 1 ? `<div class="pt-row"><span class="n">2</span><span>Distance along X</span><input id="dist" inputmode="decimal" placeholder="e.g. 500" value="${esc(C.dist)}" aria-label="Distance along X in ${u}"><button class="rm" data-rm="1" type="button" aria-label="Remove point 2">×</button></div>` : "");
  } else {
    box.innerHTML = `<div class="pt-lab"><span></span><span>Real X</span><span>Real Y</span><span></span></div>` + C.pts.map((p, i) => `<div class="pt-row"><span class="n">${i + 1}</span><input data-i="${i}" data-a="x" inputmode="decimal" value="${esc(p.x ?? "")}" aria-label="Point ${i + 1} X"><input data-i="${i}" data-a="y" inputmode="decimal" value="${esc(p.y ?? "")}" aria-label="Point ${i + 1} Y"><button class="rm" data-rm="${i}" type="button" aria-label="Remove point ${i + 1}">×</button></div>`).join("");
  }
  $$("#pointsBox .rm").forEach(b => b.addEventListener("click", () => { C.pts.splice(Number(b.dataset.rm), 1); syncCoordUi(); drawCoord(); }));
  const d = $("#dist"); if (d) d.addEventListener("input", () => (C.dist = d.value));
  $$("#pointsBox input[data-i]").forEach(i => i.addEventListener("input", () => (C.pts[Number(i.dataset.i)][i.dataset.a] = i.value)));
  $("#yUpWrap").hidden = !two;
}
$("#unitSel").addEventListener("change", syncCoordUi);
function coordGeom() { const el = cstage.img, cv = cstage.cv, { w, h, dpr } = sizeCanvas(cv), b = el.getBoundingClientRect(), f = fitRect(b.width, b.height, G.results.width, G.results.height); return { w, h, dpr, ...f }; }
function drawCoord() {
  if (!G.results || $("#pane-setup").hidden) return;
  const g = coordGeom(), c = cstage.cv.getContext("2d"); c.setTransform(g.dpr, 0, 0, g.dpr, 0, 0); c.clearRect(0, 0, g.w, g.h);
  const P = p => [g.x0 + p.px * g.s, g.y0 + p.py * g.s];
  const co = G.setup.coords;
  if (co && co.defined && co.H) { // show the saved system as a grid
    const inv = inv3(co.H);
    if (inv) {
      c.save(); c.translate(g.x0, g.y0); c.scale(g.s, g.s);
      const k = Math.max(G.results.width, G.results.height) / 1000; const sv = G.units; void sv;
      const cs = [[0, 0], [G.results.width, 0], [G.results.width, G.results.height], [0, G.results.height]].map(p => applyH(co.H, p[0], p[1]));
      const xs = cs.map(p => p[0]), ys = cs.map(p => p[1]), raw = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) / 8, pw = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / pw, step = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * pw;
      c.strokeStyle = "rgba(255,255,255,.25)"; c.lineWidth = k;
      for (let x = Math.floor(Math.min(...xs) / step) * step; x <= Math.max(...xs); x += step) { const a = applyH(inv, x, Math.min(...ys)), b = applyH(inv, x, Math.max(...ys)); c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke(); }
      for (let y = Math.floor(Math.min(...ys) / step) * step; y <= Math.max(...ys); y += step) { const a = applyH(inv, Math.min(...xs), y), b = applyH(inv, Math.max(...xs), y); c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke(); }
      const o = applyH(inv, 0, 0), ex = applyH(inv, step, 0), ey = applyH(inv, 0, step);
      c.lineWidth = 3 * k; c.strokeStyle = "#ff6b6b"; c.beginPath(); c.moveTo(o[0], o[1]); c.lineTo(ex[0], ex[1]); c.stroke(); c.strokeStyle = "#5fd68a"; c.beginPath(); c.moveTo(o[0], o[1]); c.lineTo(ey[0], ey[1]); c.stroke();
      c.restore();
    }
  }
  if (C.mode === "two" && C.pts.length === 2) { const a = P(C.pts[0]), b = P(C.pts[1]); c.strokeStyle = "#fff"; c.setLineDash([6, 5]); c.lineWidth = 2; c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke(); c.setLineDash([]); }
  C.pts.forEach((p, i) => { const [x, y] = P(p); c.fillStyle = css("--accent"); c.strokeStyle = "#fff"; c.lineWidth = 2.5; c.beginPath(); c.arc(x, y, 11, 0, 6.3); c.fill(); c.stroke(); c.fillStyle = "#fff"; c.font = "700 12px " + getComputedStyle(document.body).fontFamily; c.textAlign = "center"; c.textBaseline = "middle"; c.fillText(i + 1, x, y + 0.5); });
}
function coordPt(e) { const r = cstage.cv.getBoundingClientRect(), g = coordGeom(); return { px: Math.max(0, Math.min(G.results.width, (e.clientX - r.left - g.x0) / g.s)), py: Math.max(0, Math.min(G.results.height, (e.clientY - r.top - g.y0) / g.s)), sx: e.clientX - r.left, sy: e.clientY - r.top }; }
cstage.cv.addEventListener("pointerdown", e => {
  const p = coordPt(e), g = coordGeom();
  const near = C.pts.findIndex(q => Math.hypot(g.x0 + q.px * g.s - p.sx, g.y0 + q.py * g.s - p.sy) < 15);
  if (near >= 0) { C.drag = near; cstage.cv.setPointerCapture(e.pointerId); return; }
  if (C.mode === "two" && C.pts.length >= 2) return;
  C.pts.push({ px: p.px, py: p.py, x: "", y: "" }); syncCoordUi(); drawCoord();
});
cstage.cv.addEventListener("pointermove", e => { if (C.drag < 0) return; const p = coordPt(e); C.pts[C.drag].px = p.px; C.pts[C.drag].py = p.py; drawCoord(); });
cstage.cv.addEventListener("pointerup", () => (C.drag = -1));
cstage.img.addEventListener("load", () => { drawCoord(); });
addEventListener("resize", () => { if (G.tab === "setup") drawCoord(); });
function setMsg(id, text, kind) { const m = $(id); m.textContent = text; m.className = "msg " + (kind || ""); }
$("#applyCoord").addEventListener("click", async () => {
  const units = $("#unitSel").value, z = $("#zPlane").value || 0; let body;
  if (C.mode === "two") {
    if (C.pts.length < 2) return setMsg("#coordMsg", "Click two points on the picture first.", "err");
    body = { method: "origin_axis", origin: [C.pts[0].px, C.pts[0].py], axis: [C.pts[1].px, C.pts[1].py], distance: C.dist, units, y_up: $("#yUp").checked, z_plane: z };
  } else {
    if (C.pts.length < 2) return setMsg("#coordMsg", "Add at least two points.", "err");
    if (C.pts.some(p => p.x === "" || p.y === "" || p.x === undefined || p.y === undefined || isNaN(Number(p.x)) || isNaN(Number(p.y)))) return setMsg("#coordMsg", "Type a real X and Y for every point.", "err");
    body = { method: "points", points: C.pts.map(p => [p.px, p.py, Number(p.x), Number(p.y)]), units, z_plane: z };
  }
  try {
    const r = await api(`/jobs/${G.cur}/coords`, { method: "PUT", json: body });
    await refreshTables(); C.loaded = null;
    const worst = r.residuals && r.residuals.length ? Math.max(...r.residuals) : 0;
    setMsg("#coordMsg", `Saved. 1 pixel = ${fmt(r.upp, 4)} ${units}.` + (C.mode === "many" ? ` Largest mismatch at your points: ${fmt(worst, 3)} ${units}.` : ""), "ok");
    $("#gridTog").hidden = false; drawCoord();
    if (jobById(G.cur).kind === "video") api(`/jobs/${G.cur}/rerender`, { method: "POST" }).catch(() => {});
  } catch (e) { setMsg("#coordMsg", e.message, "err"); }
});
$("#clearCoord").addEventListener("click", async () => {
  try { await api(`/jobs/${G.cur}/coords`, { method: "PUT", json: { clear: true } }); C.pts = []; C.loaded = null; await refreshTables(); G.layers.grid = false; $("#gridTog").hidden = true; renderSetup(jobById(G.cur)); setMsg("#coordMsg", "Removed. Positions are in picture pixels, with (0, 0) at the bottom-left.", "ok"); }
  catch (e) { setMsg("#coordMsg", e.message, "err"); }
});
$("#liftOn").addEventListener("change", async e => { try { await api(`/jobs/${G.cur}/lift`, { method: "PUT", json: { on: e.target.checked } }); await refreshTables(); setMsg("#locMsg", e.target.checked ? "Z direction is now estimated from how much each tuft shortens." : "Z direction is zero (flat on the surface).", "ok"); } catch (er) { setMsg("#locMsg", er.message, "err"); } });
$("#locDownload").addEventListener("click", () => (location.href = `/api/jobs/${G.cur}/download/tuft_locations.csv`));
$("#locFile").addEventListener("change", async e => {
  const f = e.target.files[0]; if (!f) return; const fd = new FormData(); fd.append("file", f);
  try { const r = await api(`/jobs/${G.cur}/locations.csv`, { method: "POST", body: fd }); await refreshTables(); setMsg("#locMsg", `Imported positions for ${r.imported} tufts.`, "ok"); }
  catch (er) { setMsg("#locMsg", er.message, "err"); } e.target.value = "";
});
$("#locClear").addEventListener("click", async () => { if (!confirm("Remove every position you entered? Automatic positions stay.")) return; try { await api(`/jobs/${G.cur}/locations`, { method: "DELETE" }); await refreshTables(); setMsg("#locMsg", "Your positions were removed.", "ok"); } catch (er) { setMsg("#locMsg", er.message, "err"); } });

/* analysis settings + area picker */
const roi = { img: $("#roiImg"), cv: $("#roiCv"), r: null, drag: null };
$("#roiBtn").addEventListener("click", () => { roi.img.src = `/api/jobs/${G.cur}/first_frame.jpg`; roi.r = G.roi ? G.roi.slice() : null; $("#roiDlg").showModal(); setTimeout(drawRoi, 60); });
$("#roiClose").addEventListener("click", () => $("#roiDlg").close());
$("#roiAll").addEventListener("click", () => { G.roi = null; $("#roiTxt").textContent = "Whole picture"; $("#roiDlg").close(); });
$("#roiOk").addEventListener("click", () => { if (roi.r && roi.r[2] > 8 && roi.r[3] > 8) { G.roi = roi.r.map(Math.round); $("#roiTxt").textContent = `${G.roi[2]}×${G.roi[3]} px area chosen`; } $("#roiDlg").close(); });
roi.img.addEventListener("load", drawRoi);
function roiGeom() { const { w, h, dpr } = sizeCanvas(roi.cv), b = roi.img.getBoundingClientRect(); return { w, h, dpr, ...fitRect(b.width, b.height, G.results.width, G.results.height) }; }
function drawRoi() {
  if (!G.results) return; const g = roiGeom(), c = roi.cv.getContext("2d"); c.setTransform(g.dpr, 0, 0, g.dpr, 0, 0); c.clearRect(0, 0, g.w, g.h);
  if (!roi.r) return; const [x, y, w, h] = roi.r, X = g.x0 + x * g.s, Y = g.y0 + y * g.s, W = w * g.s, H = h * g.s;
  c.fillStyle = "rgba(8,14,20,.6)"; c.beginPath(); c.rect(0, 0, g.w, g.h); c.rect(X, Y, W, H); c.fill("evenodd"); c.strokeStyle = "#fff"; c.lineWidth = 2; c.strokeRect(X, Y, W, H);
}
function roiPt(e) { const g = roiGeom(), r = roi.cv.getBoundingClientRect(); return [Math.max(0, Math.min(G.results.width, (e.clientX - r.left - g.x0) / g.s)), Math.max(0, Math.min(G.results.height, (e.clientY - r.top - g.y0) / g.s))]; }
roi.cv.addEventListener("pointerdown", e => { roi.drag = roiPt(e); roi.cv.setPointerCapture(e.pointerId); });
roi.cv.addEventListener("pointermove", e => { if (!roi.drag) return; const p = roiPt(e); roi.r = [Math.min(roi.drag[0], p[0]), Math.min(roi.drag[1], p[1]), Math.abs(p[0] - roi.drag[0]), Math.abs(p[1] - roi.drag[1])]; drawRoi(); });
roi.cv.addEventListener("pointerup", () => (roi.drag = null));
$("#reBtn").addEventListener("click", async () => {
  if (!confirm("Analyze this file again with these settings? The current results are replaced.")) return;
  try { await api(`/jobs/${G.cur}/reanalyze`, { json: { polarity: $("#optPol").value, size: $("#optSize").value, step: Number($("#optStep").value), engine: $("#optEngine").value, tracking: $("#optTrack").value, weights: $("#optWeights").value, map_name: $("#optMapName").value, map_mode: "build", review: true, roi: G.roi } }); G.loadedId = null; await pollOnce(); }
  catch (e) { toast(e.message, true); }
});
/* ---- tuft map review (trained model) */
const TM = { map: null, add: [], remove: new Set(), flip: false, img: new Image(), jid: null };
function fillMapNames(sel) {
  const names = (G.status && G.status.maps) || [];
  $("#optMapName").innerHTML = `<option value="">Build from this video</option>` + names.map(n => `<option>${esc(n)}</option>`).join("");
  $("#optMapName").value = names.includes(sel) ? sel : "";
}
async function loadMapCard(j) {
  const card = $("#mapCard");
  if (j.kind !== "video" || j.status !== "done") { card.hidden = true; return; }
  try {
    const r = await api(`/jobs/${j.id}/map`);
    if (!r.map) { card.hidden = true; return; }
    TM.map = r.map; TM.add = []; TM.remove = new Set(); TM.jid = j.id; $("#mapFlip").hidden = !r.map.can_flip;
    G.status = G.status || {}; G.status.maps = r.saved; fillMapNames(($("#optMapName").value) || "");
    TM.img.onload = drawMap; TM.img.src = `/api/jobs/${j.id}/map_ref.jpg?v=${Date.now()}`;
    card.hidden = false; setMsg("#mapMsg", "");
  } catch (_) { card.hidden = true; }
}
function drawMap() {
  const cv = $("#mapCv"), m = TM.map; if (!m || !TM.img.naturalWidth) return;
  const W = TM.img.naturalWidth, H = TM.img.naturalHeight; cv.width = W; cv.height = H;
  const c = cv.getContext("2d"); c.drawImage(TM.img, 0, 0);
  const k = Math.max(W, H) / 900, ang = Math.atan2(m.flow[1], m.flow[0]);
  const len = 34 * k;
  const dot = (x, y, col, ring) => {
    c.lineWidth = 2 * k; c.strokeStyle = col; c.fillStyle = ring ? "rgba(0,0,0,0)" : col;
    c.beginPath(); c.arc(x, y, 5 * k, 0, 6.3); ring ? c.stroke() : c.fill();
    c.beginPath(); c.moveTo(x, y); c.lineTo(x + Math.cos(ang) * len, y + Math.sin(ang) * len); c.stroke();
  };
  m.anchors.forEach(a => { const gone = TM.remove.has(a.id); dot(a.x, a.y, gone ? "#ef4444" : "#4ade80", gone); });
  TM.add.forEach(p => dot(p[0], p[1], "#38bdf8", false));
  const n = m.anchors.length - TM.remove.size + TM.add.length;
  $("#mapTxt").textContent = `${n} tufts` + (TM.flip ? " · direction will be flipped" : "");
}
$("#mapCv").addEventListener("click", e => {
  const cv = e.currentTarget, r = cv.getBoundingClientRect(), x = (e.clientX - r.left) * cv.width / r.width, y = (e.clientY - r.top) * cv.height / r.height;
  const reach = 14 * Math.max(cv.width, cv.height) / 900;
  let best = null, bd = 1e9;
  TM.map.anchors.forEach(a => { const d = Math.hypot(a.x - x, a.y - y); if (d < bd) { bd = d; best = a; } });
  const q = TM.add.findIndex(p => Math.hypot(p[0] - x, p[1] - y) < reach);
  if (q >= 0) TM.add.splice(q, 1);
  else if (best && bd < reach) { TM.remove.has(best.id) ? TM.remove.delete(best.id) : TM.remove.add(best.id); }
  else TM.add.push([x, y]);
  drawMap();
});
$("#mapFlip").addEventListener("click", () => { TM.flip = !TM.flip; $("#mapFlip").classList.toggle("on", TM.flip); drawMap(); });
$("#mapUndo").addEventListener("click", () => { TM.add = []; TM.remove = new Set(); TM.flip = false; drawMap(); });
async function commitMap() {
  if (!TM.add.length && !TM.remove.size && !TM.flip) return;
  const r = await api(`/jobs/${TM.jid}/map`, { method: "PUT", json: { add: TM.add, remove: [...TM.remove], flip: TM.flip } });
  TM.map = r.map; TM.add = []; TM.remove = new Set(); TM.flip = false; drawMap();
}
$("#mapSave").addEventListener("click", async () => {
  try { await commitMap(); const r = await api(`/jobs/${TM.jid}/map/save`, { json: { name: $("#mapName").value } });
    G.status.maps = r.saved; fillMapNames(r.name); setMsg("#mapMsg", `Saved as “${r.name}”. Pick it under Tuft map to use it on other videos.`, "ok"); }
  catch (e) { setMsg("#mapMsg", e.message, "err"); }
});
$("#mapApply").addEventListener("click", async () => {
  if (!confirm("Analyze again with this tuft map? The current results are replaced.")) return;
  try { await commitMap(); await api(`/jobs/${G.cur}/reanalyze`, { json: { map_mode: "keep", review: false, map_name: "", tracking: "map", engine: $("#optEngine").value, weights: $("#optWeights").value, step: Number($("#optStep").value) } }); G.loadedId = null; await pollOnce(); }
  catch (e) { setMsg("#mapMsg", e.message, "err"); }
});

/* ---------------------------------------------------------------- the pop-up: check the tufts, then process every frame */
const RV = { jid: null, kind: "map", map: null, tufts: null, img: new Image(), W: 1, H: 1, add: [], remove: new Set(), flip: false, mode: "add",
  hist: [], zoom: 1, cx: 0, cy: 0, drag: null, hover: null, dismissed: new Set(), busy: false };
const rvDlg = $("#reviewDlg"), rvCv = $("#rvCv"), rvStage = $("#rvStage");
function showReviewPanel(j) {
  $("#reviewThumb").src = `/api/jobs/${j.id}/first_frame.jpg`;
  $("#reviewPanelTxt").textContent = `${j.tufts} tufts were found. Remove wrong ones and add missed ones, then process ${j.kind === "video" ? "every frame" : "the picture"}.`;
  if (!rvDlg.open && !RV.dismissed.has(j.id)) openReview(j.id);
}
$("#reviewOpen").addEventListener("click", () => openReview(G.cur));
$("#reviewDel").addEventListener("click", async () => { try { await api(`/jobs/${G.cur}`, { method: "DELETE" }); G.jobs = G.jobs.filter(k => k.id !== G.cur); showHome(); renderRail(); } catch (e) { toast(e.message, true); } });
async function openReview(id) {
  const j = jobById(id); if (!j || rvDlg.open) return;
  Object.assign(RV, { jid: id, kind: j.kind === "video" ? "map" : "image", add: [], remove: new Set(), flip: false, hist: [], mode: "add", map: null, tufts: null, busy: false });
  setRvMode("add"); setMsg("#rvMsg", ""); $("#rvName").value = ""; $("#rvFlip").classList.remove("on");
  rvFillSteps(j); $("#rvTitle").textContent = j.kind === "video" ? "Check the tufts before processing" : "Check the tufts in this picture";
  $("#rvNameWrap").hidden = $("#rvFlip").hidden = j.kind !== "video"; $("#rvStep").closest("label").hidden = j.kind !== "video";
  $("#rvGo").textContent = j.kind === "video" ? "Process all frames" : "Process picture"; $("#rvGo").disabled = true;
  $("#rvSub").textContent = "Loading the tufts…";
  rvDlg.showModal();
  try {
    if (RV.kind === "map") {
      const r = await api(`/jobs/${id}/map`); if (!r.map) throw new Error("No tuft map was found for this video.");
      RV.map = r.map; RV.W = r.map.size[0]; RV.H = r.map.size[1]; $("#rvFlip").hidden = !r.map.can_flip;
      await rvLoadImage(`/api/jobs/${id}/map_ref.jpg?v=${Date.now()}`);
      G.status = G.status || {}; G.status.maps = r.saved;
    } else {
      const r = await api(`/jobs/${id}/image_tufts`); RV.tufts = r.tufts; RV.W = r.size[0]; RV.H = r.size[1];
      await rvLoadImage(`/api/jobs/${id}/first_frame.jpg`);
    }
  } catch (e) { rvDlg.close(); toast(e.message, true); return; }
  RV.zoom = 1; rvUpdate(); rvHint(); $("#rvGo").disabled = false; rvResize();
}
/** "About 12 min" from the scan speed (seconds per frame the model needed while building the map). */
function rvDur(sec) { if (sec < 75) return "about 1 min"; const m = Math.round(sec / 60); return m < 90 ? `about ${m} min` : `about ${(m / 60).toFixed(1)} h`; }
function rvFillSteps(j) {
  const n = (j.info && j.info.frames) || 0, spf = j.scan_spf ? j.scan_spf * 1.15 : 0;       // a little extra for drawing and saving
  const t = st => (spf && n ? ` · ${rvDur(Math.ceil(n / st) * spf)}` : "");
  const auto = Math.max(1, Math.ceil(n / 600)), want = spf && n ? Math.max(1, Math.ceil((n * spf) / 300)) : 0;
  const rows = [[1, "Every frame (most detail)"], [2, "Every 2nd frame"], [3, "Every 3rd frame"], [4, "Every 4th frame"], [6, "Every 6th frame"], [8, "Every 8th frame"]];
  let html = rows.map(([st, lab]) => `<option value="${st}">${lab}${t(st)}</option>`).join("");
  if (want > 1) html = `<option value="${want}">Fast: about 5 min (every ${want === 2 ? "2nd" : want === 3 ? "3rd" : want + "th"} frame)</option>` + html;
  html += `<option value="0">Automatic (up to about 600 frames)${t(auto)}</option>`;
  $("#rvStep").innerHTML = html; $("#rvStep").value = "1";
}
function rvLoadImage(url) { return new Promise((res, rej) => { RV.img.onload = () => res(); RV.img.onerror = () => rej(new Error("The first picture could not be loaded.")); RV.img.src = url; }); }
function rvHint() {
  $("#rvHint").textContent = RV.kind === "map"
    ? "Each dot is the taped end of one tuft. Add tufts: click a missed tuft’s taped end (drag to move the picture). Remove tufts: click a dot, or drag a box around many. Scroll to zoom, right-drag to move. Keys: A add, R remove, Z undo."
    : "Add tufts: click a missed tuft, or drag from its taped end to its free end. Remove tufts: click a tuft, or drag a box around many. Scroll to zoom, right-drag to move. Keys: A add, R remove, Z undo.";
}
const rvItems = () => RV.kind === "map" ? RV.map.anchors : RV.tufts;
const rvCount = () => rvItems().length - RV.remove.size + RV.add.length;
function rvUpdate() {
  const n = rvCount(); $("#rvCount").textContent = `${n} tuft${n === 1 ? "" : "s"}`;
  $("#rvSub").textContent = RV.kind === "map" ? "Green dots are tufts the model found in most frames, yellow ones in fewer (check those), blue ones you added. Red rings will be removed."
    : "Green lines are the tufts the model found (the dot is the taped end), blue ones you added. Red ones will be removed.";
  $("#rvUndo").disabled = !RV.hist.length; rvDraw();
}
function rvSnap() { RV.hist.push({ add: RV.add.map(a => a.slice ? a.slice() : { ...a }), remove: [...RV.remove], flip: RV.flip }); if (RV.hist.length > 200) RV.hist.shift(); }
function rvUndo() { const h = RV.hist.pop(); if (!h) return; RV.add = h.add; RV.remove = new Set(h.remove); RV.flip = h.flip; $("#rvFlip").classList.toggle("on", RV.flip); rvUpdate(); }
function setRvMode(m) { RV.mode = m; $$("#rvMode button").forEach(b => b.classList.toggle("on", b.dataset.mode === m)); rvStage.classList.toggle("add", m === "add"); rvStage.classList.toggle("remove", m === "remove"); }
$$("#rvMode button").forEach(b => b.addEventListener("click", () => setRvMode(b.dataset.mode)));
$("#rvUndo").addEventListener("click", rvUndo);
$("#rvFlip").addEventListener("click", () => { rvSnap(); RV.flip = !RV.flip; $("#rvFlip").classList.toggle("on", RV.flip); rvUpdate(); });

/* view: the picture is fitted into the stage, then zoomed around (cx, cy) in picture pixels */
function rvView() {
  const { w, h, dpr } = sizeCanvas(rvCv), base = Math.min(w / RV.W, h / RV.H), s = base * RV.zoom;
  return { w, h, dpr, base, s, ox: w / 2 - RV.cx * s, oy: h / 2 - RV.cy * s };
}
function rvClamp() {
  const v = rvView(), vw = v.w / v.s, vh = v.h / v.s;
  RV.cx = vw >= RV.W ? RV.W / 2 : Math.max(vw / 2, Math.min(RV.W - vw / 2, RV.cx));
  RV.cy = vh >= RV.H ? RV.H / 2 : Math.max(vh / 2, Math.min(RV.H - vh / 2, RV.cy));
}
function rvResize() { if (!rvDlg.open) return; if (RV.zoom === 1) { RV.cx = RV.W / 2; RV.cy = RV.H / 2; } rvClamp(); rvDraw(); }
addEventListener("resize", rvResize);
function rvZoomAt(f, px, py) {
  const v = rvView(), ix = (px - v.ox) / v.s, iy = (py - v.oy) / v.s;
  RV.zoom = Math.max(1, Math.min(40, RV.zoom * f)); const s = rvView().s;
  RV.cx = ix - (px - v.w / 2) / s; RV.cy = iy - (py - v.h / 2) / s; rvClamp(); rvDraw();
}
$("#rvIn").addEventListener("click", () => { const v = rvView(); rvZoomAt(1.5, v.w / 2, v.h / 2); });
$("#rvOut").addEventListener("click", () => { const v = rvView(); rvZoomAt(1 / 1.5, v.w / 2, v.h / 2); });
$("#rvFit").addEventListener("click", () => { RV.zoom = 1; RV.cx = RV.W / 2; RV.cy = RV.H / 2; rvDraw(); });
rvCv.addEventListener("wheel", e => { e.preventDefault(); const r = rvCv.getBoundingClientRect(); rvZoomAt(e.deltaY < 0 ? 1.2 : 1 / 1.2, e.clientX - r.left, e.clientY - r.top); }, { passive: false });
rvCv.addEventListener("contextmenu", e => e.preventDefault());

function rvDraw() {
  if (!rvDlg.open || !RV.img.naturalWidth || (!RV.map && !RV.tufts)) return;
  const v = rvView(), c = rvCv.getContext("2d");
  c.setTransform(v.dpr, 0, 0, v.dpr, 0, 0); c.clearRect(0, 0, v.w, v.h);
  c.imageSmoothingQuality = "high"; c.drawImage(RV.img, v.ox, v.oy, RV.W * v.s, RV.H * v.s);
  const P = (x, y) => [v.ox + x * v.s, v.oy + y * v.s], R = Math.max(3.5, Math.min(7, 3 + RV.zoom * 0.6));
  c.lineCap = "round"; c.lineJoin = "round";
  const halo = (draw) => { c.strokeStyle = "rgba(0,0,0,.65)"; c.lineWidth = 4.2; draw(); };
  const dot = (x, y, col, ring) => { const [X, Y] = P(x, y); c.beginPath(); c.arc(X, Y, R, 0, 6.3); halo(() => c.stroke()); c.strokeStyle = col; c.lineWidth = 2.2; if (ring) c.stroke(); else { c.fillStyle = col; c.fill(); c.stroke(); } };
  const seg = (x1, y1, x2, y2, col) => { const [A, B] = P(x1, y1), [C, D] = P(x2, y2); c.beginPath(); c.moveTo(A, B); c.lineTo(C, D); halo(() => c.stroke()); c.strokeStyle = col; c.lineWidth = 2; c.stroke(); };
  const GREEN = "#4ade80", YEL = "#fbbf24", CYAN = "#38bdf8", RED = "#ef4444";
  if (RV.kind === "map") {
    const m = RV.map, L = m.median_length || 40, fa = Math.atan2(m.flow[1], m.flow[0]) + (RV.flip ? Math.PI : 0);
    const dir = a => RV.flip ? fa : (a.length > 0 ? -a.angle * Math.PI / 180 : fa);
    const draw = (a, col, ring) => { const ln = 0.6 * (a.length || L), d = dir(a); seg(a.x, a.y, a.x + Math.cos(d) * ln, a.y + Math.sin(d) * ln, col); dot(a.x, a.y, col, ring); };
    m.anchors.forEach(a => { const gone = RV.remove.has(a.id); draw(a, gone ? RED : a.manual ? CYAN : a.hits >= 0.5 ? GREEN : YEL, gone); });
    RV.add.forEach(p => draw({ x: p[0], y: p[1], length: L }, CYAN, false));
    // the big flow arrow, top left, like the old pop-up window
    const fx = Math.cos(fa), fy = Math.sin(fa), cx = 64, cy = 64;
    c.lineWidth = 9; c.strokeStyle = "rgba(0,0,0,.7)"; arrow(c, cx - fx * 36, cy - fy * 36, cx + fx * 36, cy + fy * 36);
    c.lineWidth = 4; c.strokeStyle = "#22d3ee"; arrow(c, cx - fx * 36, cy - fy * 36, cx + fx * 36, cy + fy * 36);
    c.font = "700 13px system-ui, sans-serif"; c.textAlign = "center"; c.fillStyle = "#22d3ee"; c.strokeStyle = "rgba(0,0,0,.7)"; c.lineWidth = 3;
    c.strokeText(RV.flip ? "flow (flipped)" : "flow", cx, 118); c.fillText(RV.flip ? "flow (flipped)" : "flow", cx, 118);
  } else {
    const draw = (t, col, ring) => { seg(t.root[0], t.root[1], t.tip[0], t.tip[1], col); dot(t.root[0], t.root[1], col, ring); };
    RV.tufts.forEach(t => { const gone = RV.remove.has(t.id); draw(t, gone ? RED : t.manual ? CYAN : GREEN, gone); });
    RV.add.forEach(a => draw(Array.isArray(a) ? rvTypical(a) : a, CYAN, false));
  }
  if (RV.drag && RV.drag.box) { const [a, b] = RV.drag.box; c.fillStyle = "rgba(239,68,68,.16)"; c.strokeStyle = RED; c.lineWidth = 1.5; c.setLineDash([6, 4]); c.fillRect(a[0], a[1], b[0] - a[0], b[1] - a[1]); c.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]); c.setLineDash([]); }
  if (RV.drag && RV.drag.line) { const [a, b] = RV.drag.line; c.strokeStyle = CYAN; c.lineWidth = 2; c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke(); }
}
function arrow(c, x1, y1, x2, y2) { const a = Math.atan2(y2 - y1, x2 - x1), h = 13; c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.moveTo(x2, y2); c.lineTo(x2 - h * Math.cos(a - 0.5), y2 - h * Math.sin(a - 0.5)); c.moveTo(x2, y2); c.lineTo(x2 - h * Math.cos(a + 0.5), y2 - h * Math.sin(a + 0.5)); c.stroke(); }
/** The line a click on a picture would add: the typical length and direction of the detected tufts (the server uses the same rule). */
function rvTypical(p) {
  const T = RV.tufts.filter(t => !t.manual); if (!T.length) return { root: p, tip: [p[0] + 40, p[1]] };
  const L = T.map(t => Math.hypot(t.tip[0] - t.root[0], t.tip[1] - t.root[1])).sort((a, b) => a - b), med = L[L.length >> 1];
  const vx = T.reduce((s, t) => s + t.tip[0] - t.root[0], 0), vy = T.reduce((s, t) => s + t.tip[1] - t.root[1], 0), n = Math.hypot(vx, vy) || 1;
  return { root: p, tip: [p[0] + vx / n * med, p[1] + vy / n * med] };
}

/* mouse / touch */
function rvPt(e) { const r = rvCv.getBoundingClientRect(), v = rvView(), px = e.clientX - r.left, py = e.clientY - r.top; return { px, py, x: (px - v.ox) / v.s, y: (py - v.oy) / v.s, s: v.s }; }
function distSeg(px, py, a, b) { const dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy || 1, t = Math.max(0, Math.min(1, ((px - a[0]) * dx + (py - a[1]) * dy) / l2)); return Math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy)); }
/** The nearest existing tuft within `reach` screen pixels: {kind: "item"|"added", id|index}. */
function rvNearest(p, reach) {
  const r = reach / p.s; let best = null, bd = r;
  rvItems().forEach(it => {
    const d = RV.kind === "map" ? Math.hypot(it.x - p.x, it.y - p.y) : Math.min(Math.hypot(it.root[0] - p.x, it.root[1] - p.y), distSeg(p.x, p.y, it.root, it.tip));
    if (d < bd) { bd = d; best = { kind: "item", id: it.id }; }
  });
  RV.add.forEach((a, i) => {
    const d = RV.kind === "map" ? Math.hypot(a[0] - p.x, a[1] - p.y) : (() => { const t = Array.isArray(a) ? rvTypical(a) : a; return Math.min(Math.hypot(t.root[0] - p.x, t.root[1] - p.y), distSeg(p.x, p.y, t.root, t.tip)); })();
    if (d < bd) { bd = d; best = { kind: "added", index: i }; }
  });
  return best;
}
rvCv.addEventListener("pointerdown", e => {
  if (!RV.map && !RV.tufts) return;
  rvCv.setPointerCapture(e.pointerId); const p = rvPt(e), right = e.button === 2 || e.button === 1;
  const v = rvView();
  if (right || (RV.mode === "add" && RV.kind === "map")) RV.drag = { pan: true, x0: e.clientX, y0: e.clientY, cx: RV.cx, cy: RV.cy, moved: false, down: p, s: v.s, right };
  else if (RV.mode === "remove") RV.drag = { box: [[p.px, p.py], [p.px, p.py]], down: p, moved: false };
  else RV.drag = { line: [[p.px, p.py], [p.px, p.py]], down: p, moved: false };
  if (right) rvStage.classList.add("pan");
});
rvCv.addEventListener("pointermove", e => {
  const d = RV.drag; if (!d) return; const p = rvPt(e);
  if (Math.hypot(p.px - d.down.px, p.py - d.down.py) > 5) d.moved = true;
  if (d.pan && d.moved) { RV.cx = d.cx - (e.clientX - d.x0) / d.s; RV.cy = d.cy - (e.clientY - d.y0) / d.s; rvClamp(); rvStage.classList.add("pan"); }
  else if (d.box) d.box[1] = [p.px, p.py];
  else if (d.line) d.line[1] = [p.px, p.py];
  rvDraw();
});
rvCv.addEventListener("pointerup", e => {
  const d = RV.drag; RV.drag = null; rvStage.classList.remove("pan"); if (!d) return;
  const p = rvPt(e), v = rvView();
  if (d.pan) { if (!d.moved && !d.right) rvClick(p); rvDraw(); return; }
  if (d.box && d.moved) {                                      // remove everything inside the box
    const [a, b] = d.box, x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]), y0 = Math.min(a[1], b[1]), y1 = Math.max(a[1], b[1]);
    const inside = (x, y) => { const X = v.ox + x * v.s, Y = v.oy + y * v.s; return X >= x0 && X <= x1 && Y >= y0 && Y <= y1; };
    const ids = rvItems().filter(it => !RV.remove.has(it.id) && (RV.kind === "map" ? inside(it.x, it.y) : inside(it.root[0], it.root[1]))).map(it => it.id);
    const keep = RV.add.filter(a => { const q = RV.kind === "map" ? a : (Array.isArray(a) ? a : a.root); return !inside(q[0], q[1]); });
    const dropped = RV.add.length - keep.length;
    if (ids.length || dropped) { rvSnap(); ids.forEach(i => RV.remove.add(i)); RV.add = keep; rvUpdate(); toast(`${ids.length + dropped} tuft${ids.length + dropped === 1 ? "" : "s"} removed.`); }
    else rvDraw();
    return;
  }
  if (d.line && d.moved && Math.hypot(p.px - d.down.px, p.py - d.down.py) > 12) {   // picture: drag = root to tip
    rvSnap(); RV.add.push({ root: [d.down.x, d.down.y], tip: [p.x, p.y] }); rvUpdate(); return;
  }
  rvClick(p);
});
function rvClick(p) {
  if (p.x < 0 || p.y < 0 || p.x > RV.W || p.y > RV.H) { rvDraw(); return; }
  const near = rvNearest(p, 12);
  if (RV.mode === "remove") {
    if (!near) { rvDraw(); return; }
    rvSnap(); if (near.kind === "added") RV.add.splice(near.index, 1); else RV.remove.has(near.id) ? RV.remove.delete(near.id) : RV.remove.add(near.id);
    rvUpdate(); return;
  }
  if (near && near.kind === "item" && RV.remove.has(near.id)) { rvSnap(); RV.remove.delete(near.id); rvUpdate(); return; }    // add mode on a removed tuft brings it back
  if (near) { toast("There is already a tuft here."); return; }
  rvSnap(); RV.add.push([p.x, p.y]); rvUpdate();
}
addEventListener("keydown", e => {
  if (!rvDlg.open || e.target.matches("input, select, textarea")) return;
  const k = e.key.toLowerCase();
  if (k === "a") setRvMode("add"); else if (k === "r") setRvMode("remove"); else if (k === "z") rvUndo(); else if (k === "+" || k === "=") $("#rvIn").click(); else if (k === "-") $("#rvOut").click(); else return;
  e.preventDefault();
});
function rvClose() { if (RV.busy) return; RV.dismissed.add(RV.jid); rvDlg.close(); }
$("#rvClose").addEventListener("click", rvClose); $("#rvLater").addEventListener("click", rvClose);
rvDlg.addEventListener("cancel", e => { if (RV.busy) e.preventDefault(); else RV.dismissed.add(RV.jid); });
$("#rvGo").addEventListener("click", async () => {
  if (RV.busy) return;
  if (rvCount() < 1 && !confirm("There are no tufts left. Process anyway?")) return;
  RV.busy = true; $("#rvGo").disabled = $("#rvLater").disabled = true; setMsg("#rvMsg", "Saving your changes…");
  try {
    if (RV.kind === "map") {
      if (RV.add.length || RV.remove.size || RV.flip) await api(`/jobs/${RV.jid}/map`, { method: "PUT", json: { add: RV.add, remove: [...RV.remove], flip: RV.flip } });
      const nm = $("#rvName").value.trim(); if (nm) { const r = await api(`/jobs/${RV.jid}/map/save`, { json: { name: nm } }); G.status.maps = r.saved; }
      await api(`/jobs/${RV.jid}/reanalyze`, { json: { map_mode: "keep", map_name: "", review: false, tracking: "map", step: Number($("#rvStep").value) } });
    } else {
      await api(`/jobs/${RV.jid}/image_tufts`, { method: "PUT", json: { remove: [...RV.remove], add: RV.add } });
      await api(`/jobs/${RV.jid}/reanalyze`, { json: { map_mode: "keep", review: false } });
    }
    RV.dismissed.delete(RV.jid); rvDlg.close(); G.loadedId = null; await pollOnce();
  } catch (e) { setMsg("#rvMsg", e.message, "err"); $("#rvGo").disabled = $("#rvLater").disabled = false; }
  RV.busy = false;
});
async function pollOnce() { const list = await api("/jobs"); G.jobs = list; renderRail(); openJob(G.cur, true, true); }

/* ---------------------------------------------------------------- start */
(async function init() {
  try { const s = await api("/status"); G.status = s; $("#engineNote").textContent = (un0(s) ? "Detector: Trained U-Net" : "Detector: Classical");
    const dp = $("#detPill"), un = !!s.unet; dp.hidden = false; dp.className = "det-pill" + (un ? " on" : "");
    dp.title = s.warning ? s.warning + " " + (s.fix || "") : "Trained model: " + (s.weights || "");
    dp.innerHTML = `<i></i>Detector: <b>${un ? "Trained U-Net" : "Classical"}</b>`; } catch (_) {}
  try { G.jobs = await api("/jobs"); } catch (_) {}
  renderRail(); route(); pollJobs();
})();
})();
