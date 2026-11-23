/* AgentML Studio — front-end. Plain JS, no build step, works offline. */
"use strict";

// ============================================================ utilities
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const icon = (name, style = "") => `<svg class="i" ${style ? `style="${style}"` : ""}><use href="#i-${name}"/></svg>`;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
const store = {
  get(k, d) { try { const v = localStorage.getItem("agentml." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("agentml." + k, JSON.stringify(v)); } catch { /* storage full or blocked */ } },
};
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

async function api(path, opts = {}) {
  const isForm = opts.body instanceof FormData;
  const r = await fetch(path, { ...opts, headers: opts.body && !isForm ? { "Content-Type": "application/json" } : {} });
  if (!r.ok) {
    let detail = "";
    try { const j = await r.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* not json */ }
    throw new Error(detail || `The server answered ${r.status}. Check the VS Code output for details.`);
  }
  return (r.headers.get("content-type") || "").includes("json") ? r.json() : r.text();
}
function toast(msg, err = false) {
  const t = document.createElement("div");
  t.className = "toast" + (err ? " err" : "");
  t.textContent = msg;
  $("toasts").append(t);
  setTimeout(() => t.remove(), err ? 6000 : 2800);
}
const fmt = (v, d = 3) => {
  if (typeof v !== "number" || !isFinite(v)) return esc(v ?? "—");
  if (Math.abs(v) >= 1e5) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (Math.abs(v) >= 100) return v.toLocaleString(undefined, { maximumFractionDigits: 1 });
  return v.toFixed(d);
};
const ago = (ts) => {
  const s = (Date.now() - ts) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return Math.floor(s / 60) + " min ago";
  if (s < 86400) return Math.floor(s / 3600) + " h ago";
  return new Date(ts).toLocaleDateString();
};
const extOf = (name) => (name.split(".").pop() || "").toLowerCase();
const DATASET_EXT = ["csv", "xlsx", "parquet"];

// ============================================================ markdown
function inline(t) {
  return esc(t)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?=[^*\w]|$)/g, "$1<em>$2</em>")
    .replace(/(^|\s)_([^_\n]+)_(?=\s|$|[.,!?;:])/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}
function md(src) {
  const lines = String(src || "").replace(/\r/g, "").split("\n");
  let out = "", i = 0;
  const isBlockStart = (l) => /^(#{1,4}\s|```|\s*[-*•]\s|\s*\d+[.)]\s|>|\|)/.test(l) || /^(-{3,}|\*{3,})\s*$/.test(l);
  while (i < lines.length) {
    const l = lines[i];
    if (l.startsWith("```")) {
      const lang = l.slice(3).trim(); const code = []; i++;
      while (i < lines.length && !lines[i].startsWith("```")) code.push(lines[i++]);
      i++;
      out += `<div class="codeblock"><div class="bar"><span>${esc(lang || "text")}</span><button class="iconbtn" data-copy title="Copy code">${icon("copy", "width:15px;height:15px")}</button></div><pre><code>${esc(code.join("\n"))}</code></pre></div>`;
      continue;
    }
    const h = l.match(/^(#{1,4})\s+(.*)/);
    if (h) { const n = h[1].length; out += `<h${n}>${inline(h[2])}</h${n}>`; i++; continue; }
    if (/^(-{3,}|\*{3,})\s*$/.test(l)) { out += "<hr>"; i++; continue; }
    if (l.trim().startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(lines[i++]);
      const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const body = rows.filter((r, k) => !(k === 1 && /^[\s|:-]+$/.test(r)));
      out += "<table>" + body.map((r, k) => "<tr>" + cells(r).map((c) => (k === 0 ? `<th>${inline(c)}</th>` : `<td>${inline(c)}</td>`)).join("") + "</tr>").join("") + "</table>";
      continue;
    }
    if (l.startsWith(">")) {
      const q = [];
      while (i < lines.length && lines[i].startsWith(">")) q.push(lines[i++].replace(/^>\s?/, ""));
      out += `<blockquote>${inline(q.join(" "))}</blockquote>`;
      continue;
    }
    const ul = /^\s*[-*•]\s+/, ol = /^\s*\d+[.)]\s+/;
    if (ul.test(l) || ol.test(l)) {
      const ordered = ol.test(l), re = ordered ? ol : ul, items = [];
      while (i < lines.length && re.test(lines[i])) {
        let item = lines[i++].replace(re, "");
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !ul.test(lines[i]) && !ol.test(lines[i])) item += " " + lines[i++].trim();
        items.push(item);
      }
      out += `<${ordered ? "ol" : "ul"}>${items.map((x) => `<li>${inline(x)}</li>`).join("")}</${ordered ? "ol" : "ul"}>`;
      continue;
    }
    if (!l.trim()) { i++; continue; }
    const para = [];
    while (i < lines.length && lines[i].trim() && !isBlockStart(lines[i])) para.push(lines[i++]);
    if (!para.length) para.push(lines[i++]);
    out += `<p>${inline(para.join(" "))}</p>`;
  }
  return out;
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]");
  if (!b) return;
  navigator.clipboard?.writeText(b.closest(".codeblock").querySelector("code").innerText).then(() => toast("Code copied"));
});

// ============================================================ state
const S = {
  chats: store.get("chats", []),
  chatId: null,
  docs: [],
  datasets: [],
  health: null,
  mode: store.get("mode", "auto"),
  scope: new Set(),
  attachments: [], // {id, name, kind:"doc"|"dataset", status:"uploading"|"ready"|"error", info}
  busy: false,
  skipTyping: false,
};
const currentChat = () => S.chats.find((c) => c.id === S.chatId);
const saveChats = () => store.set("chats", S.chats.slice(0, 60));

// ============================================================ navigation + sidebar
function show(view) {
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("on", v.id === "v-" + view));
  document.querySelectorAll(".nav button").forEach((b) => b.classList.toggle("on", b.dataset.view === view));
  $("app").classList.remove("mobile-open");
  if (view === "docs") loadDocs();
  if (view === "studio") initStudio();
  if (view === "settings") loadSettings();
  if (view === "chat") setTimeout(() => $("input").focus(), 30);
}
document.querySelectorAll(".nav button").forEach((b) => (b.onclick = () => show(b.dataset.view)));
$("collapseBtn").onclick = () => { if (innerWidth <= 900) $("app").classList.remove("mobile-open"); else $("app").classList.add("collapsed"); };
$("openSide").onclick = () => { if (innerWidth <= 900) $("app").classList.add("mobile-open"); else $("app").classList.remove("collapsed"); };
$("modelPill").onclick = () => show("settings");

function renderChatList() {
  $("chatList").innerHTML = S.chats.length
    ? S.chats.map((c) => `<div class="chatlink ${c.id === S.chatId ? "on" : ""}" data-id="${c.id}" title="${esc(c.title)}"><span>${esc(c.title)}</span><button class="iconbtn" data-del="${c.id}" aria-label="Delete chat">${icon("trash", "width:15px;height:15px")}</button></div>`).join("")
    : '<div class="chatlink" style="cursor:default"><span>Your chats will appear here</span></div>';
}
$("chatList").addEventListener("click", (e) => {
  const del = e.target.closest("[data-del]");
  if (del) {
    S.chats = S.chats.filter((c) => c.id !== del.dataset.del);
    saveChats();
    if (S.chatId === del.dataset.del) newChat(); else renderChatList();
    return;
  }
  const link = e.target.closest(".chatlink[data-id]");
  if (link) openChat(link.dataset.id);
});
function newChat() {
  S.chatId = null;
  S.scope.clear();
  renderScope();
  renderThread();
  renderChatList();
  show("chat");
}
function openChat(id) {
  S.chatId = id;
  const c = currentChat();
  S.scope = new Set((c && c.scope) || []);
  renderScope();
  renderThread();
  renderChatList();
  show("chat");
}
function ensureChat(title) {
  let c = currentChat();
  if (!c) {
    c = { id: uid(), title: title.slice(0, 60) || "New chat", created: Date.now(), messages: [], scope: [...S.scope] };
    S.chats.unshift(c);
    S.chatId = c.id;
  }
  return c;
}
$("newChat").onclick = newChat;
$("newChat2").onclick = newChat;

// ============================================================ theme
function applyTheme(t) {
  store.set("theme", t);
  const real = t === "system" ? (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark") : t;
  document.documentElement.dataset.theme = real;
  $("themeBtn").innerHTML = icon(real === "dark" ? "sun" : "moon");
  document.querySelectorAll("#themeSeg button").forEach((b) => b.classList.toggle("on", b.dataset.t === t));
}
$("themeBtn").onclick = () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
document.querySelectorAll("#themeSeg button").forEach((b) => (b.onclick = () => applyTheme(b.dataset.t)));

// ============================================================ model status
async function loadHealth() {
  try {
    S.health = await api("/api/health");
    const off = S.health.provider === "mock";
    $("modelPill").classList.toggle("offline", off);
    $("modelName").textContent = off ? "Offline mode" : S.health.model;
    $("modelSub").textContent = off ? "Add an API key for smarter answers" : `via ${S.health.provider}`;
  } catch {
    $("modelPill").classList.add("offline");
    $("modelName").textContent = "Server not reachable";
    $("modelSub").textContent = "Start it from VS Code";
  }
}

// ============================================================ chat rendering
function greeting() {
  const h = new Date().getHours();
  const part = h < 5 ? "Working late" : h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening";
  const name = store.get("name", "");
  return name ? `${part}, ${name}` : part;
}
function starters() {
  const pdf = S.docs.find((d) => d.kind === "pdf") || S.docs[0];
  const ds = S.datasets[0];
  const list = [];
  if (pdf) {
    list.push({ t: "Summarise a document", s: pdf.name, run: () => summarizeDoc(pdf.name) });
    list.push({ t: "Quiz me for my exam", s: `5 questions from ${pdf.name}`, run: () => quizDoc(pdf.name) });
    list.push({ t: "Explain the main idea", s: "in simple words, with page references", q: "Explain the main idea of the document in simple words" });
  } else {
    list.push({ t: "Chat with a PDF", s: "Attach a paper or your notes", run: () => $("fileIn").click() });
  }
  list.push({ t: "Create an image", s: "a solar microgrid village at sunset", q: "Generate an image of a solar-powered microgrid village at sunset, detailed digital art", mode: "image" });
  if (ds) list.push({ t: "Train a model", s: `on ${ds.name}`, q: `Train a model on ${ds.name}`, mode: "copilot" });
  list.push({ t: "Profile a dataset", s: ds ? ds.name : "Upload a CSV first", q: ds ? `Profile the dataset ${ds.name}` : null, mode: "copilot", run: ds ? null : () => $("fileIn").click() });
  return list.slice(0, 4);
}
function renderThread() {
  const c = currentChat();
  const th = $("thread");
  if (!c || !c.messages.length) {
    const st = starters();
    th.innerHTML = `<div class="hello"><h1>${esc(greeting())}</h1><p>Ask about your documents, train a model on a dataset, or just ask a question.</p>
      <div class="starters">${st.map((x, k) => `<button class="starter" data-k="${k}"><b>${esc(x.t)}</b><small>${esc(x.s)}</small></button>`).join("")}</div></div>`;
    th.querySelectorAll(".starter").forEach((b) => (b.onclick = () => {
      const x = st[+b.dataset.k];
      if (x.run) return x.run();
      if (x.q) send(x.q, x.mode);
    }));
    $("scroller").scrollTop = 0;
    return;
  }
  th.innerHTML = '<div class="msgs" id="msgs"></div>';
  c.messages.forEach((m, idx) => appendMessage(m, idx, false));
  scrollDown(true);
}
function scrollDown(force) {
  const s = $("scroller");
  if (force || s.scrollHeight - s.scrollTop - s.clientHeight < 240) s.scrollTop = s.scrollHeight;
}
function appendMessage(m, idx) {
  const box = $("msgs");
  const el = document.createElement("div");
  if (m.role === "user") {
    el.className = "m-user";
    el.innerHTML = (m.files && m.files.length ? `<span class="att">${icon("clip", "width:13px;height:13px;vertical-align:-2px")} ${m.files.map(esc).join(", ")}</span>` : "") + esc(m.text);
  } else {
    el.className = "m-bot";
    el.innerHTML = `<div class="av">${icon("spark")}</div><div class="content"></div>`;
    renderBot(el.querySelector(".content"), m, idx);
  }
  box.append(el);
  return el;
}

// ---------- thinking trail
function friendlySteps(m) {
  const d = m.data || {};
  if (m.kind === "rag") {
    return (d.trace || []).map((t) => {
      if (t.step === "contextualize") return `Read the follow-up as <b>“${esc(t.query)}”</b>`;
      if (t.step === "plan") return `Planned ${t.sub_queries.length > 1 ? t.sub_queries.length + " searches" : "the search"}${t.scope && t.scope !== "all" ? ` in <b>${esc(t.scope.join(", "))}</b>` : ""}: ${t.sub_queries.map((q) => `<code>${esc(q)}</code>`).join(" ")}`;
      if (t.step === "retrieve") return `Searched and found ${t.hits} passages, kept <b>${t.relevant}</b> that answer the question`;
      if (t.step === "rewrite") return `Results were weak, so tried again with <code>${esc(t.query)}</code>`;
      if (t.step === "reflect") return `Fact-checked the draft against the sources: <b>${Math.round(t.score * 100)}%</b> supported`;
      if (t.step === "regenerate") return `Rewrote the answer more strictly: <b>${Math.round(t.score * 100)}%</b> supported`;
      return esc(JSON.stringify(t));
    });
  }
  if (m.kind === "copilot") {
    const out = [];
    (d.steps || []).forEach((s) => {
      if (s.kind === "tool_call") out.push(`Used <b>${esc(s.tool)}</b> <code>${esc(JSON.stringify(s.content)).slice(0, 140)}</code>`);
      if (s.kind === "tool_result") out.push(`Got the result in ${(s.seconds || 0).toFixed(2)}s`);
    });
    return out.length ? out : ["Answered directly, no tools needed"];
  }
  return [];
}
function thinkBlock(steps, seconds, open = false) {
  if (!steps.length) return "";
  const label = `Thought for ${seconds < 1 ? "under a second" : seconds.toFixed(1) + "s"}, ${steps.length} step${steps.length > 1 ? "s" : ""}`;
  return `<details class="think" ${open ? "open" : ""}><summary><svg class="i chev"><use href="#i-right"/></svg><span>${label}</span></summary>
    <div class="trail">${steps.map((s) => `<div class="step done">${s}</div>`).join("")}</div></details>`;
}
const LIVE_STEPS = {
  rag: ["Understanding the question", "Searching your documents", "Grading the passages it found", "Writing an answer with citations", "Fact-checking the answer against the pages"],
  copilot: ["Reading the request", "Choosing a tool", "Running the tool", "Writing up the result"],
  summary: ["Reading the whole document", "Picking the key points", "Writing the summary"],
  quiz: ["Reading the document", "Choosing what to test", "Writing questions and distractors"],
  image: ["Reading your description", "Composing the scene", "Painting the image, this takes 10 to 40 seconds", "Adding the final details"],
};
function liveTrail(container, kind) {
  const steps = LIVE_STEPS[kind] || LIVE_STEPS.rag;
  container.innerHTML = `<div class="think"><div class="trail" style="margin-top:2px"></div></div>` + (kind === "image" ? '<div class="imgph"></div>' : "");
  const trail = container.querySelector(".trail");
  let k = 0, stopped = false;
  const tick = () => {
    if (stopped) return;
    trail.querySelectorAll(".step.live").forEach((s) => { s.classList.remove("live"); s.classList.add("done"); s.firstChild.className = ""; });
    if (k < steps.length) {
      const d = document.createElement("div");
      d.className = "step live";
      d.innerHTML = `<span class="live-label">${esc(steps[k++])}</span>`;
      trail.append(d);
      scrollDown();
    }
    setTimeout(tick, 650 + Math.random() * 450);
  };
  tick();
  return () => { stopped = true; };
}

// ---------- bot message renderers
function citeify(html, cites) {
  return html.replace(/\[(\d+)\]/g, (m, n) => {
    const c = cites && cites[n - 1];
    if (!c) return m;
    const tip = esc(`${c.source}${c.page ? ", page " + c.page : ""}`);
    const url = c.file ? `/api/rag/files/${encodeURIComponent(c.source)}${c.page ? "#page=" + c.page : ""}` : null;
    return url ? `<a href="${url}" target="_blank" title="${tip}"><sup class="cite">${n}</sup></a>` : `<sup class="cite" title="${tip}">${n}</sup>`;
  });
}
function sourcesHtml(cites) {
  if (!cites || !cites.length) return "";
  return `<div class="sources">${cites.map((c) => {
    const url = c.file ? `/api/rag/files/${encodeURIComponent(c.source)}${c.page ? "#page=" + c.page : ""}` : null;
    const inner = `<span class="n">${c.id}</span><span class="f">${esc(c.source)}</span>${c.page ? `<span class="p">p. ${c.page}</span>` : ""}`;
    return url ? `<a class="src" href="${url}" target="_blank" title="${esc(c.preview)}">${inner}</a>` : `<span class="src" title="${esc(c.preview)}">${inner}</span>`;
  }).join("")}</div>`;
}
function figsHtml(images) {
  if (!images || !images.length) return "";
  return `<div class="figs">${images.map((im) => `<figure class="fig" data-zoom="/api/rag/assets/${encodeURI(im.path)}"><img loading="lazy" alt="Figure from ${esc(im.source)} page ${im.page}" src="/api/rag/assets/${encodeURI(im.path)}"><figcaption>Figure, ${esc(im.source)} p. ${im.page}</figcaption></figure>`).join("")}</div>`;
}
document.addEventListener("click", (e) => {
  const f = e.target.closest("[data-zoom]");
  if (!f) return;
  const lb = document.createElement("div");
  lb.className = "lightbox";
  lb.innerHTML = `<img src="${f.dataset.zoom}" alt="">`;
  lb.onclick = () => lb.remove();
  document.body.append(lb);
});
function actionsHtml(m, idx, extra = "") {
  return `<div class="meta-row">
    <button class="iconbtn" data-act="copy" data-idx="${idx}" title="Copy">${icon("copy")}</button>
    ${m.retry ? `<button class="iconbtn" data-act="retry" data-idx="${idx}" title="Ask again">${icon("retry")}</button>` : ""}
    ${extra}</div>`;
}
function renderBot(el, m, idx, animateText = false) {
  const d = m.data || {};
  if (m.kind === "error") {
    el.innerHTML = `<div class="errbox">${esc(m.text)}${m.needsKey ? `<div style="margin-top:10px"><button class="btn sm" data-act="settings">${icon("gear")}Open Settings</button></div>` : ""}</div>${actionsHtml(m, idx)}`;
    return;
  }
  if (m.kind === "image") {
    const sizeName = { "1024x1024": "Square", "1536x1024": "Landscape", "1024x1536": "Portrait" }[d.size] || d.size;
    el.innerHTML = `<figure class="genimg"><img src="${d.url}" alt="${esc(d.subject || m.text)}" data-zoom="${d.url}"></figure>
      <details class="think" style="margin-top:8px"><summary><svg class="i chev"><use href="#i-right"/></svg><span>Prompt sent to the image model</span></summary><div class="note" style="margin:6px 0 0 4px;font-size:13.5px">${esc(d.revised_prompt || d.subject || m.text)}</div></details>
      <div class="meta-row">
        <a class="iconbtn" href="${d.url}" download="agentml-${esc(d.file)}" title="Download">${icon("download")}</a>
        <button class="iconbtn" data-act="copy" data-idx="${idx}" title="Copy the prompt">${icon("copy")}</button>
        <button class="iconbtn" data-act="retry" data-idx="${idx}" title="Make another version">${icon("retry")}</button>
        <span class="badge">${esc(sizeName)}, ${esc(d.model)}, ${fmt(d.seconds, 1)}s</span>
      </div>`;
    const img = el.querySelector("img");
    img.addEventListener("load", () => scrollDown(true));
    return;
  }
  if (m.kind === "rag" || m.kind === "copilot" || m.kind === "summary" || m.kind === "text") {
    const steps = friendlySteps(m);
    const secs = d.seconds || 0;
    const cites = m.kind === "rag" ? d.citations : null;
    let extra = "";
    if (m.kind === "rag" && d.confidence) extra = `<span class="badge ${d.confidence}"><span class="d"></span>${d.confidence} confidence, ${Math.round((d.groundedness || 0) * 100)}% grounded</span>`;
    if (m.kind === "summary") extra = `<span class="badge">Summary of ${esc(d.name || "")}</span>`;
    el.innerHTML = thinkBlock(steps, secs) + `<div class="prose"></div>` + `<div class="tail hidden">${sourcesHtml(cites)}${figsHtml(m.kind === "rag" ? d.images : null)}${actionsHtml(m, idx, extra)}</div>`;
    const prose = el.querySelector(".prose");
    const finish = () => { prose.innerHTML = citeify(md(m.text), cites); prose.classList.remove("caret"); el.querySelector(".tail").classList.remove("hidden"); scrollDown(animateText); };
    if (animateText && !reduceMotion) typewriter(prose, m.text, cites).then(finish); else finish();
    return;
  }
  if (m.kind === "quiz") { renderQuiz(el, m, idx); return; }
  if (m.kind === "dataset") {
    el.innerHTML = `<div class="card-inline"><h3>${icon("table", "width:16px;height:16px;vertical-align:-3px")} ${esc(d.name)} is ready</h3>
      <div class="note" style="margin-bottom:12px">${esc(d.size_kb)} KB uploaded. I can profile it or hand it to the AutoML agents.</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap"><button class="btn sm" data-act="studio" data-name="${esc(d.name)}">${icon("flask")}Train models in Studio</button>
      <button class="btn ghost sm" data-act="profile" data-name="${esc(d.name)}">Profile the data</button></div></div>`;
    return;
  }
  if (m.kind === "docs") {
    el.innerHTML = `<div class="card-inline"><h3>Added to your documents</h3><div class="note">${(d.files || []).map((f) =>
      `<div>${icon("file", "width:15px;height:15px;vertical-align:-3px")} <b>${esc(f.name)}</b>: ${[f.pages ? f.pages + " pages" : null, f.chunks + " passages", f.images ? f.images + " figures" : null].filter(Boolean).join(", ")}</div>`).join("")}
      <div style="margin-top:8px">Ask a question below. I'll answer only from ${d.files && d.files.length > 1 ? "these files" : "this file"} and cite the pages.</div></div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:12px">${(d.files || []).slice(0, 1).map((f) => `<button class="btn sm" data-act="summary" data-name="${esc(f.name)}">${icon("book")}Summarise</button><button class="btn ghost sm" data-act="quiz" data-name="${esc(f.name)}">${icon("target")}Quiz me</button>`).join("")}</div></div>`;
  }
}
async function typewriter(el, text, cites) {
  el.classList.add("caret");
  const words = text.split(/(\s+)/);
  const total = words.length;
  const perTick = Math.max(2, Math.ceil(total / 70)); // ~1.2s for any length
  S.skipTyping = false;
  setSendState("typing");
  for (let k = perTick; k < total + perTick; k += perTick) {
    if (S.skipTyping) break;
    el.innerHTML = citeify(md(words.slice(0, k).join("")), cites);
    scrollDown();
    await sleep(16);
  }
  setSendState();
}
$("thread").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-act]");
  if (!b) return;
  const c = currentChat();
  const act = b.dataset.act;
  if (act === "copy") {
    const m = c.messages[+b.dataset.idx];
    navigator.clipboard?.writeText(m.text || "").then(() => toast("Copied"));
  } else if (act === "retry") {
    const idx = +b.dataset.idx;
    const userIdx = c.messages.slice(0, idx).map((m) => m.role).lastIndexOf("user");
    if (userIdx < 0) return;
    const u = c.messages[userIdx];
    c.messages = c.messages.slice(0, userIdx);
    saveChats();
    renderThread();
    send(u.text, u.mode, true);
  } else if (act === "settings") {
    show("settings");
  } else if (act === "studio") {
    show("studio");
    await initStudio(b.dataset.name);
  } else if (act === "profile") {
    send(`Profile the dataset ${b.dataset.name}`, "copilot");
  } else if (act === "summary") summarizeDoc(b.dataset.name);
  else if (act === "quiz") quizDoc(b.dataset.name);
});

// ---------- quiz
function renderQuiz(el, m, idx) {
  const d = m.data;
  d.picked = d.picked || {};
  const checked = d.checked;
  const score = d.questions.reduce((s, q, k) => s + (d.picked[k] === q.answer ? 1 : 0), 0);
  el.innerHTML = `<span class="badge">Quiz on ${esc(d.name)}</span>` + d.questions.map((q, k) => `
    <div class="quiz-q">${k + 1}. ${esc(q.question)}</div>
    ${q.options.map((o, j) => {
      let cls = "opt";
      if (checked) { if (j === q.answer) cls += " ok"; else if (d.picked[k] === j) cls += " no"; } else if (d.picked[k] === j) cls += " pick";
      return `<button class="${cls}" data-q="${k}" data-j="${j}" ${checked ? "disabled" : ""}><span class="k">${"ABCD"[j]}</span><span>${esc(o)}</span></button>`;
    }).join("")}
    ${checked && q.explanation ? `<div class="why">${esc(q.explanation)}</div>` : ""}`).join("") +
    (checked ? `<div class="score">You scored ${score} out of ${d.questions.length}</div><div class="meta-row"><button class="btn ghost sm" data-quiz="again">Try again</button><button class="btn ghost sm" data-quiz="new">New questions</button></div>`
      : `<div class="meta-row"><button class="btn sm" data-quiz="check">Check answers</button><span>${Object.keys(d.picked).length} of ${d.questions.length} answered</span></div>`);
  el.onclick = (e) => {
    const o = e.target.closest(".opt");
    const qb = e.target.closest("[data-quiz]");
    if (o && !checked) { d.picked[o.dataset.q] = +o.dataset.j; saveChats(); renderQuiz(el, m, idx); }
    if (qb) {
      if (qb.dataset.quiz === "check") d.checked = true;
      if (qb.dataset.quiz === "again") { d.checked = false; d.picked = {}; }
      if (qb.dataset.quiz === "new") return quizDoc(d.name);
      saveChats();
      renderQuiz(el, m, idx);
    }
  };
}

// ============================================================ sending
const IMAGE_RE = /\b(generate|create|draw|make|design|paint|render|sketch)\s+(me\s+)?(an?\s+|the\s+|some\s+|my\s+)?([\w-]+\s+){0,2}(image|picture|photo|illustration|drawing|logo|poster|wallpaper|artwork|icon|thumbnail|banner)s?\b(?!\s+(processing|classification|segmentation|recognition|dataset|data|analysis))|\b(image|photo|picture|tasveer|logo|poster|wallpaper)\s+(banao|bana do|banado|bnao|bana de|chahiye)\b|^\s*(an?\s+)?(image|picture|photo) of\b/i;
function pickMode(text, forced) {
  if (forced) return forced;
  if (S.mode !== "auto") return S.mode;
  if (IMAGE_RE.test(text)) return "image";
  const toolish = /\b(train|dataset|csv|xlsx|profile|predict|experiments?|calculate|calc|leaderboard|automl)\b|\.(csv|xlsx|parquet)\b|\d\s*[-+*/^]\s*\d/i.test(text);
  if (toolish) return "copilot";
  return S.scope.size || S.docs.length ? "docs" : "copilot";
}
function chatHistory(c) {
  return c.messages.filter((m) => m.role === "user" || m.kind === "rag" || m.kind === "copilot")
    .map((m) => ({ role: m.role === "user" ? "user" : "assistant", content: m.text || "" })).slice(-6);
}
async function send(text, forcedMode, isRetry = false) {
  text = (text || "").trim();
  const ready = S.attachments.filter((a) => a.status === "ready");
  if ((!text && !ready.length) || S.busy) return;
  if (S.attachments.some((a) => a.status === "uploading")) return toast("Wait for the upload to finish");
  show("chat");
  const c = ensureChat(text || ready.map((a) => a.name).join(", "));
  if (!$("msgs")) $("thread").innerHTML = '<div class="msgs" id="msgs"></div>';
  const mode = text ? pickMode(text, forcedMode) : null;
  if (!isRetry) {
    const um = { role: "user", text, files: ready.map((a) => a.name), mode: forcedMode };
    c.messages.push(um);
    appendMessage(um, c.messages.length - 1);
  }
  // attachments turn into cards
  const docFiles = ready.filter((a) => a.kind === "doc").map((a) => a.info);
  const dsFiles = ready.filter((a) => a.kind === "dataset");
  S.attachments = [];
  renderAttachments();
  $("input").value = "";
  autosize();
  if (docFiles.length) {
    docFiles.forEach((f) => S.scope.add(f.name));
    c.scope = [...S.scope];
    renderScope();
    if (!text) pushBot(c, { role: "bot", kind: "docs", data: { files: docFiles } });
  }
  if (dsFiles.length && !text) dsFiles.forEach((a) => pushBot(c, { role: "bot", kind: "dataset", data: a.info }));
  saveChats();
  renderChatList();
  if (!text) return;

  S.busy = true;
  setSendState("busy");
  const holder = appendMessage({ role: "bot", kind: "pending" }, -1);
  const content = holder.querySelector(".content");
  const stop = liveTrail(content, mode === "docs" ? "rag" : mode === "image" ? "image" : "copilot");
  scrollDown(true);
  let m;
  try {
    if (mode === "image") {
      const r = await api("/api/images", { method: "POST", body: JSON.stringify({ prompt: text }) });
      m = { role: "bot", kind: "image", text, data: r, retry: true };
    } else if (mode === "docs") {
      const a = await api("/api/rag/query", { method: "POST", body: JSON.stringify({ question: text, history: chatHistory(c).slice(0, -1), sources: [...S.scope] }) });
      m = { role: "bot", kind: "rag", text: a.answer, data: a, retry: true };
    } else {
      const r = await api("/api/chat", { method: "POST", body: JSON.stringify({ message: text, session_id: c.id }) });
      m = { role: "bot", kind: "copilot", text: r.answer, data: { steps: r.steps, seconds: r.seconds }, retry: true };
    }
  } catch (err) {
    m = { role: "bot", kind: "error", text: err.message, retry: true, needsKey: /API key/i.test(err.message) };
  }
  stop();
  c.messages.push(m);
  saveChats();
  renderBot(content, m, c.messages.length - 1, true);
  S.busy = false;
  setSendState();
}
function pushBot(c, m) {
  c.messages.push(m);
  appendMessage(m, c.messages.length - 1);
  scrollDown(true);
}
async function summarizeDoc(name) {
  S.scope = new Set([name]);
  renderScope();
  await runDocTask(`Summarise ${name}`, "summary", async () => {
    const r = await api("/api/rag/summarize", { method: "POST", body: JSON.stringify({ name }) });
    return { role: "bot", kind: "summary", text: r.summary, data: { name, seconds: 0 } };
  });
}
async function quizDoc(name) {
  S.scope = new Set([name]);
  renderScope();
  await runDocTask(`Quiz me on ${name}`, "quiz", async () => {
    const r = await api("/api/rag/quiz", { method: "POST", body: JSON.stringify({ name, n: 5 }) });
    return { role: "bot", kind: "quiz", text: r.questions.map((q, k) => `${k + 1}. ${q.question}`).join("\n"), data: { name, questions: r.questions, picked: {} } };
  });
}
async function runDocTask(label, kind, fn) {
  if (S.busy) return;
  show("chat");
  const c = ensureChat(label);
  c.scope = [...S.scope];
  if (!$("msgs")) $("thread").innerHTML = '<div class="msgs" id="msgs"></div>';
  const um = { role: "user", text: label };
  c.messages.push(um);
  appendMessage(um, c.messages.length - 1);
  renderChatList();
  S.busy = true;
  setSendState("busy");
  const holder = appendMessage({ role: "bot", kind: "pending" }, -1);
  const content = holder.querySelector(".content");
  const stop = liveTrail(content, kind);
  scrollDown(true);
  let m;
  try { m = await fn(); } catch (err) { m = { role: "bot", kind: "error", text: err.message }; }
  stop();
  c.messages.push(m);
  saveChats();
  renderBot(content, m, c.messages.length - 1, true);
  S.busy = false;
  setSendState();
}

// ============================================================ composer
const autosize = () => { const t = $("input"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 220) + "px"; setSendState(); };
function setSendState(state) {
  const b = $("sendBtn");
  if (state === "typing") { b.disabled = false; b.classList.add("stop"); b.innerHTML = icon("stop"); b.title = "Show the full answer"; return; }
  b.classList.remove("stop");
  b.innerHTML = icon("up");
  b.title = "Send";
  const has = $("input").value.trim() || S.attachments.some((a) => a.status === "ready");
  b.disabled = state === "busy" || S.busy || !has;
}
$("input").addEventListener("input", autosize);
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send($("input").value); }
});
$("sendBtn").onclick = () => { if ($("sendBtn").classList.contains("stop")) { S.skipTyping = true; return; } send($("input").value); };
$("attachBtn").onclick = () => $("fileIn").click();
$("fileIn").onchange = (e) => { attachFiles(e.target.files); e.target.value = ""; };
$("composer").addEventListener("dragover", (e) => e.preventDefault());
$("composer").addEventListener("drop", (e) => { e.preventDefault(); attachFiles(e.dataTransfer.files); });

async function attachFiles(files) {
  for (const f of [...files]) {
    const kind = DATASET_EXT.includes(extOf(f.name)) ? "dataset" : "doc";
    const a = { id: uid(), name: f.name, kind, status: "uploading", info: null };
    S.attachments.push(a);
    renderAttachments();
    try {
      const fd = new FormData();
      if (kind === "dataset") {
        fd.append("file", f);
        a.info = await api("/api/datasets", { method: "POST", body: fd });
        a.name = a.info.name;
        loadDatasets();
      } else {
        fd.append("files", f);
        const r = await api("/api/rag/upload", { method: "POST", body: fd });
        if (!r.files.length) throw new Error(r.errors.join("; ") || "No text could be read from this file");
        a.info = r.files[0];
        a.name = a.info.name;
        loadDocs(false);
      }
      a.status = "ready";
    } catch (err) {
      a.status = "error";
      a.error = err.message;
      toast(`${f.name}: ${err.message}`, true);
    }
    renderAttachments();
  }
}
function renderAttachments() {
  $("attachments").innerHTML = S.attachments.map((a) => {
    const e = extOf(a.name);
    const cls = e === "pdf" ? "pdf" : DATASET_EXT.includes(e) ? "csv" : "";
    let sub = "Uploading…";
    if (a.status === "ready") sub = a.kind === "dataset" ? `Dataset, ${a.info.size_kb} KB` : [a.info.pages ? a.info.pages + " pages" : null, a.info.chunks + " passages", a.info.images ? a.info.images + " figures" : null].filter(Boolean).join(", ");
    if (a.status === "error") sub = a.error;
    return `<div class="attach"><span class="ic ${cls}">${a.status === "uploading" ? '<span class="spin"></span>' : icon(a.kind === "dataset" ? "table" : "file", "width:16px;height:16px")}</span>
      <span style="min-width:0"><span class="nm">${esc(a.name)}</span><small>${esc(sub)}</small></span>
      <button class="iconbtn" data-rm="${a.id}" aria-label="Remove attachment" style="width:26px;height:26px">${icon("x", "width:14px;height:14px")}</button></div>`;
  }).join("");
  setSendState();
}
$("attachments").addEventListener("click", (e) => {
  const b = e.target.closest("[data-rm]");
  if (!b) return;
  S.attachments = S.attachments.filter((a) => a.id !== b.dataset.rm);
  renderAttachments();
});

function renderScope() {
  $("scope").innerHTML = S.scope.size ? [...S.scope].map((n) => `<span class="chip">${icon("file", "width:13px;height:13px")} ${esc(n)}<button data-unscope="${esc(n)}" aria-label="Stop limiting to ${esc(n)}">${icon("x", "width:12px;height:12px")}</button></span>`).join("") : "";
}
$("scope").addEventListener("click", (e) => {
  const b = e.target.closest("[data-unscope]");
  if (!b) return;
  S.scope.delete(b.dataset.unscope);
  const c = currentChat();
  if (c) { c.scope = [...S.scope]; saveChats(); }
  renderScope();
});

// mode menu
const MODES = {
  auto: { label: "Auto", desc: "Picks documents or tools for each message" },
  docs: { label: "Documents", desc: "Answer only from your files, with page citations" },
  copilot: { label: "Copilot tools", desc: "Profile data, train models, calculate" },
  image: { label: "Create image", desc: "Generate a picture from a description (OpenAI)" },
};
function setMode(m) { S.mode = m; store.set("mode", m); $("modeLabel").textContent = MODES[m].label; }
$("modeBtn").onclick = (e) => {
  e.stopPropagation();
  closeMenus();
  const r = $("modeBtn").getBoundingClientRect();
  const menu = document.createElement("div");
  menu.className = "menu";
  menu.id = "modeMenu";
  menu.innerHTML = Object.entries(MODES).map(([k, v]) => `<button data-mode="${k}" class="${S.mode === k ? "sel" : ""}"><span><b>${v.label}</b><small>${v.desc}</small></span><span class="tick">${icon("check")}</span></button>`).join("");
  document.body.append(menu);
  menu.style.left = r.left + "px";
  menu.style.top = r.top - menu.offsetHeight - 8 + "px";
  menu.onclick = (ev) => { const b = ev.target.closest("[data-mode]"); if (b) { setMode(b.dataset.mode); closeMenus(); $("input").focus(); } };
};
function closeMenus() { document.querySelectorAll(".menu").forEach((m) => m.remove()); }
document.addEventListener("click", (e) => { if (!e.target.closest(".menu")) closeMenus(); });

// ============================================================ documents page
async function loadDocs(render = true) {
  try { S.docs = await api("/api/rag/docs"); } catch { S.docs = []; }
  $("docCount").textContent = S.docs.length || "";
  if (render) renderDocs();
}
function renderDocs() {
  const q = $("docSearch").value.trim().toLowerCase();
  const list = S.docs.filter((d) => !q || d.name.toLowerCase().includes(q));
  if (!S.docs.length) {
    $("docList").innerHTML = `<div class="empty">${icon("file", "width:34px;height:34px")}<h3>No documents yet</h3><div>Upload a research paper, lecture notes or a report. You can then ask questions, get a summary or take a quiz.</div></div>`;
    return;
  }
  $("docList").innerHTML = `<div class="doclist">${list.map((d) => {
    const facts = [d.pages ? `${d.pages} pages` : null, `${d.chunks} passages`, d.n_images ? `${d.n_images} figures` : null, d.added ? `added ${ago(d.added * 1000)}` : null].filter(Boolean);
    return `<div class="docrow"><span class="ic ${d.kind === "pdf" ? "pdf" : ""}">${icon("file")}</span>
      <div style="min-width:0"><div class="nm" title="${esc(d.name)}">${esc(d.name)}</div><div class="facts">${facts.map((f) => `<span>${esc(f)}</span>`).join("")}</div></div>
      <div class="acts">
        <button class="btn sm" data-doc="ask" data-name="${esc(d.name)}">${icon("chat")}Ask</button>
        <button class="btn ghost sm" data-doc="summary" data-name="${esc(d.name)}">Summarise</button>
        <button class="btn ghost sm" data-doc="quiz" data-name="${esc(d.name)}">Quiz</button>
        ${d.file ? `<a class="btn ghost sm" href="/api/rag/files/${encodeURIComponent(d.name)}" target="_blank" title="Open the file">${icon("open")}</a>` : ""}
        <button class="btn ghost sm danger" data-doc="delete" data-name="${esc(d.name)}" title="Remove from documents">${icon("trash")}</button>
      </div></div>`;
  }).join("") || '<div class="empty">No document matches that filter.</div>'}</div>`;
}
$("docSearch").addEventListener("input", renderDocs);
$("docList").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-doc]");
  if (!b) return;
  const name = b.dataset.name;
  if (b.dataset.doc === "ask") { newChat(); S.scope = new Set([name]); renderScope(); $("input").placeholder = `Ask about ${name}`; $("input").focus(); }
  if (b.dataset.doc === "summary") { newChat(); summarizeDoc(name); }
  if (b.dataset.doc === "quiz") { newChat(); quizDoc(name); }
  if (b.dataset.doc === "delete") {
    if (!confirm(`Remove ${name} from your documents? Chats keep their text but can no longer cite it.`)) return;
    await api(`/api/rag/docs/${encodeURIComponent(name)}`, { method: "DELETE" });
    toast(`Removed ${name}`);
    loadDocs();
  }
});
async function uploadDocs(files) {
  if (!files.length) return;
  $("dropzone").innerHTML = `<span class="spin"></span> Reading ${files.length} file${files.length > 1 ? "s" : ""}…`;
  try {
    const fd = new FormData();
    [...files].forEach((f) => fd.append("files", f));
    const r = await api("/api/rag/upload", { method: "POST", body: fd });
    toast(`Added ${r.documents} document${r.documents === 1 ? "" : "s"}${r.images ? `, ${r.images} figures` : ""}`);
    if (r.errors.length) toast(r.errors.join("; "), true);
  } catch (err) { toast(err.message, true); }
  $("dropzone").innerHTML = "<b>Drop PDFs, notes or web pages here</b><br>or click to choose files. PDF, Markdown, TXT, HTML.";
  loadDocs();
}
$("docUpload").onclick = () => $("docFileIn").click();
$("dropzone").onclick = () => $("docFileIn").click();
$("docFileIn").onchange = (e) => { uploadDocs(e.target.files); e.target.value = ""; };
["dragenter", "dragover"].forEach((ev) => $("dropzone").addEventListener(ev, (e) => { e.preventDefault(); $("dropzone").classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => $("dropzone").addEventListener(ev, (e) => { e.preventDefault(); $("dropzone").classList.remove("over"); }));
$("dropzone").addEventListener("drop", (e) => uploadDocs(e.dataTransfer.files));
$("pasteAdd").onclick = async () => {
  const text = $("pasteText").value.trim();
  if (!text) return toast("Paste some text first", true);
  const title = $("pasteTitle").value.trim() || "Pasted notes";
  try {
    await api("/api/rag/ingest", { method: "POST", body: JSON.stringify({ text, source: title }) });
    $("pasteText").value = ""; $("pasteTitle").value = "";
    toast(`Added ${title}`);
    loadDocs();
  } catch (err) { toast(err.message, true); }
};

// ============================================================ AutoML studio
const AGENTS = [
  { name: "Planner", em: "🧭", role: "frames the goal as an ML task" },
  { name: "Data Analyst", em: "🔍", role: "audits data quality" },
  { name: "Feature Engineer", em: "🧪", role: "decides how to encode columns" },
  { name: "Model Trainer", em: "🏋️", role: "benchmarks model families" },
  { name: "Critic", em: "🧐", role: "tunes the leaders, checks over-fitting" },
  { name: "Reporter", em: "📝", role: "writes the report" },
];
const ST = { budget: "medium", runId: null, polling: null, columns: [], inited: false, result: null };
async function loadDatasets() {
  try { S.datasets = await api("/api/datasets"); } catch { S.datasets = []; }
  return S.datasets;
}
async function initStudio(selectName) {
  await loadDatasets();
  const sel = $("dsSelect");
  const cur = selectName || sel.value || (S.datasets[0] && S.datasets[0].name);
  sel.innerHTML = S.datasets.map((d) => `<option ${d.name === cur ? "selected" : ""}>${esc(d.name)}</option>`).join("") || "<option value=''>Upload a dataset first</option>";
  if (!ST.inited) { renderCrew([]); ST.inited = true; }
  await loadPreview();
  loadRuns();
}
async function loadPreview() {
  const name = $("dsSelect").value;
  if (!name) { $("dsPreview").innerHTML = ""; return; }
  try {
    const p = await api(`/api/datasets/${encodeURIComponent(name)}/preview?rows=6`);
    ST.columns = p.columns;
    $("dsPreview").innerHTML = `<table><tr>${p.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${p.rows.map((r) => `<tr>${p.columns.map((c) => `<td>${esc(r[c] ?? "")}</td>`).join("")}</tr>`).join("")}</table>`;
    $("targetSel").innerHTML = '<option value="">Let the agents decide</option>' + p.columns.map((c) => `<option>${esc(c)}</option>`).join("");
    if (!$("goal").value.trim() || $("goal").dataset.auto === "1") {
      $("goal").value = name.includes("house") ? "Estimate the selling price of a house" : name.includes("churn") ? "Predict which customers will churn" : "";
      $("goal").dataset.auto = "1";
    }
  } catch (err) { $("dsPreview").innerHTML = `<div class="errbox">${esc(err.message)}</div>`; }
}
$("goal").addEventListener("input", () => ($("goal").dataset.auto = "0"));
$("dsSelect").onchange = loadPreview;
$("dsUpload").onclick = () => $("dsFileIn").click();
$("dsFileIn").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  try { const r = await api("/api/datasets", { method: "POST", body: fd }); toast(`Uploaded ${r.name}`); await initStudio(r.name); } catch (err) { toast(err.message, true); }
  e.target.value = "";
};
document.querySelectorAll("#budgetSeg button").forEach((b) => (b.onclick = () => {
  ST.budget = b.dataset.b;
  document.querySelectorAll("#budgetSeg button").forEach((x) => x.classList.toggle("on", x === b));
}));
function renderCrew(events, status) {
  const by = {};
  events.forEach((e) => { (by[e.agent] = by[e.agent] || []).push(e); });
  const doneAll = status === "completed";
  let lastActive = -1;
  AGENTS.forEach((a, k) => { if (by[a.name]) lastActive = k; });
  $("crew").innerHTML = AGENTS.map((a, k) => {
    const evs = (by[a.name] || []).filter((e) => e.kind === "log");
    let cls = "";
    if (doneAll || k < lastActive) cls = "done";
    else if (k === lastActive && status === "running") cls = "live";
    else if (k === lastActive) cls = "done";
    const logs = evs.slice(-4).map((e) => `<div>${esc(e.message)}</div>`).join("");
    return `<div class="agent ${cls}"><div class="av">${a.em}</div><div><div class="who">${a.name}<small>${a.role}</small></div><div class="logs">${logs}</div></div></div>`;
  }).join("");
  const st = $("runState");
  st.textContent = status || "idle";
  st.className = "badge " + (status === "completed" ? "high" : status === "failed" ? "low" : status ? "medium" : "");
}
$("runBtn").onclick = async () => {
  const dataset = $("dsSelect").value;
  if (!dataset) return toast("Upload a dataset first", true);
  $("runBtn").disabled = true;
  $("results").innerHTML = "";
  try {
    const r = await api("/api/automl/runs", { method: "POST", body: JSON.stringify({ dataset, goal: $("goal").value, target: $("targetSel").value || null, budget: ST.budget }) });
    watchRun(r.run_id);
  } catch (err) { toast(err.message, true); $("runBtn").disabled = false; }
};
async function watchRun(id) {
  ST.runId = id;
  let events = [];
  const tick = async () => {
    if (ST.runId !== id) return;
    let r;
    try { r = await api(`/api/automl/runs/${id}?since=${events.length}`); } catch { setTimeout(tick, 2000); return; }
    if (ST.runId !== id) return;
    events = events.concat(r.events);
    renderCrew(events, r.status);
    if (r.status === "completed" || r.status === "failed") {
      $("runBtn").disabled = false;
      loadRuns();
      if (r.status === "failed") $("results").innerHTML = `<div class="errbox">${esc(r.error)}</div>`;
      else showResults(r.result);
      return;
    }
    setTimeout(tick, 900);
  };
  tick();
}
async function loadRuns() {
  try {
    const runs = await api("/api/automl/runs");
    $("runList").innerHTML = runs.length ? runs.slice(0, 10).map((r) => `<button data-run="${esc(r.run_id)}"><span class="state ${r.status}"></span><span style="min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(r.goal || r.run_id)}</span><small>${esc(r.best_model || r.status)}</small></button>`).join("") : '<div class="note">No experiments yet.</div>';
  } catch { /* server offline */ }
}
$("runList").addEventListener("click", (e) => { const b = e.target.closest("[data-run]"); if (b) watchRun(b.dataset.run); });

function barRows(items, valueKey, labelFn, bestIdx = 0, digits = 4) {
  const vals = items.map((x) => x[valueKey]);
  const max = Math.max(...vals.map((v) => Math.abs(v)), 1e-9);
  const min = Math.min(...vals);
  const floor = min > 0 && (max - min) / max < 0.25 ? min * 0.9 : 0; // zoom when values are close
  return `<div class="bars">${items.map((x, k) => {
    const w = Math.max(3, (100 * (Math.abs(x[valueKey]) - floor)) / (max - floor));
    return `<div class="barrow ${k === bestIdx ? "best" : ""}"><span class="lbl">${labelFn(x)}</span><span class="track"><span class="fill" style="width:${w}%;display:block"></span></span><span class="num">${fmt(x[valueKey], digits)}</span></div>`;
  }).join("")}</div>`;
}
function showResults(res) {
  ST.result = res;
  const tm = res.test_metrics;
  const cls = res.task === "classification";
  const keys = cls ? ["f1_macro", "accuracy", "roc_auc"] : ["r2", "rmse", "mae"];
  const names = { f1_macro: "F1 score on unseen data", accuracy: "Accuracy", roc_auc: "ROC-AUC", r2: "R² on unseen data", rmse: "RMSE", mae: "Mean absolute error" };
  const tiles = `<div class="tiles"><div class="tile hero"><div class="v">${esc(res.best_model.replace(/_/g, " "))}</div><div class="k">Best model, chosen from ${new Set(res.leaderboard.map((r) => r.model)).size} families in ${fmt(res.seconds, 1)}s</div></div>` +
    keys.filter((k) => tm[k] != null).map((k) => `<div class="tile"><div class="v">${fmt(tm[k])}</div><div class="k">${names[k]}</div></div>`).join("") + "</div>";
  const lb = res.leaderboard.slice(0, 8);
  const leader = `<div class="panel"><h2>Leaderboard</h2>${barRows(lb, "cv_mean", (r) => `${esc(r.model.replace(/_/g, " "))}<small>${r.stage}</small>`)}<div class="note" style="margin-top:10px;font-size:12.5px">Cross-validated ${esc(res.metric)}. Tuned rows were improved by the Critic agent.</div></div>`;
  const fi = res.feature_importance.slice(0, 8);
  const drivers = `<div class="panel"><h2>What drives the prediction</h2>${fi.length ? barRows(fi, "importance", (f) => esc(f.feature), 0, 3) : '<div class="note">Not available for this model.</div>'}</div>`;
  let cm = "";
  if (cls && tm.confusion_matrix) {
    const labels = tm.labels, M = tm.confusion_matrix, mx = Math.max(...M.flat(), 1);
    cm = `<div class="panel"><h2>Where it gets it right</h2><div class="cm" style="grid-template-columns:90px repeat(${labels.length}, minmax(0,1fr))">
      <span></span>${labels.map((l) => `<span class="h">predicted ${esc(l)}</span>`).join("")}
      ${M.map((row, i) => `<span class="h" style="align-self:center;text-align:right">actual ${esc(labels[i])}</span>` + row.map((v, j) => {
        const a = 0.12 + 0.75 * (v / mx);
        const col = i === j ? `color-mix(in srgb, var(--good) ${Math.round(a * 100)}%, transparent)` : `color-mix(in srgb, var(--bad) ${Math.round(a * 80)}%, transparent)`;
        return `<span class="cell" style="background:${col}">${v}</span>`;
      }).join("")).join("")}</div><div class="note" style="margin-top:10px;font-size:12.5px">Green cells are correct predictions on the held-out test set.</div></div>`;
  }
  const playground = predictForm(res);
  const execMatch = (res.report_markdown || "").match(/## Executive summary\s+([\s\S]*?)\n## /);
  const concerns = (res.decisions && res.decisions.review && res.decisions.review.concerns) || [];
  const summary = `<div class="panel"><h2>What the Reporter found</h2><div class="prose">${md(execMatch ? execMatch[1].trim() : "")}</div>
    ${concerns.length ? `<div class="note" style="margin-top:10px"><b>Critic's notes</b>${concerns.map((x) => `<div>${esc(x)}</div>`).join("")}</div>` : ""}</div>`;
  $("results").innerHTML = tiles + `<div class="two">${leader}${drivers}</div>` + `<div class="two">${cm || ""}${summary}</div>` + playground +
    `<details class="panel" style="margin:16px 0"><summary style="cursor:pointer;font-weight:600">Full report from the Reporter agent</summary><div class="prose" style="margin-top:14px">${md(res.report_markdown)}</div></details>` +
    `<a class="btn ghost" href="/api/automl/runs/${encodeURIComponent(res.run_id)}/model">${icon("download")}Download trained model (.joblib)</a>`;
  if (!cm) $("results").querySelectorAll(".two")[1].style.gridTemplateColumns = "1fr";
  $("predictBtn").onclick = runPredict;
}
function predictForm(res) {
  const fp = res.feature_plan;
  const cols = Object.fromEntries((res.profile.columns || []).map((c) => [c.name, c]));
  const field = (name, type) => {
    const c = cols[name] || {};
    if (type === "categorical" && c.top_values) {
      return `<label class="field"><span>${esc(name)}</span><select class="input" data-f="${esc(name)}">${Object.keys(c.top_values).map((v) => `<option>${esc(v)}</option>`).join("")}</select></label>`;
    }
    if (type === "numeric") {
      const isInt = /int|bool/.test(c.dtype || "");
      const v = c.mean != null ? (isInt ? Math.round(c.mean) : +(+c.mean).toFixed(2)) : "";
      const range = c.min != null && c.max != null ? `<small>${fmt(c.min, 2)} to ${fmt(c.max, 2)}</small>` : "";
      return `<label class="field"><span>${esc(name)}</span><input class="input" type="number" step="${isInt ? 1 : "any"}" data-f="${esc(name)}" data-num="1" value="${v}">${range}</label>`;
    }
    if (type === "datetime") return `<label class="field"><span>${esc(name)}</span><input class="input" type="date" data-f="${esc(name)}"></label>`;
    return `<label class="field"><span>${esc(name)}</span><input class="input" data-f="${esc(name)}"></label>`;
  };
  const fields = [...fp.categorical.map((n) => field(n, "categorical")), ...fp.numeric.map((n) => field(n, "numeric")), ...fp.datetime.map((n) => field(n, "datetime")), ...fp.text.map((n) => field(n, "text"))].slice(0, 16);
  return `<div class="panel"><h2>Try the model</h2><div class="note" style="margin:-6px 0 14px;font-size:13px">Change any value and ask the trained ${esc(res.best_model.replace(/_/g, " "))} to predict <b>${esc(res.target)}</b>. Numbers start at the dataset average.</div>
    <div class="predict-grid">${fields.join("")}</div><div class="predict-out"><button class="btn" id="predictBtn">${icon("play")}Predict</button><div class="verdict" id="verdict"></div></div></div>`;
}
async function runPredict() {
  const row = {};
  document.querySelectorAll("[data-f]").forEach((el) => {
    if (el.value === "") return;
    row[el.dataset.f] = el.dataset.num ? Number(el.value) : el.value;
  });
  try {
    const r = await api(`/api/automl/runs/${encodeURIComponent(ST.result.run_id)}/predict`, { method: "POST", body: JSON.stringify({ rows: [row] }) });
    const p = r.predictions[0];
    $("verdict").innerHTML = `${esc(ST.result.target)}: ${typeof p === "number" ? fmt(p, 2) : esc(p)}<small>Prediction from the trained model for the values above.</small>`;
  } catch (err) { toast(err.message, true); }
}

// ============================================================ settings
const PROVIDERS = {
  openai: { name: "OpenAI", desc: "GPT models. Key from platform.openai.com", key: true },
  gemini: { name: "Google Gemini", desc: "Key from aistudio.google.com", key: true },
  ollama: { name: "Ollama", desc: "Free local models, no key. Needs Ollama running", key: false },
  mock: { name: "Offline mode", desc: "No AI model. Answers stitched from your documents", key: false },
};
let SET = null, chosen = null;
async function loadSettings() {
  try { SET = await api("/api/settings"); } catch (err) { $("setMsg").innerHTML = `<span class="status-err">${esc(err.message)}</span>`; return; }
  chosen = chosen || SET.provider;
  $("envPath").textContent = SET.env_file;
  $("userName").value = store.get("name", "");
  applyTheme(store.get("theme", "dark"));
  renderProviders();
}
function renderProviders() {
  $("providers").innerHTML = SET.providers.map((p) => {
    const info = PROVIDERS[p] || { name: p, desc: "" };
    return `<button class="prov ${p === chosen ? "on" : ""}" data-p="${p}">${p === SET.provider ? '<span class="cur">in use</span>' : ""}<b>${info.name}</b><small>${info.desc}</small></button>`;
  }).join("");
  const info = PROVIDERS[chosen] || {};
  $("setModel").value = chosen === SET.provider ? SET.model : "";
  $("setModel").placeholder = SET.default_models[chosen] || "";
  $("modelHint").textContent = SET.default_models[chosen] ? `Leave empty for ${SET.default_models[chosen]}.` : "Enter the model name exactly as your provider's console lists it.";
  $("keyField").classList.toggle("hidden", !info.key);
  const saved = SET.keys[chosen];
  $("keyLabel").textContent = `${info.name} API key`;
  $("keyHint").textContent = saved ? `A key ending ${saved.replace(/•/g, "")} is saved. Leave empty to keep it.` : "Paste the key once. It stays on this computer.";
}
$("providers").addEventListener("click", (e) => { const b = e.target.closest("[data-p]"); if (b) { chosen = b.dataset.p; $("setMsg").innerHTML = ""; renderProviders(); } });
$("showKey").onclick = () => { const k = $("setKey"); k.type = k.type === "password" ? "text" : "password"; $("showKey").textContent = k.type === "password" ? "Show" : "Hide"; };
$("saveSet").onclick = async () => {
  $("saveSet").disabled = true;
  $("setMsg").innerHTML = '<span class="spin"></span> Sending a test message to the model…';
  try {
    const r = await api("/api/settings", { method: "POST", body: JSON.stringify({ provider: chosen, model: $("setModel").value, api_key: $("setKey").value }) });
    $("setKey").value = "";
    $("setMsg").innerHTML = `<span class="status-ok">${icon("check", "width:16px;height:16px;vertical-align:-3px")} Connected to ${esc(r.model)}. All agents now use it.</span>`;
    toast("Model connected");
    SET = null;
    await loadSettings();
    loadHealth();
  } catch (err) { $("setMsg").innerHTML = `<span class="status-err">${esc(err.message)}</span>`; }
  $("saveSet").disabled = false;
};
$("userName").addEventListener("input", () => store.set("name", $("userName").value.trim()));
$("clearChats").onclick = () => {
  if (!confirm("Delete all chats in this browser? Documents and experiments are kept.")) return;
  S.chats = [];
  saveChats();
  newChat();
  toast("Chats deleted");
};

// ============================================================ keyboard
document.addEventListener("keydown", (e) => {
  const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName);
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); newChat(); }
  else if (e.key === "Escape") { closeMenus(); document.querySelector(".lightbox")?.remove(); }
  else if (e.key === "/" && !typing) { e.preventDefault(); show("chat"); $("input").focus(); }
});

// ============================================================ boot
(async function boot() {
  applyTheme(store.get("theme", "dark"));
  setMode(S.mode in MODES ? S.mode : "auto");
  renderChatList();
  await Promise.all([loadHealth(), loadDocs(false), loadDatasets()]);
  renderThread();
  $("input").focus();
  setInterval(loadHealth, 30000);
})();
