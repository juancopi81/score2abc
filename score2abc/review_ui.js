"use strict";

const $ = id => document.getElementById(id);
const desk = {data: null, work: null, dirty: false, rendering: null, audio: null,
  oscillators: [], audioEvents: [], symbols: [], reviewMs: 0, lastTick: Date.now(),
  lastAction: Date.now(), renderTimer: null, playbackTimer: null, loading: false};

function message(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function workPath() { return `/api/work/${encodeURIComponent(desk.work.slug)}`; }
function canRender() { return typeof abc2svg !== "undefined" && desk.data?.renderer_available; }

function setBusy(busy) {
  desk.loading = busy;
  for (const id of ["abc", "unresolved", "save"]) $(id).disabled = busy || !desk.work;
  if (busy) {
    clearTimeout(desk.renderTimer);
    stopPlayback();
    for (const id of ["approve", "play", "download"]) $(id).disabled = true;
  } else renderNotation();
}

function renderLibrary() {
  const search = $("search").value.trim().toLocaleLowerCase();
  $("works").replaceChildren();
  for (const work of desk.data.works) {
    if (!`${work.title} ${work.composer}`.toLocaleLowerCase().includes(search)) continue;
    const button = document.createElement("button");
    button.className = "work-link" + (work.slug === desk.work?.slug ? " active" : "") +
      (work.review_state === "reviewed" ? " reviewed" : "");
    button.setAttribute("aria-current", work.slug === desk.work?.slug ? "page" : "false");
    const dot = document.createElement("span"); dot.className = "work-dot";
    const label = document.createElement("span"); label.textContent = work.title;
    button.append(dot, label);
    button.onclick = () => loadWork(work.slug);
    $("works").append(button);
  }
  const approved = desk.data.works.filter(work => work.review_state === "reviewed").length;
  $("library-count").textContent = `${approved} reviewed · ${desk.data.works.length} available here`;
}

function setDirty() {
  if (!desk.work) return;
  desk.dirty = true;
  $("dirty").textContent = "Unsaved changes";
  $("review-state").textContent = "Draft";
  $("review-state").classList.remove("reviewed");
  $("download").disabled = true;
  clearTimeout(desk.renderTimer);
  desk.renderTimer = setTimeout(renderNotation, 350);
}

function sources() {
  const images = desk.work.sources.filter(source => source.kind !== "pdf");
  $("source-select").replaceChildren();
  for (const source of images) {
    const option = document.createElement("option"); option.value = source.id;
    option.textContent = source.label; $("source-select").append(option);
  }
  const first = images.find(source => source.kind === "system") || images[0];
  if (first) $("source-select").value = first.id;
  $("source-select").disabled = !images.length;
  const pdf = desk.work.sources.find(source => source.kind === "pdf");
  $("source-pdf").hidden = !pdf;
  if (pdf) $("source-pdf").href = pdf.url;
  changeSource();
}

function changeSource() {
  const source = desk.work.sources.find(item => item.id === $("source-select").value);
  $("source-image").hidden = !source;
  $("source-empty").hidden = Boolean(source);
  if (source) {
    $("source-image").src = source.url;
    $("source-image").alt = `${desk.work.metadata.title} · ${source.label}`;
  } else $("source-empty").textContent = "No source image is available for this work yet.";
  $("zoom").value = 100; $("source-image").style.width = "100%";
}

async function loadWork(slug) {
  if (desk.loading) return;
  if (desk.dirty && !confirm("Leave this melody and discard the unsaved changes?")) return;
  setBusy(true);
  try {
    message("Opening melody…");
    const work = await api(`/api/work/${encodeURIComponent(slug)}`);
    desk.work = work; desk.dirty = false; desk.reviewMs = 0; desk.lastTick = Date.now();
    $("abc").value = work.abc;
    $("unresolved").value = work.unresolved.join("\n");
    $("title").textContent = work.metadata.title;
    $("metadata").textContent = [work.metadata.composer, work.metadata.rhythm].filter(Boolean).join(" · ");
    document.title = `${work.metadata.title} · A Puño y Letra`;
    $("dirty").textContent = "Saved";
    $("review-state").textContent = work.review_state === "reviewed" ? "Reviewed" : "Draft";
    $("review-state").classList.toggle("reviewed", work.review_state === "reviewed");
    const notices = [];
    if (work.source_status === "no_recognition") notices.push("This melody has no recognized transcription yet. Confirm the meter (M:) and key (K:) from the manuscript, then begin a draft.");
    else if (/fixture|manual|supplied/.test(work.source_status)) notices.push("This starting transcription comes from supplied MusicXML. Use it to try the review workflow; it is not a new automatic recognition result.");
    if (work.base_changed) notices.push("Generated output has changed since this review began. Your saved corrections are preserved.");
    if (!canRender()) notices.push("The local ABC renderer is unavailable. Drafts can still be saved; preview, approval, and validated export need abc2svg and Node.js.");
    $("notice").textContent = notices.join(" "); $("notice").hidden = !notices.length;
    sources(); renderLibrary(); renderNotation(); updateTime();
    const url = new URL(location.href); url.searchParams.set("slug", slug);
    history.replaceState(null, "", url);
    message("Ready. Compare the manuscript and transcription, then save your corrections.");
  } catch (error) { message(error.message, true); }
  finally { setBusy(false); }
}

function renderNotation() {
  if (!desk.work || desk.loading) return;
  stopPlayback(); desk.audioEvents = []; desk.symbols = [];
  $("notation").replaceChildren(); $("validation").replaceChildren();
  $("approve").disabled = true; $("download").disabled = true; $("play").disabled = true;
  if (!canRender()) {
    $("notation").textContent = "Preview requires the local ABC renderer."; return;
  }
  const errors = [], warnings = []; let markup = "", notes = 0, abc;
  const text = $("abc").value;
  if (/^(?:M|K):\s*\?\s*$/m.test(text)) {
    errors.push("Confirm the meter (M:) and key (K:) from the manuscript.");
  }
  if (/^\s*(?:%%|I:)\s*(?:begin\w*|end\w*|include|abc-include|abcm2ps|ss-pref|js|javascript|ps|postscript|svg|html)\b/im.test(text) || /<\s*\/?\s*(?:script|svg|html|iframe)\b/i.test(text)) {
    $("validation").textContent = "Embedded code, markup, and external includes are not supported.";
    $("validation").className = "validation error"; return;
  }
  try {
    abc = new abc2svg.Abc({
      img_out: value => { markup += value; },
      errbld: (severity, value) => (String(severity).toLowerCase().startsWith("w") ? warnings : errors).push(value),
      read_file: () => { throw new Error("External includes are not supported."); },
      anno_stop: (type, start, end, x, y, width, height) => {
        if (!["note", "rest", "bar"].includes(type)) return;
        if (!Number.isInteger(start) || !Number.isInteger(end)) return;
        desk.symbols.push({start, end});
        abc.out_svg(`<rect class="score-hit" data-start="${start}" data-end="${end}" x="`);
        abc.out_sxsy(x, '" y="', y);
        abc.out_svg(`" width="${width.toFixed(2)}" height="${abc.sh(height).toFixed(2)}"/>`);
      },
      get_abcmodel: (first, voices) => {
        for (let symbol = first; symbol; symbol = symbol.ts_next) {
          if (symbol.type === abc2svg.C.NOTE) notes += symbol.notes.length;
        }
        if (typeof ToAudio !== "undefined") {
          const audio = new ToAudio(); audio.add(first, voices);
          desk.audioEvents.push(...Array.from(audio.clear() || []).map(event => Array.from(event)));
        }
      }
    });
    abc.tosvg("layout", `%%pagewidth ${Math.max(320, $("notation").clientWidth - 34)}px\n%%leftmargin 8px\n%%rightmargin 8px\n`);
    abc.tosvg("review", text);
    const safe = new DOMParser().parseFromString(`<div>${markup}</div>`, "text/html");
    // SVG output is locally generated; strip executable content as a second boundary.
    safe.querySelectorAll("script,iframe,object,embed,foreignObject").forEach(node => node.remove());
    safe.querySelectorAll("*").forEach(node => {
      for (const attr of [...node.attributes]) {
        if (/^on/i.test(attr.name) || (/href$/i.test(attr.name) && !attr.value.startsWith("#"))) node.removeAttribute(attr.name);
      }
    });
    $("notation").append(...safe.body.firstElementChild.childNodes);
    if (!notes) {
      const empty = document.createElement("p"); empty.className = "empty";
      empty.textContent = "Add notes to begin the transcription."; $("notation").append(empty);
    }
  } catch (error) { errors.push(error.message); }
  $("validation").className = "validation" + (errors.length ? " error" : "");
  const summary = document.createElement("p");
  summary.textContent = errors.length ? "Check the notation before marking this score reviewed." : notes ? `${notes} notes rendered. Check the music against the source.` : "No notes yet. You can save this incomplete draft.";
  $("validation").append(summary);
  for (const item of [...errors, ...warnings]) {
    const line = document.createElement("p"); line.textContent = item; $("validation").append(line);
  }
  $("approve").disabled = errors.length > 0 || !notes || $("unresolved").value.trim().length > 0;
  $("play").disabled = errors.length > 0 || !desk.audioEvents.some(event => event[2] >= 0 && event[3] > 0 && event[5] > 0);
  $("download").disabled = desk.dirty || desk.work.revision < 1 || !desk.work.validation?.valid;
}

async function save(reviewState = "draft") {
  if (!desk.work || desk.loading) return;
  setBusy(true);
  try {
    message("Saving corrections…");
    const work = await api(workPath(), {method: "POST", headers: {
      "Content-Type": "application/json", "X-Review-Token": desk.data.csrf_token
    }, body: JSON.stringify({revision: desk.work.revision, abc: $("abc").value,
      unresolved: $("unresolved").value.split("\n").map(line => line.trim()).filter(Boolean),
      review_state: reviewState, review_ms: Math.round(desk.reviewMs)})});
    desk.work = work; desk.dirty = false; desk.reviewMs = 0;
    const listed = desk.data.works.find(item => item.slug === work.slug);
    listed.review_state = work.review_state; listed.has_draft = true;
    $("dirty").textContent = "Saved";
    $("review-state").textContent = reviewState === "reviewed" ? "Reviewed" : "Draft";
    $("review-state").classList.toggle("reviewed", reviewState === "reviewed");
    renderLibrary(); renderNotation(); updateTime();
    message(reviewState === "reviewed" ? "Review recorded. Your transcription is ready to download." : "Draft saved. Your corrections will survive reopening and pipeline reruns.");
    if (work.validation?.errors?.length) message("Draft saved with notation errors. Resolve them before approval or export.");
  } catch (error) { message(error.message, true); }
  finally { setBusy(false); }
}

async function play() {
  stopPlayback();
  const Context = window.AudioContext || window.webkitAudioContext;
  if (!Context) { message("Audio playback is unavailable in this browser.", true); return; }
  try {
    desk.audio ||= new Context(); await desk.audio.resume();
    const base = desk.audio.currentTime + .1;
    let last = 0;
    for (const event of desk.audioEvents) {
      const [, time, instrument, pitch, duration, volume] = event;
      if (instrument < 0 || pitch <= 0 || volume <= 0 || duration <= 0) continue;
      if (![time, pitch, duration].every(Number.isFinite) || time > 1800) continue;
      const oscillator = desk.audio.createOscillator(), gain = desk.audio.createGain();
      oscillator.type = "triangle";
      oscillator.frequency.value = 440 * 2 ** ((pitch - 69) / 12);
      const start = base + time, finish = start + Math.min(duration, 60);
      gain.gain.setValueAtTime(0, start); gain.gain.linearRampToValueAtTime(.075, start + .008);
      gain.gain.setValueAtTime(.075, Math.max(start + .008, finish - .04));
      gain.gain.linearRampToValueAtTime(0, finish);
      oscillator.connect(gain); gain.connect(desk.audio.destination);
      oscillator.start(start); oscillator.stop(finish + .01);
      oscillator.onended = () => { oscillator.disconnect(); gain.disconnect(); };
      desk.oscillators.push(oscillator); last = Math.max(last, time + duration);
    }
    $("stop").disabled = false; $("play").disabled = true;
    desk.playbackTimer = setTimeout(stopPlayback, (last + .2) * 1000);
    message("Playing the current editor contents.");
  } catch (error) { stopPlayback(); message(error.message, true); }
}

function stopPlayback() {
  clearTimeout(desk.playbackTimer);
  for (const oscillator of desk.oscillators) { try { oscillator.stop(); } catch (_) {} }
  desk.oscillators = []; $("stop").disabled = true;
  $("play").disabled = !desk.audioEvents.some(event => event[2] >= 0 && event[3] > 0 && event[5] > 0);
}

function updateTime() {
  const minutes = ((desk.work?.active_review_ms || 0) + desk.reviewMs) / 60000;
  $("review-time").textContent = `Active review: ${minutes.toFixed(1)} min`;
}

document.addEventListener("pointerdown", () => { desk.lastAction = Date.now(); });
document.addEventListener("keydown", event => {
  desk.lastAction = Date.now();
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
    event.preventDefault(); save("draft");
  }
});
setInterval(() => {
  const now = Date.now();
  if (desk.work && !desk.loading && !document.hidden && document.hasFocus() && now - desk.lastAction < 30000) {
    desk.reviewMs += now - desk.lastTick;
  }
  desk.lastTick = now; updateTime();
}, 1000);
window.addEventListener("beforeunload", event => {
  if (desk.dirty) { event.preventDefault(); event.returnValue = ""; }
});
$("search").addEventListener("input", renderLibrary);
$("abc").addEventListener("input", setDirty);
$("unresolved").addEventListener("input", setDirty);
$("source-select").addEventListener("change", changeSource);
$("zoom").addEventListener("input", () => { $("source-image").style.width = `${$("zoom").value}%`; });
$("save").onclick = () => save("draft");
$("approve").onclick = () => save("reviewed");
$("play").onclick = play; $("stop").onclick = stopPlayback;
$("download").onclick = () => { if (!desk.dirty) location.href = `${workPath()}/export.abc`; };
$("notation").onclick = event => {
  const hit = event.target.closest(".score-hit"); if (!hit) return;
  $("notation").querySelectorAll(".selected").forEach(node => node.classList.remove("selected"));
  hit.classList.add("selected"); $("abc").focus();
  $("abc").setSelectionRange(Number(hit.dataset.start), Number(hit.dataset.end));
};
window.addEventListener("resize", () => {
  clearTimeout(desk.renderTimer); desk.renderTimer = setTimeout(renderNotation, 300);
});
api("/api/state").then(data => {
  desk.data = data; renderLibrary();
  const requested = new URL(location.href).searchParams.get("slug");
  const slug = data.works.find(work => work.slug === requested)?.slug || data.initial_slug || data.works[0]?.slug;
  if (slug) return loadWork(slug);
  $("title").textContent = "No melodies available";
  message("Ingest a collection to begin reviewing.");
}).catch(error => message(error.message, true));
