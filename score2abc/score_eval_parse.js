"use strict";

// Runs as a bounded subprocess. Keep this adapter separate from the browser
// renderer so evaluation does not change review or playback behavior.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const {abc} = JSON.parse(fs.readFileSync(0, "utf8"));
const errors = [], warnings = [];
const result = {
  schema_version: 1, valid: false, errors, warnings, ticks_per_quarter: 384,
  measures: [], route: null, route_status: "unavailable",
  counts: {measures: 0, noteheads: 0, rests: 0, chord_labels: 0, performed_measures: null},
};
const unsafe = /^\s*(?:%%|I:)\s*(?:begin\w*|end\w*|include|abc-include|abcm2ps|ss-pref|js|javascript|ps|postscript|svg|html)\b/im;
if (typeof abc !== "string" || abc.length > 1_000_000) errors.push("ABC input must be text of at most 1 MB.");
if (!errors.length) {
  for (const field of ["X", "M", "K"]) {
    if (!new RegExp("^" + field + ":\\s*\\S+", "m").test(abc)) errors.push("Missing " + field + " header.");
  }
  if ((abc.match(/^X:/gm) || []).length !== 1) errors.push("Exactly one tune is required.");
  if (/^(?:M|K):\s*\?\s*$/m.test(abc)) errors.push("Confirm meter and key from the manuscript.");
  if (unsafe.test(abc) || /<\s*\/?\s*(?:script|svg|html|iframe)\b/i.test(abc)) {
    errors.push("Executable, embedded markup and include directives are not allowed.");
  }
}
if (!errors.length) {
  try {
    const dir = process.argv[2], context = vm.createContext({});
    for (const filename of ["abc2svg-1.js", "toaudio-1.js"]) {
      vm.runInContext(fs.readFileSync(path.join(dir, filename), "utf8"), context, {timeout: 3000});
    }
    const playback = vm.runInContext(fs.readFileSync(
      path.join(__dirname, "review_playback.js"), "utf8") + "\nReviewPlayback", context,
      {timeout: 3000});
    const C = context.abc2svg.C;
    let tunes = 0;
    new context.abc2svg.Abc({
      img_out: () => {},
      read_file: () => {throw new Error("Includes are not allowed.");},
      errbld: (severity, message) => (severity === "warn" ? warnings : errors).push(String(message)),
      get_abcmodel: (first, voices) => {
        tunes++;
        if (tunes !== 1) return;
        normalizeModel(first, voices, C, context.ToAudio, playback);
      },
    }).tosvg("score-eval.abc", abc);
    if (tunes !== 1) errors.push("The renderer must parse exactly one tune.");
  } catch (error) {
    errors.push("Renderer rejected notation: " + String(error.message || error));
  }
}
result.valid = errors.length === 0;
process.stdout.write(JSON.stringify(result));

function normalizeModel(first, voices, C, ToAudio, playback) {
  const symbols = [];
  for (let s = first; s; s = s.ts_next) symbols.push(s);
  const sounding = symbols.filter(s => s.type === C.NOTE || s.type === C.REST);
  if (!sounding.length) {errors.push("Notation has no notes or rests."); return;}
  const explicitBarsByVoice = new Map();
  for (const s of symbols.filter(s => s.type === C.BAR)) {
    if (!explicitBarsByVoice.has(s.v)) explicitBarsByVoice.set(s.v, new Set());
    explicitBarsByVoice.get(s.v).add(s.time);
  }
  const explicitVoices = [...explicitBarsByVoice.keys()];
  for (let i = 0; i < explicitVoices.length; i++) for (let j = i + 1; j < explicitVoices.length; j++) {
    const left = explicitVoices[i], right = explicitVoices[j];
    const leftNotes = sounding.filter(s => s.v === left), rightNotes = sounding.filter(s => s.v === right);
    if (!leftNotes.length || !rightNotes.length) continue;
    const start = Math.max(Math.min(...leftNotes.map(s => s.time)),
      Math.min(...rightNotes.map(s => s.time)));
    const stop = Math.min(Math.max(...leftNotes.map(s => s.time + s.dur)),
      Math.max(...rightNotes.map(s => s.time + s.dur)));
    if (stop <= start) continue;
    const within = times => [...times].filter(time => time > start && time <= stop).sort((a, b) => a - b);
    const a = within(explicitBarsByVoice.get(left)), b = within(explicitBarsByVoice.get(right));
    if (a.length !== b.length || a.some((time, index) => time !== b[index])) {
      errors.push("Explicit voice barlines are not synchronized in their shared span.");
      return;
    }
  }
  const end = Math.max(...sounding.map(s => s.time + s.dur));
  const boundaries = [...new Set([0, ...symbols.filter(s => s.type === C.BAR &&
    s.time > 0 && s.time < end).map(s => s.time), end])].sort((a, b) => a - b);
  const symbolByIndex = new Map(symbols.filter(s => Number.isInteger(s.istart))
    .map(s => [s.istart, s]));
  const meterChanges = symbols.filter(s => s.type === C.METER && Array.isArray(s.a_meter))
    .map(s => ({time: s.time, meter: meterOf(s.a_meter)}));
  const initial = /(?:^|\n)M:\s*([^\n%]+)/.exec(abc);
  const initialMeter = meterText(initial ? initial[1].trim() : "");

  // The review audio converter applies ABC key and accidental state. Ask it for
  // every written head, including tied continuations, without altering the
  // durations or onset symbols captured above.
  const ties = [];
  for (const s of symbols) if (s.type === C.NOTE) for (const note of s.notes) {
    ties.push([note, note.tie_ty, note.ti2]);
    delete note.tie_ty; delete note.ti2;
  }
  let played;
  try {played = playback.audioEvents(ToAudio, first, voices, C);} catch (error) {
    result.route_status = "unsupported";
    warnings.push(String(error.message || error));
  }
  // audioEvents may repeat source offsets. One occurrence suffices for each
  // written head; retain duplicates at a single source offset for chords.
  const pitchByIndex = new Map();
  if (played) for (const event of played) {
    const [index, , instrument, pitch, duration, volume] = event;
    if (instrument < 0 || duration <= 0 || volume <= 0 || pitch <= 0) continue;
    if (!pitchByIndex.has(index)) pitchByIndex.set(index, []);
    const list = pitchByIndex.get(index);
    if (list.length < (symbolByIndex.get(index)?.notes?.length || 0)) list.push(pitch);
  }
  // If the repeat route is unsupported, a linear ToAudio conversion still
  // yields pitches; remove the unsafe navigation symbols only on this model.
  if (!played) {
    const bars = symbols.filter(s => s.type === C.BAR);
    const old = bars.map(s => [s, s.bar_type, s.text]);
    try {
      for (const s of bars) {s.bar_type = "|"; s.text = "";}
      const audio = new ToAudio(); audio.add(first, voices);
      for (const event of Array.from(audio.clear() || [], e => Array.from(e))) {
        const [index, , instrument, pitch, duration, volume] = event;
        if (instrument < 0 || duration <= 0 || volume <= 0 || pitch <= 0) continue;
        if (!pitchByIndex.has(index)) pitchByIndex.set(index, []);
        const list = pitchByIndex.get(index);
        if (list.length < (symbolByIndex.get(index)?.notes?.length || 0)) list.push(pitch);
      }
    } catch (error) {errors.push("Pitch conversion failed: " + String(error.message || error));}
    finally {for (const [s, type, text] of old) {s.bar_type = type; s.text = text;}}
  }
  for (const [note, tie, ti2] of ties) {if (tie !== undefined) note.tie_ty = tie;
    if (ti2 !== undefined) note.ti2 = ti2;}

  const voice = s => String(voices[s.v]?.id ?? s.v ?? 0);
  const sourceMeasure = new Map();
  for (let i = 0; i < boundaries.length - 1; i++) {
    const start = boundaries[i], stop = boundaries[i + 1];
    const meter = [...meterChanges].reverse().find(x => x.time <= start)?.meter || initialMeter;
    const expected = meter ? 1536 * meter.numerator / meter.denominator : null;
    const duration = stop - start;
    const measure = {
      number: i + 1, duration_ticks: duration, expected_ticks: expected,
      meter, meter_valid: expected === null ? null : duration === expected,
      partial_reason: null, notes: [], rests: [], chords: [],
    };
    for (const s of sounding.filter(x => x.time >= start && x.time < stop)) {
      sourceMeasure.set(s.istart, i + 1);
      const onset = s.time - start;
      if (s.type === C.NOTE) {
        const pitches = pitchByIndex.get(s.istart) || [];
        if (pitches.length !== s.notes.length) {
          errors.push(`Could not resolve all pitches in written measure ${i + 1}.`);
        }
        s.notes.forEach((n, j) => measure.notes.push({
          onset_ticks: onset, duration_ticks: n.dur || s.dur,
          pitch_midi: pitches[j] ?? null, voice: voice(s),
          tie_start: Boolean(ties.find(t => t[0] === n)?.[1]),
          tie_end: Boolean(n.tie_s),
        }));
      } else {
        measure.rests.push({onset_ticks: onset, duration_ticks: s.dur,
          voice: voice(s), visible: !s.invis});
      }
      for (const label of (s.a_gch || []).filter(g => g.type === "g")) {
        measure.chords.push({onset_ticks: onset, symbol: label.text, voice: voice(s)});
      }
    }
    for (const name of ["notes", "rests", "chords"]) {
      measure[name].sort((a, b) => a.onset_ticks - b.onset_ticks ||
        a.voice.localeCompare(b.voice) || JSON.stringify(a).localeCompare(JSON.stringify(b)));
    }
    result.measures.push(measure);
  }
  const measures = result.measures;
  if (measures.length && measures[0].expected_ticks !== null &&
    measures[0].duration_ticks < measures[0].expected_ticks) {
    measures[0].partial_reason = "pickup_candidate"; measures[0].meter_valid = null;
    const last = measures.at(-1);
    if (last !== measures[0] && last.expected_ticks === measures[0].expected_ticks &&
        last.duration_ticks + measures[0].duration_ticks === last.expected_ticks) {
      measures[0].partial_reason = "pickup"; measures[0].meter_valid = true;
      last.partial_reason = "pickup_complement"; last.meter_valid = true;
    }
  }
  result.counts = {
    measures: measures.length,
    noteheads: measures.reduce((n, m) => n + m.notes.length, 0),
    rests: measures.reduce((n, m) => n + m.rests.filter(r => r.visible).length, 0),
    chord_labels: measures.reduce((n, m) => n + m.chords.length, 0),
    performed_measures: null,
  };
  if (!played) return;
  if (symbols.some(s => s.part && !boundaries.includes(s.time))) {
    result.route_status = "unsupported";
    warnings.push("A section label begins inside a written measure; physical-bar route is unsupported.");
    return;
  }
  const firstIndex = new Map();
  for (let i = 0; i < boundaries.length - 1; i++) {
    const start = boundaries[i], stop = boundaries[i + 1];
    const first = sounding.filter(s => s.time >= start && s.time < stop)
      .sort((a, b) => a.time - b.time || a.istart - b.istart)[0];
    if (first) firstIndex.set(i + 1, first.istart);
  }
  const route = [];
  let lastVisit = null;
  for (const event of played) {
    const [index, time, instrument, , duration] = event;
    if (instrument < 0 || duration <= 0) continue;
    const measure = sourceMeasure.get(index);
    if (!measure || index !== firstIndex.get(measure)) continue;
    const visit = `${measure}:${time}`;
    if (visit !== lastVisit) {route.push(measure); lastVisit = visit;}
  }
  if (route.length && new Set(route).size === measures.length) {
    result.route = route; result.route_status = "resolved";
    result.counts.performed_measures = route.length;
  } else {result.route_status = "unsupported"; warnings.push("Playback route has no resolved measures.");}
}

function meterText(text) {
  const match = /^(\d+)\s*\/\s*(\d+)$/.exec(text);
  if (match) return {numerator: Number(match[1]), denominator: Number(match[2])};
  if (text === "C") return {numerator: 4, denominator: 4};
  if (text === "C|") return {numerator: 2, denominator: 2};
  return null;
}

function meterOf(items) {
  if (items.length !== 1) return null;
  const item = items[0];
  return meterText(item.top + (item.bot ? "/" + item.bot : ""));
}
