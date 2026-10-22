"use strict";

// Keep score following on the same clock and source offsets as audio playback.
const ReviewPlayback = (() => {
  const unsafeRepeatMessage = "Playback is disabled: a backward repeat follows a closed repeat without a new left repeat. Check the repeat route against the manuscript; you can still edit and save this draft.";

  function repeatWarning(first, constants) {
    let open = false, closed = false;
    for (let symbol = first; symbol; symbol = symbol.ts_next) {
      if (symbol.type !== constants.BAR || !symbol.seqst) continue;
      const bar = symbol.bar_type || "";
      if (bar.startsWith(":")) {
        if (closed && !open) return unsafeRepeatMessage;
        if (open) { open = false; closed = true; }
      }
      // abc2svg represents :|: as ::, which also opens the next repeat.
      if (bar.endsWith(":")) open = true;
    }
    return null;
  }

  function requireSafeRepeat(first, constants) {
    const warning = playbackWarning(first, constants);
    if (warning) throw new Error(warning);
  }

  function partPlan(first, constants) {
    if (!first.parts) return null;
    const fail = reason => { throw new Error(`Playback is disabled: ${reason} You can still edit and save.`); };
    const order = first.parts.replace(/\s/g, "");
    if (!/^[A-Z]{1,128}$/.test(order)) fail("Use at most 128 single-letter section names in the P: order.");
    const sections = [], names = new Set();
    let end = 0;
    for (let s = first; s; s = s.ts_next) {
      if (s.type === constants.BAR && /:/.test(s.bar_type || "")) {
        fail("Use either a P: section order or local repeat bars, rather than combining them.");
      }
      if ([constants.NOTE, constants.REST].includes(s.type)) end = Math.max(end, s.time + s.dur);
      if (!s.part) continue;
      const name = s.part.text.trim();
      if (!/^[A-Z]$/.test(name) || names.has(name)) fail("Each section needs a unique single-letter P: label.");
      names.add(name); sections.push({name, start: s.time});
    }
    if (!sections.length || sections[0].start !== 0) fail("Label the first section at the start of the music.");
    for (let i = 0; i < sections.length; i++) {
      sections[i].end = sections[i + 1]?.start ?? end;
      if (sections[i].end <= sections[i].start) fail("Each section must contain music.");
    }
    if ([...order].some(name => !names.has(name))) fail("The P: order refers to a missing section label.");
    return {order, sections};
  }

  function playbackWarning(first, constants) {
    const warning = repeatWarning(first, constants);
    if (warning) return warning;
    try { partPlan(first, constants); } catch (error) { return error.message; }
    return null;
  }

  function fillMissingTieIds(first, constants) {
    // This abc2svg build links tied heads but leaves b40 unset. ToAudio uses
    // b40 to find each continuation, so unset chord heads all match one head.
    for (let symbol = first; symbol; symbol = symbol.ts_next) {
      if (symbol.type !== constants.NOTE) continue;
      for (const note of symbol.notes) {
        if (!note.tie_ty || !note.tie_e || note.tie_s) continue;
        const chain = [], seen = new Set();
        let head = note;
        while (head && !seen.has(head)) {
          chain.push(head); seen.add(head); head = head.tie_e;
        }
        if (head) continue; // Leave an invalid cyclic tie to the renderer.
        const existing = new Set(chain.map(item => item.b40).filter(id => id != null));
        if (existing.size > 1) continue; // Keep conflicting supplied identities intact.
        const id = existing.size ? existing.values().next().value : Symbol("tie");
        for (const item of chain) if (item.b40 == null) item.b40 = id;
      }
    }
  }

  function audioEvents(Audio, first, voices, constants) {
    requireSafeRepeat(first, constants);
    const plan = partPlan(first, constants), times = new Map();
    for (let s = first; s; s = s.ts_next) {
      if (Number.isInteger(s.istart)) times.set(s.istart, s.time);
    }
    fillMissingTieIds(first, constants);
    const audio = new Audio(); audio.add(first, voices);
    const raw = Array.from(audio.clear() || [], event => Array.from(event));
    if (!plan) return raw;
    // Reorder audio occurrences, retaining original source offsets for score following.
    // MIDI pitches already include each section's written key/accidental state.
    const timed = raw.filter(e => times.has(e[0]) && e[2] >= 0 && e[4] > 0);
    const rendered = new Map();
    for (const section of plan.sections) {
      const music = timed.filter(e => times.get(e[0]) >= section.start && times.get(e[0]) < section.end);
      if (!music.length || Math.min(...music.map(e => times.get(e[0]))) !== section.start) {
        throw new Error("Playback is disabled: each section must begin with a note or rest.");
      }
      const start = Math.min(...music.map(e => e[1]));
      const next = timed.filter(e => times.get(e[0]) >= section.end);
      const end = next.length ? Math.min(...next.map(e => e[1])) : Math.max(...music.map(e => e[1] + e[4]));
      if (music.some(e => e[1] + e[4] > end + .0001)) {
        throw new Error("Playback is disabled: a held note crosses a section boundary. Check the tie or section label.");
      }
      const events = raw.filter(e => {
        const time = times.get(e[0]);
        return time >= section.start && (time < section.end ||
          (time === section.end && e[3] === 0 && e[4] === 0)); // Closing bar markers.
      });
      rendered.set(section.name, {events, start, duration: end - start});
    }
    const result = raw.filter(e => !times.has(e[0])), duration = [...plan.order]
      .reduce((total, name) => total + rendered.get(name).duration, 0);
    if (!Number.isFinite(duration) || duration <= 0 || duration > 1800) {
      throw new Error("Playback is disabled: the section order exceeds the 30-minute playback limit.");
    }
    let offset = 0;
    for (const name of plan.order) {
      const section = rendered.get(name);
      for (const event of section.events) {
        const copy = [...event]; copy[1] = offset + event[1] - section.start; result.push(copy);
      }
      offset += section.duration;
    }
    return result;
  }

  function initialBpm(first, constants) {
    let bpm = 120; // ToAudio's default is quarter note = 120.
    for (let symbol = first; symbol && symbol.time === 0; symbol = symbol.ts_next) {
      if (symbol.type !== constants.TEMPO || !symbol.tempo) continue;
      const candidate = symbol.tempo * (symbol.tempo_notes || []).reduce((sum, value) => sum + value, 0) /
        (constants.BLEN / 4);
      if (Number.isFinite(candidate) && candidate > 0) bpm = candidate;
    }
    return bpm;
  }

  function timeline(events) {
    const seen = new Set(), result = [];
    for (const [index, start, instrument, pitch, duration, volume] of events) {
      if (![index, start, instrument, pitch, duration, volume].every(Number.isFinite) ||
          !Number.isInteger(index) || index < 0 || start < 0 || start > 1800 ||
          instrument < 0 || duration <= 0 ||
          !((pitch > 0 && volume > 0) || (pitch === 0 && volume === 0))) continue;
      const end = start + Math.min(duration, 60);
      const key = `${index}:${start}:${end}`;
      if (seen.has(key)) continue; // All heads of one written chord share its highlight.
      seen.add(key);
      result.push({index, start, end});
    }
    return result.sort((left, right) => left.start - right.start || left.index - right.index);
  }

  function activeAt(events, seconds) {
    if (!Number.isFinite(seconds)) return [];
    return events.filter(event => event.start <= seconds && seconds < event.end);
  }

  function writtenEvents(Abc, Audio, constants, text) {
    let events = [];
    // Parse a separate display model: audio keeps its ties, while the cursor moves
    // across each written continuation. The user's ABC and rendered model stay intact.
    const parser = new Abc({
      img_out: () => {},
      read_file: () => { throw new Error("External includes are not supported."); },
      get_abcmodel: (first, voices) => {
        requireSafeRepeat(first, constants);
        for (let symbol = first; symbol; symbol = symbol.ts_next) {
          if (symbol.type !== constants.NOTE) continue;
          for (const note of symbol.notes) {
            delete note.tie_ty;
            delete note.ti2;
          }
        }
        events.push(...audioEvents(Audio, first, voices, constants));
      }
    });
    parser.tosvg("playback-follow", text);
    return timeline(events);
  }

  function accompaniment(Abc, Audio, constants, text) {
    const warnings = new Set(), result = [];
    const qualities = {
      "": [0, 4, 7], maj: [0, 4, 7], M: [0, 4, 7],
      m: [0, 3, 7], min: [0, 3, 7], "-": [0, 3, 7],
      "6": [0, 4, 7, 9], m6: [0, 3, 7, 9],
      "7": [0, 4, 7, 10], maj7: [0, 4, 7, 11], M7: [0, 4, 7, 11],
      m7: [0, 3, 7, 10], min7: [0, 3, 7, 10],
      dim: [0, 3, 6], o: [0, 3, 6], dim7: [0, 3, 6, 9], o7: [0, 3, 6, 9],
      aug: [0, 4, 8], "+": [0, 4, 8], sus: [0, 5, 7], sus4: [0, 5, 7],
      sus2: [0, 2, 7], "7sus4": [0, 5, 7, 10],
    };
    function pitchClass(note) {
      return ({C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11}[note[0]] +
        (note[1] === "#" ? 1 : note[1] === "b" ? -1 : 0) + 12) % 12;
    }
    function chord(symbol) {
      const normalized = symbol.trim().replace(/♯/g, "#").replace(/♭/g, "b");
      if (/^N\.?C\.?$/i.test(normalized)) return {symbol, pitches: []};
      const match = /^([A-G][#b]?)([^/]*)(?:\/([A-G][#b]?))?$/.exec(normalized);
      if (!match || !Object.hasOwn(qualities, match[2])) {
        warnings.add(`Accompaniment is silent for unsupported chord “${symbol}”.`);
        return {symbol, pitches: []};
      }
      const root = 48 + pitchClass(match[1]);
      const pitches = qualities[match[2]].map(interval => root + interval);
      if (match[3]) pitches.unshift(36 + pitchClass(match[3]));
      return {symbol, pitches};
    }
    function resolve(symbols) {
      const unique = [...new Set(symbols)];
      if (unique.length !== 1) {
        warnings.add("Accompaniment is silent where simultaneous chord symbols disagree.");
        return {symbol: unique.join(" / "), pitches: []};
      }
      return chord(unique[0]);
    }
    const parser = new Abc({
      img_out: () => {},
      read_file: () => { throw new Error("External includes are not supported."); },
      get_abcmodel: (first, voices) => {
        requireSafeRepeat(first, constants);
        const symbols = new Map(), barMarkers = new Set(), markers = [];
        for (let s = first; s; s = s.ts_next) {
          if (s.type === constants.BAR) {
            // Zero-length rests expose barline times on this private audio model.
            // Insert before the bar so repeat jumps cannot skip the closing marker.
            const index = text.length + barMarkers.size + 1;
            const marker = {type: constants.REST, dur: 0, time: s.time, v: s.v,
              istart: index, ts_prev: s.ts_prev, ts_next: s};
            if (s.ts_prev) s.ts_prev.ts_next = marker;
            else first = marker;
            s.ts_prev = marker; barMarkers.add(index); markers.push(marker);
          }
          if (![constants.NOTE, constants.REST].includes(s.type)) continue;
          const labels = (s.a_gch || []).filter(g => g.type === "g").map(g => g.text);
          symbols.set(s.istart, {time: s.time, labels});
          // Reveal tied continuations as timing markers without changing sounding melody.
          if (s.type === constants.NOTE) for (const note of s.notes) {
            delete note.tie_ty; delete note.ti2;
          }
        }
        let raw;
        try {
          raw = audioEvents(Audio, first, voices, constants);
        } finally {
          // The parser continues engraving after this callback; remove timing-only symbols.
          for (const marker of markers) {
            if (marker.ts_prev) marker.ts_prev.ts_next = marker.ts_next;
            marker.ts_next.ts_prev = marker.ts_prev;
          }
        }
        const played = timeline(raw);
        const groups = new Map();
        for (const event of played) {
          if (!symbols.has(event.index)) continue;
          if (!groups.has(event.start)) groups.set(event.start, []);
          groups.get(event.start).push(event);
        }
        const boundaries = new Set(raw.filter(event => barMarkers.has(event[0]))
          .map(event => event[1]).filter(start => Number.isFinite(start) && start >= 0 && start <= 1800));
        for (const start of boundaries) if (!groups.has(start)) groups.set(start, []);
        let active = null, previousTime = -Infinity, previousSignature = "";
        function change(next, start, index) {
          if (active && start > active.start && active.pitches.length) {
            result.push({...active, duration: start - active.start});
          }
          active = next ? {symbol: next.symbol, pitches: next.pitches, start, index} : null;
        }
        for (const [start, group] of [...groups].sort((a, b) => a[0] - b[0])) {
          if (boundaries.has(start)) change(null, start);
          if (!group.length) continue;
          const time = Math.min(...group.map(event => symbols.get(event.index).time));
          const signature = group.map(event => event.index).join(":");
          const jump = time < previousTime || (time === previousTime && signature === previousSignature);
          if (jump) change(null, start);
          const labels = group.flatMap(event => symbols.get(event.index).labels);
          if (labels.length) change(resolve(labels), start, group[0].index);
          previousTime = time; previousSignature = signature;
        }
        const end = played.reduce((latest, event) => Math.max(latest, event.end), 0);
        if (active && end > active.start && active.pitches.length) {
          result.push({...active, duration: end - active.start});
        }
      }
    });
    parser.tosvg("playback-accompaniment", text);
    return {events: result, warnings: [...warnings]};
  }

  return {initialBpm, repeatWarning, playbackWarning, audioEvents, timeline, activeAt, writtenEvents, accompaniment};
})();

if (typeof module === "object") module.exports = ReviewPlayback;
