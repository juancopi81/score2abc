"use strict";

// Keep score following on the same clock and source offsets as audio playback.
const ReviewPlayback = (() => {
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
        for (let symbol = first; symbol; symbol = symbol.ts_next) {
          if (symbol.type !== constants.NOTE) continue;
          for (const note of symbol.notes) {
            delete note.tie_ty;
            delete note.ti2;
          }
        }
        const audio = new Audio();
        audio.add(first, voices);
        events.push(...Array.from(audio.clear() || [], event => Array.from(event)));
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
        const audio = new Audio(); let raw;
        try {
          audio.add(first, voices);
          raw = Array.from(audio.clear() || [], event => Array.from(event));
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

  return {initialBpm, timeline, activeAt, writtenEvents, accompaniment};
})();

if (typeof module === "object") module.exports = ReviewPlayback;
