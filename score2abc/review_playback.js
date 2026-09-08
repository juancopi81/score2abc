"use strict";

// Keep score following on the same clock and source offsets as audio playback.
const ReviewPlayback = (() => {
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

  return {timeline, activeAt, writtenEvents};
})();

if (typeof module === "object") module.exports = ReviewPlayback;
