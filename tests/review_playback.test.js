const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const playback = require('../score2abc/review_playback.js');

test('one highlight per chord, including rests, with end-exclusive timing', () => {
  const events = playback.timeline([
    [0, 0, -1, 121, 0, 1],
    [10, 0, 0, 60, 1, 1], [10, 0, 0, 64, 1, 1],
    [20, 1, 0, 0, 1, 0], [30, 2, 0, 67, 1, 1],
    [40, NaN, 0, 60, 1, 1], [50, 0, -1, 7, 1, 1],
  ]);
  assert.equal(events.length, 3);
  assert.deepEqual(playback.activeAt(events, -.01), []);
  assert.deepEqual(playback.activeAt(events, .5).map(e => e.index), [10]);
  assert.deepEqual(playback.activeAt(events, 1).map(e => e.index), [20]);
  assert.deepEqual(playback.activeAt(events, 2).map(e => e.index), [30]);
  assert.deepEqual(playback.activeAt(events, 3), []);
});

test('overlapping voices highlight both sounding symbols', () => {
  const events = playback.timeline([[4, 0, 0, 60, 3, 1], [8, 1, 0, 67, 1, 1]]);
  assert.deepEqual(playback.activeAt(events, 1.5).map(e => e.index), [4, 8]);
  assert.deepEqual(playback.activeAt(events, 2).map(e => e.index), [4]);
});

test('selection clips overlapping voices and chords, preserves polyphony and final rests without mutation', () => {
  const audio = [
    [0, 0, -1, 121, 0, 1], [2, 0, 0, 55, 4, 1, 'held voice'],
    [6, 0, 0, 60, 2, 1], [10, 2, 0, 64, 1, 1], [10, 2, 0, 67, 1, 1],
    [14, 3, 0, 0, 2, 0], [18, 5, 0, 72, 1, 1], [22, 6, 0, 0, 3, 0],
  ];
  const chords = [{index: 6, start: 0, duration: 3, pitches: [48, 52, 55], symbol: 'C'},
    {index: 18, start: 5, duration: 1, pitches: [55, 59, 62], symbol: 'G'}];
  const visual = playback.timeline(audio);
  const before = JSON.stringify([audio, chords, visual]);
  for (const event of audio) Object.freeze(event);
  for (const event of chords) { Object.freeze(event.pitches); Object.freeze(event); }
  visual.forEach(Object.freeze);
  [audio, chords, visual].forEach(Object.freeze);
  const selected = playback.fromSelection(audio, chords, visual, 10);
  assert.equal(selected.start, 2); assert.equal(selected.end, 7);
  assert.deepEqual(selected.audioEvents, [
    [2, 0, 0, 55, 2, 1, 'held voice'], [10, 0, 0, 64, 1, 1], [10, 0, 0, 67, 1, 1],
    [14, 1, 0, 0, 2, 0], [18, 3, 0, 72, 1, 1], [22, 4, 0, 0, 3, 0],
  ]);
  assert.deepEqual(selected.chordEvents, [
    {index: 6, start: 0, duration: 1, pitches: [48, 52, 55], symbol: 'C'},
    {index: 18, start: 3, duration: 1, pitches: [55, 59, 62], symbol: 'G'},
  ]);
  assert.deepEqual(playback.activeAt(selected.visualEvents, 0).map(event => event.index), [2, 10]);
  assert.deepEqual(playback.activeAt(selected.visualEvents, 4).map(event => event.index), [22]);
  assert.deepEqual(playback.activeAt(selected.visualEvents, 7), []);
  assert.equal(JSON.stringify([audio, chords, visual]), before);
  assert.notEqual(selected.audioEvents[0], audio[1]);
  assert.notEqual(selected.chordEvents[0].pitches, chords[0].pitches);
  assert.notEqual(selected.visualEvents[0], visual[0]);
});

test('selection uses the earliest valid performed occurrence, including source index zero', () => {
  const audio = [[0, 4, 0, 60, 1, 1], [0, 1, 0, 60, 1, 1]];
  const visual = [{index: 0, start: 4, end: 5}, {index: 0, start: -1, end: 1},
    {index: 0, start: .5, end: .5}, {index: 0, start: NaN, end: 2},
    {index: 0, start: 1, end: 2}];
  const selected = playback.fromSelection(audio, [], visual, 0);
  assert.equal(selected.start, 1); assert.equal(selected.end, 4);
  assert.deepEqual(selected.audioEvents.map(event => event[1]), [3, 0]);
  for (const index of [undefined, null, -1, 1.5, NaN, Infinity, '0', 999]) {
    assert.equal(playback.fromSelection(audio, [], visual, index), null);
  }
  assert.equal(playback.fromSelection(audio, [], [], 0), null);
  assert.equal(playback.fromSelection(audio, [], [{index: 0, start: 2, end: 1}], 0), null);
  assert.equal(playback.fromSelection(null, [], visual, 0), null);
  assert.equal(playback.fromSelection(audio, null, visual, 0), null);
  assert.equal(playback.fromSelection(audio, [], null, 0), null);
});

test('selection retains silent rests without inventing melody or inherited accompaniment', () => {
  const audio = [[2, 0, 0, 60, 1, 1], [8, 1, 0, 0, 2, 0]];
  const chords = [{start: 0, duration: 1, pitches: [48, 52, 55], symbol: 'C', index: 2}];
  const selected = playback.fromSelection(audio, chords, playback.timeline(audio), 8);
  assert.deepEqual(selected.audioEvents, [[8, 0, 0, 0, 2, 0]]);
  assert.deepEqual(selected.chordEvents, []);
  assert.deepEqual(selected.visualEvents, [{index: 8, start: 0, end: 2}]);
  assert.equal(selected.start, 1); assert.equal(selected.end, 2);
});

test('selection clips the original bounded holds rather than extending them after rebasing', () => {
  const audio = [[2, 0, 0, 60, 100, 1], [8, 50, 0, 67, 1, 1],
    [16, 70, 0, 72, -1, 1], [17, 70, -1, 72, 1, 1], [18, NaN, 0, 72, 1, 1]];
  const selected = playback.fromSelection(audio, [], playback.timeline(audio), 8);
  assert.deepEqual(selected.audioEvents, [[2, 0, 0, 60, 10, 1], [8, 0, 0, 67, 1, 1]]);
  assert.equal(selected.end, 10);
  const boundedChord = playback.fromSelection([], [
    {start: 1795, duration: 20, pitches: [48, 52, 55], symbol: 'C', index: 2},
    {start: 1790, duration: 20, pitches: [], symbol: 'N.C.', index: 3},
  ], [{index: 8, start: 1798, end: 1799}], 8);
  assert.equal(boundedChord.chordEvents[0].duration, 2);
  assert.equal(boundedChord.end, 2);
});

const renderer = process.env.SCORE2ABC_TEST_RENDERER;
function engine() {
  const context = vm.createContext({});
  for (const name of ['abc2svg-1.js', 'toaudio-1.js']) {
    vm.runInContext(fs.readFileSync(path.join(renderer, name), 'utf8'), context);
  }
  return context;
}

test('written cursor follows tuplets, both tied heads, and the final rest', {skip: !renderer}, () => {
  const context = engine();
  const abc = 'X:1\nM:3/4\nL:1/4\nQ:1/4=60\nK:C\n(3C/2D/2E/2 [GB]2- | [GB] z2 |]\n';
  let sounding;
  new context.abc2svg.Abc({img_out: () => {}, get_abcmodel: (first, voices) => {
    const audio = new context.ToAudio(); audio.add(first, voices);
    sounding = Array.from(audio.clear(), event => Array.from(event));
  }}).tosvg('audio', abc);
  const firstDyad = abc.indexOf('[GB]'), secondDyad = abc.indexOf('[GB]', firstDyad + 1);
  const audioDyad = sounding.filter(event => event[0] === firstDyad);
  assert.equal(audioDyad.length, 2);
  assert.equal(audioDyad[0][4], 3); // Audio remains one sustained three-beat tie.
  const visual = playback.writtenEvents(context.abc2svg.Abc, context.ToAudio, context.abc2svg.C, abc);
  assert.deepEqual(playback.activeAt(visual, .5).map(e => e.index), [abc.indexOf('D/2')]);
  assert.deepEqual(playback.activeAt(visual, 2).map(e => e.index), [firstDyad]);
  assert.deepEqual(playback.activeAt(visual, 3.5).map(e => e.index), [secondDyad]);
  assert.deepEqual(playback.activeAt(visual, 5).map(e => e.index), [abc.indexOf('z2')]);
  assert.equal(visual.at(-1).end, 6);
});

test('tied dyad playback sustains each head once through a continuation and a chain', {skip: !renderer}, () => {
  const prefix = 'X:1\nM:5/4\nL:1/4\nQ:1/4=60\nK:C\n';
  for (const body of ['[Dc]-[Dc]4', '[Dc]-[Dc]-[Dc]3']) {
    const text = prefix + body + ' |]\n', first = text.indexOf('[Dc]');
    assert.deepEqual(sound(text).map(e => [e[0], e[1], e[3], e[4]]),
      [[first, 0, 62, 5], [first, 0, 72, 5]], body);
  }
});

test('tied dyads keep separate repeat occurrences', {skip: !renderer}, () => {
  const repeated = 'X:1\nM:3/4\nL:1/4\nQ:1/4=60\nK:C\n|: [Dc]-[Dc]2 :|\n';
  const first = repeated.indexOf('[Dc]');
  assert.deepEqual(sound(repeated).map(e => [e[0], e[1], e[3], e[4]]), [
    [first, 0, 62, 3], [first, 0, 72, 3], [first, 3, 62, 3], [first, 3, 72, 3],
  ]);
});

test('tie repair preserves supplied identities and is stable on repeated conversion', {skip: !renderer}, () => {
  const text = 'X:1\nM:5/4\nL:1/4\nQ:1/4=60\nK:C\n[Dc]-[Dc]4|\n';
  const context = engine(); let events;
  new context.abc2svg.Abc({img_out: () => {}, get_abcmodel: (first, voices) => {
    const notes = [];
    for (let symbol = first; symbol; symbol = symbol.ts_next) {
      if (symbol.type === context.abc2svg.C.NOTE) notes.push(...symbol.notes);
    }
    const before = notes.map(note => [note.midi, note.pit, note.dur, note.tie_ty]);
    notes[0].b40 = 777;
    const once = playback.audioEvents(context.ToAudio, first, voices, context.abc2svg.C);
    const twice = playback.audioEvents(context.ToAudio, first, voices, context.abc2svg.C);
    assert.deepEqual(twice, once);
    assert.equal(notes[0].b40, 777);
    assert.equal(notes[2].b40, 777);
    assert.notEqual(notes[1].b40, notes[0].b40);
    assert.equal(notes[3].b40, notes[1].b40);
    assert.deepEqual(notes.map(note => [note.midi, note.pit, note.dur, note.tie_ty]), before);
    events = once.filter(e => e[2] >= 0 && e[3] > 0 && e[5] > 0);
  }}).tosvg('reused-audio-model', text);
  assert.deepEqual(events.map(e => [e[1], e[3], e[4]]), [[0, 62, 5], [0, 72, 5]]);
});

test('explicit naturals cancel flat and sharp playback state', {skip: !renderer}, () => {
  const context = engine();
  const abc = 'X:1\nM:3/4\nL:1/4\nK:F\n_B/2 =B/2 ^F/2 =F/2 [=Bd] |\n';
  let sounding = [];
  new context.abc2svg.Abc({img_out: () => {}, get_abcmodel: (first, voices) => {
    const audio = new context.ToAudio(); audio.add(first, voices);
    sounding = Array.from(audio.clear(), event => Array.from(event));
  }}).tosvg('natural-cancellations', abc);
  assert.deepEqual(sounding.filter(event => event[2] >= 0 && event[3] > 0).map(event => event[3]),
    [70, 71, 66, 65, 71, 74]);
});

test('repeats revisit source symbols and tempo changes keep their timing', {skip: !renderer}, () => {
  const context = engine();
  const abc = 'X:1\nM:2/4\nL:1/4\nQ:1/4=60\nK:C\n|: C D |\nQ:1/4=120\nE F :|\n';
  const events = playback.writtenEvents(context.abc2svg.Abc, context.ToAudio, context.abc2svg.C, abc);
  assert.deepEqual(events.map(e => e.start), [0, 1, 2, 2.5, 3, 4, 5, 5.5]);
  assert.equal(events[0].index, events[4].index);
  assert.equal(events[3].index, events[7].index);
  assert.equal(events.at(-1).end, 6);
});

test('parsed unmatched later repeat blocks every audio path before ToAudio.add', {skip: !renderer}, () => {
  const context = engine();
  const abc = 'X:1\nM:3/4\nL:1/8\nK:C\nC6 |: D6 :| E6 | [1 F6 :| [2 G6 |]\n';
  let adds = 0, warning;
  class ForbiddenAudio {
    add() { adds++; throw new Error('ToAudio.add was called'); }
  }
  new context.abc2svg.Abc({img_out: () => {}, get_abcmodel: (first, voices) => {
    warning = playback.repeatWarning(first, context.abc2svg.C);
    assert.equal(playback.playbackWarning(first, context.abc2svg.C), warning);
    assert.throws(() => playback.audioEvents(ForbiddenAudio, first, voices,
      context.abc2svg.C), /Playback is disabled/);
  }}).tosvg('unsafe-render', abc);
  assert.match(warning, /Playback is disabled.*repeat route.*edit and save/);
  assert.throws(() => playback.writtenEvents(context.abc2svg.Abc, ForbiddenAudio,
    context.abc2svg.C, abc), /Playback is disabled/);
  assert.throws(() => playback.accompaniment(context.abc2svg.Abc, ForbiddenAudio,
    context.abc2svg.C, abc), /Playback is disabled/);
  assert.equal(adds, 0);
});

test('implicit, matched, double, and first/second ending repeats remain playable', {skip: !renderer}, () => {
  const context = engine();
  const header = 'X:1\nM:6/4\nL:1/4\nK:C\n';
  for (const body of [
    'C6 :| D6 |]',
    'C6 |: D6 :| E6 |: [1 F6 :| [2 G6 |]',
    '|: C6 [1 D6 :| [2 E6 |]',
    '|: C6 :|: D6 :|',
    'C6 :|: D6 :|: E6 :|', // Carrizal uses consecutive implicit/combined repeats.
  ]) {
    let warning;
    new context.abc2svg.Abc({img_out: () => {}, get_abcmodel: first => {
      warning = playback.repeatWarning(first, context.abc2svg.C);
      assert.equal(playback.playbackWarning(first, context.abc2svg.C), warning);
    }}).tosvg('safe-render', header + body + '\n');
    assert.equal(warning, null, body);
    assert.ok(playback.writtenEvents(context.abc2svg.Abc, context.ToAudio,
      context.abc2svg.C, header + body + '\n').length > 0, body);
  }
});

function backing(text) {
  const context = engine();
  return playback.accompaniment(context.abc2svg.Abc, context.ToAudio, context.abc2svg.C, text);
}

function sound(text) {
  const context = engine(); let result;
  new context.abc2svg.Abc({img_out: () => {}, get_abcmodel: (first, voices) => {
    result = playback.audioEvents(context.ToAudio, first, voices, context.abc2svg.C);
  }}).tosvg('performance', text);
  return result.filter(e => e[2] >= 0 && e[3] > 0 && e[5] > 0);
}

function performance(text) {
  const context = engine(); let audioEvents;
  new context.abc2svg.Abc({img_out() {}, get_abcmodel(first, voices) {
    audioEvents = playback.audioEvents(context.ToAudio, first, voices, context.abc2svg.C);
  }}).tosvg('selection-performance', text);
  const visualEvents = playback.writtenEvents(context.abc2svg.Abc, context.ToAudio, context.abc2svg.C, text);
  const chordEvents = playback.accompaniment(context.abc2svg.Abc, context.ToAudio, context.abc2svg.C, text).events;
  return {audioEvents, chordEvents, visualEvents};
}

test('selecting a tied continuation sounds both held heads once for their remaining duration', {skip: !renderer}, () => {
  const text = 'X:1\nM:4/4\nL:1/4\nQ:1/4=60\nK:C\n' +
    '"Dm"[Dc]2- [Dc]2 | z4 |\n';
  const first = text.indexOf('[Dc]'), continuation = text.indexOf('[Dc]', first + 1);
  const streams = performance(text);
  const before = JSON.stringify(streams);
  const selected = playback.fromSelection(streams.audioEvents, streams.chordEvents, streams.visualEvents, continuation);
  assert.equal(selected.start, 2); assert.equal(selected.end, 6);
  assert.deepEqual(selected.audioEvents.filter(event => event[3] > 0).map(event =>
    [event[0], event[1], event[3], event[4]]).sort((a, b) => a[2] - b[2]), [
    [first, 0, 62, 2], [first, 0, 72, 2],
  ]);
  assert.deepEqual(selected.chordEvents.map(event => [event.start, event.duration, event.symbol]), [[0, 2, 'Dm']]);
  assert.deepEqual(playback.activeAt(selected.visualEvents, 0).map(event => event.index), [continuation]);
  assert.equal(JSON.stringify(streams), before);
});

test('selecting an in-bar rest keeps its silence and active chord, then leaves the unmarked bar silent', {skip: !renderer}, () => {
  const text = 'X:1\nM:3/4\nL:1/4\nQ:1/4=60\nK:C\n"C"C z D | E z2 |]\n';
  const streams = performance(text), rest = text.indexOf(' z ') + 1;
  const selected = playback.fromSelection(streams.audioEvents, streams.chordEvents, streams.visualEvents, rest);
  assert.equal(selected.start, 1); assert.equal(selected.end, 5);
  assert.deepEqual(selected.audioEvents.filter(event => event[3] > 0).map(event =>
    [event[1], event[3], event[4]]), [[1, 62, 1], [2, 64, 1]]);
  assert.deepEqual(selected.chordEvents.map(event => [event.start, event.duration, event.symbol]), [[0, 2, 'C']]);
  assert.deepEqual(playback.activeAt(selected.visualEvents, .5).map(event => event.index), [rest]);
  assert.deepEqual(playback.activeAt(selected.visualEvents, 4).map(event => event.index), [text.indexOf('z2')]);
  const silentBar = playback.fromSelection(streams.audioEvents, streams.chordEvents,
    streams.visualEvents, text.indexOf('E z2'));
  assert.deepEqual(silentBar.chordEvents, []);
});

test('selection in ordinary repeats keeps the first visit, repeat return and second ending', {skip: !renderer}, () => {
  const text = 'X:1\nM:2/4\nL:1/4\nQ:1/4=60\nK:C\n|: "C"C D |1 "G"E z :|2 F G |]\n';
  const streams = performance(text), index = text.indexOf('D |1');
  const selected = playback.fromSelection(streams.audioEvents, streams.chordEvents, streams.visualEvents, index);
  assert.equal(selected.start, 1); assert.equal(selected.end, 7);
  assert.deepEqual(selected.audioEvents.filter(event => event[3] > 0).map(event =>
    [event[1], event[3]]), [[0, 62], [1, 64], [3, 60], [4, 62], [5, 65], [6, 67]]);
  assert.deepEqual(selected.visualEvents.map(event => [event.start, event.index]),
    [[0, index], [1, text.indexOf('E z')], [2, text.indexOf('z :|')],
      [3, text.indexOf('C D')], [4, index], [5, text.indexOf('F G')], [6, text.indexOf('G |]')]]);
  assert.deepEqual(selected.chordEvents.map(event => [event.start, event.duration, event.symbol]),
    [[0, 1, 'C'], [1, 2, 'G'], [3, 2, 'C']]);
});

const repeatedSections = 'X:1\nP:AB AC DE DF AC DF\nM:3/4\nL:1/4\nQ:1/4=60\nK:C\n' +
  'P:A\n|: "C"C3 |\nP:B\n[1 "G"D3 :|\nP:C\n[2 "Am"E3 ||\n' +
  'P:D\n|: "F"F3 |\nP:E\n[1 "Dm"G3 :|\nP:F\n[2 "G7"A3 |]\n';

test('selection keeps the complete explicit part-route suffix and rejects an unperformed part', {skip: !renderer}, () => {
  const streams = performance(repeatedSections), index = repeatedSections.indexOf('E3');
  const selected = playback.fromSelection(streams.audioEvents, streams.chordEvents, streams.visualEvents, index);
  assert.equal(selected.start, 9); assert.equal(selected.end, 27);
  assert.deepEqual(selected.audioEvents.filter(event => event[3] > 0).map(event =>
    [event[1], event[3]]), [64, 65, 67, 65, 69, 60, 64, 65, 69].map((pitch, i) => [i * 3, pitch]));
  assert.deepEqual(selected.chordEvents.map(event => [event.start, event.symbol]),
    ['Am', 'F', 'Dm', 'F', 'G7', 'C', 'Am', 'F', 'G7'].map((symbol, i) => [i * 3, symbol]));
  assert.deepEqual(selected.visualEvents.filter(event => event.index === index).map(event => event.start), [0, 18]);
  const text = 'X:1\nP:AA\nM:3/4\nL:1/4\nQ:1/4=60\nK:C\nP:A\nC3 |\nP:B\nD3 |]\n';
  const unplayed = performance(text);
  assert.equal(playback.fromSelection(unplayed.audioEvents, unplayed.chordEvents,
    unplayed.visualEvents, text.indexOf('D3')), null);
});

test('section order owns repeated bars and both endings, including the final return without repeats', {skip: !renderer}, () => {
  const text = repeatedSections;
  const notes = ['C3', 'D3', 'C3', 'E3', 'F3', 'G3', 'F3', 'A3', 'C3', 'E3', 'F3', 'A3'];
  const indices = notes.map(note => text.indexOf(note));
  const pitches = [60, 62, 60, 64, 65, 67, 65, 69, 60, 64, 65, 69];
  const audio = sound(text);
  assert.deepEqual(audio.map(e => [e[0], e[1], e[3], e[4]]),
    indices.map((index, i) => [index, i * 3, pitches[i], 3]));
  const context = engine(), cursor = playback.writtenEvents(context.abc2svg.Abc,
    context.ToAudio, context.abc2svg.C, text);
  assert.deepEqual(cursor, indices.map((index, i) => ({index, start: i * 3, end: (i + 1) * 3})));
  const chords = backing(text);
  assert.deepEqual(chords.events.map(e => [e.index, e.start, e.duration, e.symbol]),
    ['C', 'G', 'C', 'Am', 'F', 'Dm', 'F', 'G7', 'C', 'Am', 'F', 'G7']
      .map((symbol, i) => [indices[i], i * 3, 3, symbol]));
  assert.deepEqual(chords.warnings, []);
});

test('fractional-tempo section returns retain full-bar chords across a combined repeat boundary', {skip: !renderer}, () => {
  const text = 'X:1\nP:IABABI\nM:3/4\nL:1/4\nQ:1/4=60\nK:C\n' +
    'P:I\n"F"C3 |:\nP:A\nQ:1/4=190\n"Dm"D3 :|:\nP:B\n"G"E3 | "C"F3 :|\n';
  const context = engine(), cursor = playback.writtenEvents(context.abc2svg.Abc,
    context.ToAudio, context.abc2svg.C, text);
  const audio = sound(text), chords = backing(text);
  assert.deepEqual(audio.map(e => [e[0], e[1]]), cursor.map(e => [e.index, e.start]));
  assert.deepEqual(chords.events.map(e => [e.index, e.start, e.symbol]),
    ['F', 'Dm', 'G', 'C', 'Dm', 'G', 'C', 'F']
      .map((symbol, i) => [cursor[i].index, cursor[i].start, symbol]));
  assert.deepEqual(chords.events.map(e => e.duration),
    cursor.map((e, i) => (cursor[i + 1]?.start ?? e.end) - e.start));
  assert.deepEqual(chords.warnings, []);
});

test('section conversion restores exact repeat and ending properties before engraving on success or error', {skip: !renderer}, () => {
  const text = repeatedSections, baseline = [], plain = engine();
  new plain.abc2svg.Abc({img_out: svg => baseline.push(svg)}).tosvg('navigation-bars', text);
  for (const fail of [false, true]) {
    const context = engine(), rendered = []; let calls = 0;
    function CheckedAudio() {
      const audio = new context.ToAudio();
      return {add(first, voices) {
        calls++;
        for (let s = first; s; s = s.ts_next) if (s.type === context.abc2svg.C.BAR) {
          assert.equal(s.bar_type, '|');
          assert.equal(s.text, '');
        }
        if (fail) throw new Error('conversion failed');
        audio.add(first, voices);
      }, clear: () => audio.clear()};
    }
    new context.abc2svg.Abc({img_out: svg => rendered.push(svg), get_abcmodel(first, voices) {
      const bars = [];
      for (let s = first; s; s = s.ts_next) if (s.type === context.abc2svg.C.BAR) bars.push(s);
      const properties = () => bars.map(s => ({
        bar_type: Object.getOwnPropertyDescriptor(s, 'bar_type'),
        text: Object.getOwnPropertyDescriptor(s, 'text'),
      }));
      const before = properties();
      assert.ok(before.some(bar => bar.bar_type.value.includes(':')));
      assert.ok(before.some(bar => bar.text?.value === '1'));
      assert.ok(before.some(bar => bar.text?.value === '2'));
      assert.ok(before.some(bar => !bar.text));
      if (fail) assert.throws(() => playback.audioEvents(CheckedAudio, first, voices,
        context.abc2svg.C), /conversion failed/);
      else playback.audioEvents(CheckedAudio, first, voices, context.abc2svg.C);
      assert.deepEqual(properties(), before);
    }}).tosvg('navigation-bars', text);
    assert.equal(calls, 1);
    assert.deepEqual(rendered, baseline);
  }
});

test('valid section orders bypass unsafe local navigation while ordinary repeats still unfold', {skip: !renderer}, () => {
  const ordinary = 'X:1\nM:3/4\nL:1/4\nQ:1/4=60\nK:C\n|: C3 [1 D3 :| [2 E3 |]\n';
  assert.deepEqual(sound(ordinary).map(e => [e[1], e[3]]), [[0, 60], [3, 62], [6, 60], [9, 64]]);
  const text = 'X:1\nP:ABA\nM:3/4\nL:1/4\nQ:1/4=60\nK:C\n' +
    'P:A\n"C"C3 |: "C"C3 :|\nP:B\n"G"D3 :|\n';
  const context = engine();
  new context.abc2svg.Abc({img_out() {}, get_abcmodel(first) {
    assert.match(playback.repeatWarning(first, context.abc2svg.C), /Playback is disabled/);
    assert.equal(playback.playbackWarning(first, context.abc2svg.C), null);
  }}).tosvg('explicit-unsafe-navigation', text);
  assert.deepEqual(sound(text).map(e => [e[1], e[3]]),
    [[0, 60], [3, 60], [6, 62], [9, 60], [12, 60]]);
  assert.deepEqual(playback.writtenEvents(context.abc2svg.Abc, context.ToAudio, context.abc2svg.C, text)
    .map(e => [e.start, e.end]), [[0, 3], [3, 6], [6, 9], [9, 12], [12, 15]]);
  assert.deepEqual(backing(text).events.map(e => [e.start, e.duration, e.symbol]),
    [[0, 3, 'C'], [3, 3, 'C'], [6, 3, 'G'], [9, 3, 'C'], [12, 3, 'C']]);
});

test('section order shares melody, ties, cursor, key changes and bar-limited chords', {skip: !renderer}, () => {
  const text = 'X:1\nP:AABBCCABCC\nM:3/4\nL:1/4\nQ:1/4=60\nK:F\n' +
    'P:A\n"C"C2- C | z3 ||\nP:B\n"Gm"B3 | d3 ||\nP:C\n[K:D]"D"F3 | z3 |]\n';
  const audio = sound(text), firstC = text.indexOf('C2-'), continuation = text.indexOf(' C |') + 1;
  assert.deepEqual(audio.map(e => [e[1], e[3], e[4]]), [
    [0,60,3], [6,60,3], [12,70,3], [15,74,3], [18,70,3], [21,74,3],
    [24,66,3], [30,66,3], [36,60,3], [42,70,3], [45,74,3], [48,66,3], [54,66,3],
  ]);
  assert.deepEqual(audio.filter(e => e[3] === 60).map(e => e[0]), [firstC, firstC, firstC]);
  const context = engine(), cursor = playback.writtenEvents(context.abc2svg.Abc,
    context.ToAudio, context.abc2svg.C, text);
  assert.equal(cursor.at(-1).end, 60); // Includes the final three-beat rest.
  assert.deepEqual(playback.activeAt(cursor, 2.5).map(e => e.index), [continuation]);
  assert.deepEqual(playback.activeAt(cursor, 36.5).map(e => e.index), [firstC]);
  const chords = backing(text);
  assert.deepEqual(chords.events.map(e => [e.start,e.duration,e.symbol]), [
    [0,3,'C'], [6,3,'C'], [12,3,'Gm'], [18,3,'Gm'], [24,3,'D'],
    [30,3,'D'], [36,3,'C'], [42,3,'Gm'], [48,3,'D'], [54,3,'D'],
  ]);
  assert.deepEqual(chords.warnings, []);
});

test('section returns retain their own tempo rather than the preceding section tempo', {skip: !renderer}, () => {
  const text = 'X:1\nP:ABA\nM:3/4\nL:1/4\nQ:1/4=60\nK:C\n' +
    'P:A\nC3 ||\nP:B\nQ:1/4=120\nD3 |]\n';
  const audio = sound(text);
  assert.deepEqual(audio.map(e => [e[1],e[3],e[4]]), [[0,60,3],[3,62,1.5],[4.5,60,3]]);
});

test('unsupported section orders stop every audio path before conversion', {skip: !renderer}, () => {
  const base = 'X:1\nP:AA\nM:3/4\nL:1/4\nK:C\nP:A\nC3 |]\n';
  for (const text of [
    base.replace('P:AA', 'P:AZ'),
    base.replace('P:AA', 'P:(A)2'),
    base.replace('P:AA', 'P:' + 'A'.repeat(129)),
    base.replace('P:AA', 'P:AZ').replace('C3 |]', 'C3 |: D3 :| E3 :|'),
    base.replace('P:A\n', 'P:AB\n'),
    base.replace('C3 |]', 'C3 |\nP:A\nD3 |]'),
    base.replace('P:A\n', ''),
  ]) {
    const context = engine(); let calls = 0;
    class ForbiddenAudio { add() { calls++; throw new Error('Unexpected conversion'); } }
    new context.abc2svg.Abc({img_out: () => {}, get_abcmodel: (first,voices) => {
      assert.match(playback.playbackWarning(first,context.abc2svg.C), /Playback is disabled/);
      if (text.includes('P:AZ')) {
        assert.match(playback.playbackWarning(first,context.abc2svg.C), /missing section label/);
      }
      assert.throws(() => playback.audioEvents(ForbiddenAudio,first,voices,context.abc2svg.C), /Playback is disabled/);
    }}).tosvg('invalid-order',text);
    assert.throws(() => playback.writtenEvents(context.abc2svg.Abc,ForbiddenAudio,context.abc2svg.C,text), /Playback is disabled/);
    assert.throws(() => playback.accompaniment(context.abc2svg.Abc,ForbiddenAudio,context.abc2svg.C,text), /Playback is disabled/);
    assert.equal(calls,0);
  }
});

test('section expansion is bounded and rejects held notes crossing section boundaries', {skip: !renderer}, () => {
  assert.throws(() => sound('X:1\nP:AB\nM:3/4\nL:1/4\nK:C\nP:A\nC3- |\nP:B\nC3 |]\n'), /Playback is disabled/);
  assert.throws(() => sound('X:1\nP:' + 'A'.repeat(128) +
    '\nM:3/4\nL:1/4\nQ:1/4=1\nK:C\nP:A\nC3 |]\n'), /30-minute/);
});
const header = 'X:1\nM:4/4\nL:1/4\nQ:1/4=60\nK:C\n';

test('accompaniment uses edited harmonies at note/rest/tied onsets, without pickup chords', {skip: !renderer}, () => {
  const abc = header + 'C "Gm6"D "D7"z E- | "Cm6"E "N.C."z "^instruction"z2 |';
  const {events, warnings} = backing(abc);
  assert.deepEqual(events.map(e => [e.start, e.duration, e.symbol]), [
    [1, 1, 'Gm6'], [2, 2, 'D7'], [4, 1, 'Cm6']
  ]);
  assert.deepEqual(events[0].pitches, [55, 58, 62, 64]);
  assert.deepEqual(warnings, []);
  assert.equal(backing(abc.replace('Gm6', 'Dm')).events[0].symbol, 'Dm');
  assert.deepEqual(backing(abc.replace('Gm6', 'Dm')).events[0].pitches, [50, 53, 57]);
});

test('unsupported harmony silences previous chord, while annotations do not change it', {skip: !renderer}, () => {
  const {events, warnings} = backing(header + '"G"C "^louder"D "Gwhatever"E "A7"F|');
  assert.deepEqual(events.map(e => [e.start, e.duration, e.symbol]), [[0, 2, 'G'], [3, 1, 'A7']]);
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /Gwhatever/);
});

test('repeats leave unlabeled bars silent and tempo before the bar stays aligned', {skip: !renderer}, () => {
  const abc = header + '"G"C |: D E | "D7"F G :|\nQ:1/4=120\nA B |';
  const {events, warnings} = backing(abc);
  assert.deepEqual(events.map(e => [e.start, e.duration, e.symbol]), [
    [0, 1, 'G'], [3, 2, 'D7'], [7, 2, 'D7']
  ]);
  assert.deepEqual(warnings, []);
  const sustained = backing(header + '"Dm"C2\nQ:1/4=120\nD2 | E4 |');
  const context = engine();
  const markers = playback.writtenEvents(context.abc2svg.Abc, context.ToAudio,
    context.abc2svg.C, header + '\"Dm\"C2\nQ:1/4=120\nD2 | E4 |');
  // Follow the installed ToAudio clock even across its mid-system tempo boundary.
  assert.deepEqual(sustained.events.map(e => [e.start, e.duration]),
    [[0, markers.find(event => event.index === (header + '"Dm"C2\nQ:1/4=120\nD2 | E4 |').indexOf('E4')).start]]);
});

test('repeat into an unharmonized pickup clears the final harmony', {skip: !renderer}, () => {
  const {events} = backing(header + '|: C "Dm"D E F :|');
  assert.deepEqual(events.map(e => [e.start, e.duration, e.symbol]), [[1, 3, 'Dm'], [5, 3, 'Dm']]);
});

test('known chord families and slash bass use bounded lower-register voicing', {skip: !renderer}, () => {
  const symbols = ['Gm6', 'Dm', 'Cm6', 'D7', 'Gm', 'A7', 'D', 'Em', 'G',
    'Cmaj7', 'Cm7', 'Cdim', 'Cdim7', 'Caug', 'Csus2', 'Csus4', 'D/F#'];
  const {events, warnings} = backing(header + symbols.map(symbol => `"${symbol}"C`).join(' ') + '|');
  assert.equal(events.length, symbols.length);
  assert.deepEqual(events.map(e => e.symbol), symbols);
  assert.deepEqual(events.at(-1).pitches, [42, 50, 54, 57]);
  assert.deepEqual(events[9].pitches, [48, 52, 55, 59]);
  assert.deepEqual(events[11].pitches, [48, 51, 54]);
  assert.deepEqual(warnings, []);
});

test('simultaneous conflicting harmonies silence instead of guessing', {skip: !renderer}, () => {
  const {events, warnings} = backing(header + '[V:1] "C"C4|\n[V:2] "Dm"E4|');
  assert.deepEqual(events, []);
  assert.equal(warnings.length, 1);
});

test('initial BPM uses positive time-zero beat units and ignores later tempo changes', () => {
  const constants = {TEMPO: 14, BLEN: 1536};
  const chain = symbols => symbols.reduceRight((next, symbol) => ({...symbol, ts_next: next}), null);
  assert.equal(playback.initialBpm(null, constants), 120);
  assert.equal(playback.initialBpm(chain([
    {type:14,time:0,tempo:60,tempo_notes:[384,192]},
    {type:8,time:0}, {type:14,time:384,tempo:240,tempo_notes:[384]}
  ]), constants), 90);
  assert.equal(playback.initialBpm(chain([
    {type:14,time:0,tempo:NaN,tempo_notes:[384]},
    {type:14,time:0,tempo:-20,tempo_notes:[384]},
    {type:14,time:0,tempo:20,tempo_notes:[]},
    {type:14,time:10,tempo:80,tempo_notes:[384]}
  ]), constants), 120);
});

test('initial BPM matches installed renderer no-Q, beat-unit, text, and later-Q semantics', {skip: !renderer}, () => {
  const cases = [
    ['', 120], ['Q:90\n', 120], ['Q:"Allegro"\n', 120],
    ['Q:1/8=120\n', 60], ['Q:3/8=60\n', 90], ['Q:1/4 1/8=60\n', 90],
    ['Q:1/4=ca. 72\n', 72], ['Q:1/4=1/8\n', 120], ['Q:1/4=60\n', 60]
  ];
  for (const [q, expected] of cases) {
    const context = engine(); let bpm;
    new context.abc2svg.Abc({img_out(){}, errbld(){}, get_abcmodel(first) {
      bpm = playback.initialBpm(first, context.abc2svg.C);
    }}).tosvg('bpm', 'X:1\nM:4/4\nL:1/4\n' + q + 'K:C\nC D |\nQ:1/4=200\nE F|');
    assert.equal(bpm, expected, `initial quarter BPM for ${JSON.stringify(q)}`);
  }
});

test('written bars stop accompaniment at pickups, tied continuations, rests, and mid-bar changes', {skip: !renderer}, () => {
  const start = 'X:1\nM:2/4\nL:1/4\nQ:1/4=60\nK:C\n';
  const cases = [
    ['"C"C | D2 |', [[0, 1, 'C']]],
    ['"C"C2- | C2 |', [[0, 2, 'C']]],
    ['"C"z2 | z2 |', [[0, 2, 'C']]],
    ['"C"C "Dm"D | E2 |', [[0, 1, 'C'], [1, 1, 'Dm']]],
    ['|: "C"C2 |1 "G"D2 :|2 E2 |', [[0, 2, 'C'], [2, 2, 'G'], [4, 2, 'C']]],
    ['"C"C |: D2 | E2 :|', [[0, 1, 'C']]],
  ];
  for (const [body, expected] of cases) {
    const result = backing(start + body);
    assert.deepEqual(result.events.map(event => [event.start, event.duration, event.symbol]), expected, body);
    assert.deepEqual(result.warnings, []);
  }
});

test('a barline stops harmony even with no new onset and a different voice still sounding', {skip: !renderer}, () => {
  const abc = 'X:1\nM:2/4\nL:1/4\nQ:1/4=60\nK:C\n[V:1] "C"C2|\n[V:2] E4|';
  const context = engine();
  const visual = playback.writtenEvents(context.abc2svg.Abc, context.ToAudio, context.abc2svg.C, abc);
  assert.deepEqual(visual.map(event => event.start), [0, 0]);
  assert.equal(Math.max(...visual.map(event => event.end)), 4);
  assert.deepEqual(backing(abc).events.map(event => [event.start, event.duration, event.symbol]), [[0, 2, 'C']]);
});

test('private bar markers preserve genuine untied event timing across tempo, grace, ties, and repeats', {skip: !renderer}, () => {
  const cases = [
    '"C"C2- | C2 "Dm"D2|',
    '"C"C2\nQ:1/4=120\nD2 | E4 |',
    '|: "C"C2 |1 "G"D2 :|2 E2 |',
    '"C"{d}C D | "G"E {fg}F |',
    '[V:1] "C"C2|\n[V:2] E4|',
  ];
  for (const body of cases) {
    const abc = header + body, context = engine(); let captured = [];
    function CapturedAudio() {
      const audio = new context.ToAudio();
      return {add: (...args) => audio.add(...args), clear() {
        const events = audio.clear() || [];
        captured.push(...Array.from(events, event => Array.from(event)).filter(event => event[0] <= abc.length));
        return events;
      }};
    }
    playback.accompaniment(context.abc2svg.Abc, CapturedAudio, context.abc2svg.C, abc);
    const expected = playback.writtenEvents(context.abc2svg.Abc, context.ToAudio, context.abc2svg.C, abc);
    assert.deepEqual(playback.timeline(captured), expected, body);
  }
});
