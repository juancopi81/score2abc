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

test('repeats revisit source symbols and tempo changes keep their timing', {skip: !renderer}, () => {
  const context = engine();
  const abc = 'X:1\nM:2/4\nL:1/4\nQ:1/4=60\nK:C\n|: C D |\nQ:1/4=120\nE F :|\n';
  const events = playback.writtenEvents(context.abc2svg.Abc, context.ToAudio, context.abc2svg.C, abc);
  assert.deepEqual(events.map(e => e.start), [0, 1, 2, 2.5, 3, 4, 5, 5.5]);
  assert.equal(events[0].index, events[4].index);
  assert.equal(events[3].index, events[7].index);
  assert.equal(events.at(-1).end, 6);
});

function backing(text) {
  const context = engine();
  return playback.accompaniment(context.abc2svg.Abc, context.ToAudio, context.abc2svg.C, text);
}
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
