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
