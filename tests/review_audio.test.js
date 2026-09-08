const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function harness({pendingResume = false} = {}) {
  const elements = new Map(), voices = [], frames = new Map();
  let resolveResume, nextFrame = 1;
  const resumed = pendingResume ? new Promise(resolve => { resolveResume = resolve; }) : Promise.resolve();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      value: '', disabled: false, textContent: '', style: {},
      classList: {toggle() {}, remove() {}},
      addEventListener() {}, querySelectorAll: () => [], replaceChildren() {}, append() {},
    });
    return elements.get(id);
  }
  class AudioContext {
    currentTime = 12;
    destination = {};
    resume() { return resumed; }
    createOscillator() {
      const voice = {
        frequency: {value: 0}, starts: [], stops: [],
        connect(gain) { this.gain = gain; }, disconnect() {},
        start(time) { this.starts.push(time); }, stop(time) { this.stops.push(time); },
      };
      voices.push(voice); return voice;
    }
    createGain() {
      const changes = [];
      return {
        changes, connect() {}, disconnect() {},
        gain: {
          setValueAtTime(value, time) { changes.push({value, time}); },
          linearRampToValueAtTime(value, time) { changes.push({value, time}); },
        },
      };
    }
  }
  const context = vm.createContext({
    document: {getElementById: element, addEventListener() {}},
    window: {AudioContext, addEventListener() {}},
    setTimeout: () => 1, clearTimeout() {}, setInterval() {},
    requestAnimationFrame(callback) { const id = nextFrame++; frames.set(id, callback); return id; },
    cancelAnimationFrame(id) { frames.delete(id); },
    fetch: () => new Promise(() => {}),
    Date: class extends Date { static now() { return 1000; } },
    URL, location: {href: 'http://127.0.0.1/'},
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../score2abc/review_ui.js'), 'utf8'), context);
  const run = code => vm.runInContext(code, context);
  run(`desk.work = {slug: 'sample'};
    desk.audioEvents = [[0, 0, 0, 72, 1, 1], [4, 1, 0, 74, 1, 1]];
    desk.chordEvents = [{start: 0, duration: 1, pitches: [48,52,55]},
      {start: 1, duration: 1, pitches: [50,53,57]}];`);
  return {run, voices, frames, element, resolveResume: () => resolveResume()};
}

function near(actual, expected) {
  assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} differs from ${expected}`);
}

test('melody and quieter chord voices use one AudioContext start time', async () => {
  const h = harness(); await h.run('play()');
  assert.equal(h.voices.length, 8);
  const melody = h.voices.filter(voice => voice.type === 'triangle');
  const chords = h.voices.filter(voice => voice.type === 'sine');
  assert.equal(melody.length, 2); assert.equal(chords.length, 6);
  for (const [index, onset] of [12.1, 13.1].entries()) {
    near(melody[index].starts[0], onset);
    const group = chords.filter(voice => Math.abs(voice.starts[0] - onset) < 1e-10);
    assert.equal(group.length, 3);
    const level = voice => Math.max(...voice.gain.changes.map(change => change.value));
    near(group.reduce((sum, voice) => sum + level(voice), 0), .045);
    near(level(melody[index]), .075);
    assert.ok(group.every(voice => level(voice) < level(melody[index])));
    group.forEach(voice => near(voice.stops[0], onset + 1.01));
  }
  near(melody[0].frequency.value, 440 * 2 ** ((72 - 69) / 12));
  assert.equal(h.run('desk.playing'), true); assert.equal(h.frames.size, 1);
});

for (const action of ['stopPlayback()', 'setDirty()']) {
  test(`${action} stops all melody and accompaniment voices`, async () => {
    const h = harness(); await h.run('play()'); h.run(action);
    assert.equal(h.voices.length, 8);
    assert.ok(h.voices.every(voice => voice.stops.length === 2 && voice.stops[1] === undefined));
    assert.equal(h.run('desk.oscillators.length'), 0);
    assert.equal(h.run('desk.playing'), false); assert.equal(h.frames.size, 0);
    assert.equal(h.element('stop').disabled, true);
    if (action === 'setDirty()') {
      assert.equal(h.run('desk.dirty'), true); assert.equal(h.element('play').disabled, true);
    }
  });
  test(`${action} cancels pending resume without creating stale voices`, async () => {
    const h = harness({pendingResume: true}); const playing = h.run('play()');
    assert.equal(h.voices.length, 0); h.run(action); h.resolveResume(); await playing;
    assert.equal(h.voices.length, 0); assert.equal(h.frames.size, 0);
    assert.equal(h.run('desk.playing'), false);
  });
}
