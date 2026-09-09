// Validate notation using the same local abc2svg parser as the browser.
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const {abc} = JSON.parse(fs.readFileSync(0, 'utf8'));
const errors = [], warnings = [];
let note_count = 0, tunes = 0;
for (const field of ['X', 'M', 'K']) {
  if (!new RegExp('^' + field + ':\\s*\\S+', 'm').test(abc)) errors.push('Missing ' + field + ' header.');
}
if (/^(?:M|K):\s*\?\s*$/m.test(abc)) errors.push('Confirm meter and key from the manuscript.');
if ((abc.match(/^X:/gm) || []).length !== 1) errors.push('Exactly one tune is required.');
if (/^\s*(?:%%|I:)\s*(?:begin\w*|end\w*|include|abc-include|abcm2ps|ss-pref|js|javascript|ps|postscript|svg|html)\b/im.test(abc) || /<\s*\/?\s*(?:script|svg|html|iframe)\b/i.test(abc)) {
  errors.push('Executable, embedded markup and include directives are not allowed.');
}
if (!errors.length) {
  try {
    const context = vm.createContext({});
    vm.runInContext(fs.readFileSync(path.join(process.argv[2], 'abc2svg-1.js'), 'utf8'), context);
    const parser = new context.abc2svg.Abc({
      img_out: () => {},
      read_file: () => {errors.push('Includes are not allowed.'); return '';},
      errbld: (severity, message) => (severity === 'warn' ? warnings : errors).push(String(message)),
      get_abcmodel: (symbol) => {
        tunes++;
        for (let s = symbol; s; s = s.ts_next) {
          if (s.type === context.abc2svg.C.NOTE) note_count += (s.nhd || 0) + 1;
        }
      }
    });
    parser.tosvg('review.abc', abc);
    if (tunes !== 1) errors.push('The renderer must parse exactly one tune.');
  } catch (error) { errors.push('Renderer rejected notation: ' + error.message); }
}
process.stdout.write(JSON.stringify({valid: errors.length === 0, errors, warnings, note_count}));
