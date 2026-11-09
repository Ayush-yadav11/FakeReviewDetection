// The page's detector: the logistic regression from train.py, scored in the browser.
// Same steps as train.py's pipeline: drop punctuation, split on whitespace, drop English
// stopwords (case-insensitive, case kept), count, TF-IDF with the trained idf, scale the
// vector to length one, add the learned weights. model.json holds the vocabulary, idf and
// weights; data/samples.json holds held-out reviews from the test quarter.

const PUNCT = new Set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~');
const SHORT = {
  'Clothing Shoes and Jewelry': 'Clothing', 'Tools and Home Improvement': 'Tools', 'Home and Kitchen': 'Home',
  'Sports and Outdoors': 'Sports', 'Movies and TV': 'Movies', 'Toys and Games': 'Toys', 'Kindle Store': 'Kindle',
  'Pet Supplies': 'Pets', Books: 'Books', Electronics: 'Electronics',
};
const $ = (id) => document.getElementById(id);
const clean = (w) => [...w].filter((c) => !PUNCT.has(c)).join('');
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pct = (x) => `${Math.round(x * 100)}%`;
const signed = (x) => `${x < 0 ? '−' : '+'}${Math.abs(x).toFixed(2)}`;

function makeModel(m) {
  const index = new Map(m.terms.map((t, i) => [t, i]));
  const stop = new Set(m.stopwords);
  return function score(text) {
    // chunks keep the review's own spacing, so the words can be marked in place
    const chunks = text.split(/(\s+)/);
    const counts = new Map();
    const kept = [];
    let words = 0;
    for (const chunk of chunks) {
      if (!chunk || /^\s+$/.test(chunk)) continue;
      const w = clean(chunk);
      if (!w) continue;
      words += 1;
      if (stop.has(w.toLowerCase())) continue;
      kept.push(w);
      if (index.has(w)) counts.set(w, (counts.get(w) || 0) + 1);
    }
    let norm = 0;
    const x = new Map();
    for (const [w, n] of counts) { const v = n * m.idf[index.get(w)]; x.set(w, v); norm += v * v; }
    norm = Math.sqrt(norm) || 1;
    const contrib = new Map();
    let z = m.intercept;
    for (const [w, v] of x) { const c = (v / norm) * m.coef[index.get(w)]; contrib.set(w, c); z += c; }
    const pOr = 1 / (1 + Math.exp(-z));
    return { pOr, contrib, chunks, words, kept: kept.length };
  };
}

// The open pane shows whole rows only: a bar row that would be sliced by the window's foot is
// hidden, so the pane ends on whole rows at every width (it is laid out again on resize).
function fitBars() {
  const bars = $('det-bars');
  const pane = bars && bars.closest('.app__pane');
  if (!pane) return;
  const rows = [...bars.children];
  rows.forEach((row) => { row.hidden = false; });
  const limit = pane.getBoundingClientRect().bottom - 24;
  rows.forEach((row) => { if (row.getBoundingClientRect().bottom > limit) row.hidden = true; });
}
let fitTimer;
window.addEventListener('resize', () => { clearTimeout(fitTimer); fitTimer = setTimeout(fitBars, 120); });

function render(result, meta) {
  const { pOr, contrib, chunks, words, kept } = result;
  const verdict = $('det-verdict');
  const split = $('det-split');
  const text = $('det-text');
  const bars = $('det-bars');
  if (!words) {
    verdict.textContent = 'Paste a review to score it';
    verdict.dataset.v = '';
    split.style.setProperty('--cg', '50%');
    text.textContent = '';
    bars.innerHTML = '';
    $('det-meta').textContent = meta || 'Nothing to score yet';
    return;
  }
  const isOr = pOr >= 0.5;
  verdict.dataset.v = isOr ? 'OR' : 'CG';
  verdict.textContent = isOr ? `Original, ${pct(pOr)} sure` : `Generated, ${pct(1 - pOr)} sure`;
  split.style.setProperty('--cg', pct(1 - pOr));
  const max = Math.max(0.05, ...[...contrib.values()].map(Math.abs));
  text.innerHTML = chunks.map((chunk) => {
    if (!chunk) return '';
    if (/^\s+$/.test(chunk)) return ' ';
    const c = contrib.get(clean(chunk));
    if (c === undefined || Math.abs(c) < 0.02) return esc(chunk);
    const a = (0.18 + 0.5 * Math.min(1, Math.abs(c) / max)).toFixed(2);
    return `<mark data-v="${c < 0 ? 'CG' : 'OR'}" style="--a:${a}">${esc(chunk)}</mark>`;
  }).join('');
  const top = [...contrib.entries()].sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 10);
  bars.innerHTML = top.map(([w, c]) => {
    const width = Math.max(4, Math.round((Math.abs(c) / max) * 100));
    return `<div><span>${esc(w)}</span><i class="${c < 0 ? 'is-cg' : 'is-or'}" style="--w:${width}%"></i><em>${signed(c)}</em></div>`;
  }).join('');
  fitBars();
  $('det-meta').textContent = meta || `${kept} of ${words} words kept`;
}

async function main() {
  const [model, data] = await Promise.all([
    fetch('model.json').then((r) => r.json()),
    fetch('data/samples.json').then((r) => r.json()),
  ]);
  const score = makeModel(model);
  const input = $('det-input');
  const list = $('det-list');

  // the list: right answers by category with the four misses mixed in
  const wrong = data.samples.filter((s) => s.pred !== s.label);
  const right = data.samples.filter((s) => s.pred === s.label);
  const first = right.findIndex((s) => s.category === 'Tools and Home Improvement' && s.label === 'CG');
  const ordered = first > -1 ? [right[first], ...right.filter((_, i) => i !== first)] : right;
  [3, 7, 12, 17].forEach((at, i) => wrong[i] && ordered.splice(at, 0, wrong[i]));
  ordered.length = Math.min(ordered.length, 11);

  list.innerHTML = ordered.map((s, i) => {
    const sure = s.pred === 'OR' ? s.p_or : 1 - s.p_or;
    const verdict = `${s.pred === 'OR' ? 'Original' : 'Generated'} ${pct(sure)}`;
    const ok = s.pred === s.label;
    const short = SHORT[s.category] || s.category;
    const initials = short.slice(0, 2).toUpperCase();
    const flat = s.text.replace(/\s+/g, ' ').trim();
    const snippet = flat.length > 64 ? `${flat.slice(0, 64).replace(/\s+\S*$/, '')}…` : flat;
    return `<li class="ui-win__row${i === 0 ? ' is-sel' : ''}" data-row data-i="${data.samples.indexOf(s)}" tabindex="0" role="button" aria-label="${esc(short)} review, model says ${verdict}, ${ok ? 'right' : 'wrong'}">`
      + `<span class="ui-avatar" data-tone="${(i % 5) + 1}">${initials}</span>`
      + `<span class="ui-win__txt"><span class="ui-win__who">${esc(short)} · ${'★'.repeat(Math.round(s.rating))}${ok ? '' : '<span class="miss">Missed</span>'}</span><span class="ui-win__sub">${esc(snippet)}</span></span>`
      + `<span class="ui-win__st" data-v="${s.pred}">${verdict}</span></li>`;
  }).join('');

  const show = (s) => {
    const text = s.text.replace(/\s+/g, ' ').trim();
    input.value = text;
    const label = s.label === 'OR' ? 'written by a person' : 'generated';
    render(score(text), `${s.category} · labelled ${label}`);
  };
  const pick = (li) => {
    list.querySelectorAll('.is-sel').forEach((el) => el.classList.remove('is-sel'));
    li.classList.add('is-sel');
    show(data.samples[+li.dataset.i]);
  };
  list.addEventListener('click', (e) => { const li = e.target.closest('[data-row]'); if (li) pick(li); });
  list.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const li = e.target.closest('[data-row]');
    if (li) { e.preventDefault(); pick(li); }
  });
  $('det-go').addEventListener('click', () => render(score(input.value)));
  let t;
  input.addEventListener('input', () => { clearTimeout(t); t = setTimeout(() => render(score(input.value)), 150); });

  render(score(input.value), 'Tools and Home Improvement · labelled generated');
  if (document.fonts) document.fonts.ready.then(fitBars);
}
fitBars();

main().catch((err) => {
  console.error('detector: could not load the model', err);
  const meta = document.getElementById('det-meta');
  if (meta) meta.textContent = 'The model did not load. Reload the page to try again.';
});
