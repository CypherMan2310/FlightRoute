// Small DOM helpers. API data is only ever inserted as text nodes (never innerHTML).
export function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  add(el, kids);
  return el;
}
function add(el, kids) {
  for (const k of kids.flat(Infinity)) {
    if (k === null || k === undefined || k === false) continue;
    el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
}
/** Append children, skipping null / undefined / false (native Element.append would write the text "null"). */
export const put = (el, ...kids) => { add(el, kids); return el; };
export const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };
export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

export const fmt = (x, d = 0) => (x === null || x === undefined || Number.isNaN(x) ? '–' : Number(x).toFixed(d));
export const pct = (x, d = 0) => (x === null || x === undefined ? '–' : `${(x * 100).toFixed(d)}%`);
export function ago(sec) {
  if (sec === null || sec === undefined) return '';
  if (sec < 90) return `${Math.round(sec)} s old`;
  if (sec < 5400) return `${Math.round(sec / 60)} min old`;
  return `${(sec / 3600).toFixed(1)} h old`;
}

const SYMBOL = { live: '●', cached: '◐', simulated: '◌' };
/** Provenance badge: LIVE (solid) / CACHED (ruled) / SIMULATED (dashed + hatch). */
export function badge(source, info = {}) {
  const parts = [info.provider, ago(info.age_s), info.fallback_reason && `fallback: ${info.fallback_reason}`, info.degraded && info.degraded.length && `reduced: no ${info.degraded.join(', ')}`].filter(Boolean);
  return h('span', { class: `badge b-${source}`, title: parts.join(' · ') || source }, SYMBOL[source] || '?', ' ', String(source).toUpperCase());
}
export const trainedBadge = (t) => h('span', { class: `badge b-${t === 'real' ? 'real' : 'synthetic'}`, title: t === 'real' ? 'Trained on real flight data' : 'Trained on SYNTHETIC data: demonstration only' }, t === 'real' ? 'trained on real data' : t === 'synthetic' ? 'trained on SYNTHETIC data' : 'not trained');
export const kindBadge = (k) => h('span', { class: 'badge b-kind' }, k);

/** Severity level 0..4 from a 0..100 stress value (hatch density, never colour). */
export const level = (v) => (v < 20 ? 0 : v < 40 ? 1 : v < 60 ? 2 : v < 80 ? 3 : 4);
/** Airport traffic class from stress (0..100). Colour is used ONLY for the airport bars (globe columns and stress bars) and their legend:
 *  green = free, yellow = moderate, red = heavy. The bar length, the number and the title text carry the same information for colour-blind readers. */
export const TRAFFIC = [
  { key: 'free', label: 'free', color: '#2ecc71', from: 0, range: 'stress under 40' },
  { key: 'moderate', label: 'moderate', color: '#f1c40f', from: 40, range: 'stress 40 to 69' },
  { key: 'heavy', label: 'heavy', color: '#ff453a', from: 70, range: 'stress 70 and over' },
];
export const traffic = (v) => (v >= TRAFFIC[2].from ? TRAFFIC[2] : v >= TRAFFIC[1].from ? TRAFFIC[1] : TRAFFIC[0]);
export const stressBar = (v) => {
  const t = traffic(v);
  return h('span', { class: 'bar', title: `stress ${fmt(v, 0)} / 100 · ${t.label} traffic` }, h('i', { style: { width: `${Math.max(2, Math.min(100, v))}%`, background: t.color } }));
};
export const trafficLegend = () => h('div', { class: 'legend' },
  h('b', null, 'Traffic'), TRAFFIC.map((t) => h('span', null, h('i', { class: 'sev', style: { background: t.color } }), `${t.label} (${t.range})`)),
  h('span', null, 'column height and size = stress'));

export function sourcesPanel(sources) {
  const rows = Object.entries(sources || {}).map(([name, s]) => h('tr', null, h('td', null, name.replace(/_/g, ' ')), h('td', null, badge(s.source, s)), h('td', { class: 'mute small' }, [s.provider, ago(s.age_s), s.fallback_reason && `fallback: ${s.fallback_reason}`, s.degraded && s.degraded.length && `reduced: no ${s.degraded.join(', ')}`].filter(Boolean).join(' · '))));
  return h('div', { class: 'tscroll' }, h('table', { class: 'tbl' }, h('tbody', null, rows)));
}

export const loading = (msg = 'loading') => h('div', { class: 'loading', role: 'status' }, msg);
export const errorBox = (e) => h('div', { class: 'err', role: 'alert' }, `Error: ${e.message || e}`);
export const notice = (label, text, cls = '') => h('div', { class: `notice ${cls}` }, h('b', null, label), text);

/** Airport autocomplete bound to a text input through a <datalist>. */
export function airportInput(api, { id, value = '', required = true, placeholder = 'IATA or city' } = {}) {
  const list = h('datalist', { id: `${id}-list` });
  const input = h('input', { id, list: `${id}-list`, value, required, placeholder, autocomplete: 'off', spellcheck: 'false', maxlength: 40 });
  let seq = 0;
  input.addEventListener('input', async () => {
    const q = input.value.trim();
    if (q.length < 1) return;
    const mine = ++seq;
    try {
      const rows = await api.airports(q, 8);
      if (mine !== seq) return;
      clear(list);
      for (const a of rows) list.append(h('option', { value: a.iata }, `${a.name} · ${a.city}, ${a.country}`));
    } catch { /* suggestions are optional */ }
  });
  return { input, list };
}
export const code = (s) => String(s).trim().toUpperCase().split(/[\s,(]/)[0];
