// Search: origin, destination, optional airline -> predicted delay, 80% interval, cause breakdown, drivers, route and bird advice.
import { api } from '../api.js';
import { h, clear, badge, trainedBadge, kindBadge, sourcesPanel, airportInput, code, fmt, pct, errorBox, loading, notice, put } from '../ui.js';

const AIRLINES = [['', 'Any / unknown'], ['AA', 'American'], ['DL', 'Delta'], ['UA', 'United'], ['WN', 'Southwest'], ['B6', 'JetBlue'], ['AS', 'Alaska'], ['NK', 'Spirit'], ['F9', 'Frontier'], ['HA', 'Hawaiian'], ['OO', 'SkyWest'], ['MQ', 'Envoy'], ['YX', 'Republic'], ['9E', 'Endeavor']];
const AIRCRAFT = [['narrowbody', 'Narrow-body (A320 / 737)'], ['regional', 'Regional jet'], ['widebody', 'Wide-body (787 / A350)'], ['ultra_long', 'Ultra-long range']];
const localInput = (d) => new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16);

export default async function mount(root) {
  const params = new URLSearchParams(location.hash.split('?')[1] || '');
  const out = h('div', { class: 'stack' });
  const from = airportInput(api, { id: 'origin', value: params.get('from') || '', placeholder: 'e.g. JFK' });
  const to = airportInput(api, { id: 'dest', value: params.get('to') || '', placeholder: 'e.g. LAX' });
  const airline = h('select', { id: 'airline' }, AIRLINES.map(([v, t]) => h('option', { value: v }, v ? `${v} · ${t}` : t)));
  const when = h('input', { id: 'when', type: 'datetime-local', value: localInput(new Date(Date.now() + 2 * 3600000)) });
  const aircraft = h('select', { id: 'aircraft' }, AIRCRAFT.map(([v, t]) => h('option', { value: v }, t)));
  const go = h('button', { type: 'submit' }, 'Predict');

  const form = h('form', { class: 'f', onsubmit: submit },
    h('label', { class: 'fld' }, 'From', from.input, from.list), h('label', { class: 'fld' }, 'To', to.input, to.list),
    h('label', { class: 'fld' }, 'Airline', airline), h('label', { class: 'fld' }, 'Departure (local time)', when),
    h('label', { class: 'fld' }, 'Aircraft class', aircraft), go);

  root.append(h('section', { class: 'page' },
    h('div', { class: 'hero' }, h('h1', null, 'Will it be late, and why?'), h('span', { class: 'lede' }, 'Predicted arrival delay with an 80% interval and an exact cause breakdown.')),
    h('div', { class: 'panel' }, form), out));

  async function submit(ev) {
    ev.preventDefault();
    go.disabled = true;
    put(clear(out), loading('predicting'));
    try {
      const body = { origin: code(from.input.value), destination: code(to.input.value), aircraft: aircraft.value };
      if (airline.value) body.airline = airline.value;
      if (when.value) body.departure_time = new Date(when.value).toISOString();
      const r = await api.predict(body);
      put(clear(out), ...result(r));
    } catch (e) { put(clear(out), errorBox(e)); } finally { go.disabled = false; }
  }
  if (from.input.value && to.input.value) form.requestSubmit();
}

function result(r) {
  const hi = Math.max(r.interval80[1], r.delay_min, 30) * 1.1;
  const at = (v) => `${Math.min(100, (v / hi) * 100)}%`;
  const range = h('div', { class: 'range', role: 'img', 'aria-label': `80% interval ${r.interval80[0]} to ${r.interval80[1]} minutes` },
    h('div', { class: 'span', style: { left: at(r.interval80[0]), width: `calc(${at(r.interval80[1])} - ${at(r.interval80[0])})` } }),
    h('div', { class: 'tick', style: { left: at(r.delay_min) } }),
    h('span', { class: 'scale', style: { left: '0' } }, '0'), h('span', { class: 'scale', style: { right: '0' } }, `${fmt(hi)} min`));

  const total = r.causes.reduce((s, c) => s + c.minutes, 0) + r.baseline_min;
  const segs = [...r.causes.map((c) => ({ key: c.key, label: c.label, minutes: c.minutes })), { key: 'baseline', label: 'Unattributed baseline', minutes: r.baseline_min }];
  const stack = h('div', { class: 'stackbar', role: 'img', 'aria-label': 'cause breakdown' },
    segs.filter((s) => s.minutes > 0).map((s) => h('span', { class: `p-${s.key}`, style: { width: `${(s.minutes / Math.max(total, 0.01)) * 100}%` }, title: `${s.label}: ${fmt(s.minutes, 1)} min` })));

  const low = r.confidence.level === 'low';
  return [
    h('div', { class: 'row' }, h('h1', { class: 'mono' }, `${r.origin} → ${r.destination}`), h('span', { class: 'mute' }, `${r.airline || 'any airline'} · departs ${r.departure_time.replace('T', ' ')} UTC · ${r.horizon_h} h horizon`),
      trainedBadge(r.trained_on), low ? h('span', { class: 'badge b-low' }, 'lower confidence') : null),
    ...r.confidence.notes.map((n) => notice(n.includes('SYNTHETIC') ? 'Demo models' : low && n.startsWith('Outside') ? 'Low confidence' : 'Note', n, n.includes('SYNTHETIC') || low ? 'warn' : '')),
    h('div', { class: 'grid two' },
      h('div', { class: 'panel stack' },
        h('div', { class: 'stat' }, h('span', { class: 'k' }, 'Predicted arrival delay'), h('span', { class: 'v' }, `${fmt(r.delay_min, 0)} min`)),
        h('div', null, h('span', { class: 'k small' }, `80% interval: ${fmt(r.interval80[0])} to ${fmt(r.interval80[1])} min`), h('div', { style: { marginTop: '8px', marginBottom: '22px' } }, range)),
        h('p', { class: 'small' }, r.confidence.interval),
        h('div', { class: 'row' }, h('span', { class: 'small' }, 'FAA status'), h('span', { class: 'mono' }, `${r.origin}: ${statusText(r.nas.origin)}`), h('span', { class: 'mono' }, `${r.destination}: ${statusText(r.nas.destination)}`)),
      ),
      h('div', { class: 'panel stack' },
        h('h2', null, 'Why: cause breakdown'),
        stack,
        h('div', { class: 'tscroll' }, h('table', { class: 'tbl' }, h('tbody', null,
          segs.map((s) => h('tr', null,
            h('td', null, h('span', { class: `swatch p-${s.key}` }), s.label, ['route', 'bird'].includes(s.key) ? ' ' : '', s.key === 'route' ? kindBadge('algorithmic') : s.key === 'bird' ? kindBadge('prior-based estimate') : null),
            h('td', { class: 'r mono' }, `${fmt(s.minutes, 1)} min`), h('td', { class: 'r mono mute' }, s.key === 'baseline' ? '' : pct(s.minutes / Math.max(r.delay_min, 0.01))))),
        ))),
        h('p', { class: 'small' }, 'Parts and baseline add up exactly to the prediction. The bird part is a statistical weight on an ecology prior, not a measured strike delay.'),
      )),
    h('div', { class: 'grid two' },
      h('div', { class: 'panel' }, h('h2', null, 'Top drivers'), ...Object.entries(r.drivers).map(([k, list]) => h('div', { style: { marginBottom: '10px' } },
        h('h3', null, h('span', { class: `swatch p-${k}` }), k),
        h('ul', { class: 'plain small' }, list.map((d) => h('li', null, d.label, d.value !== undefined && d.value !== null ? ` = ${fmt(d.value, 2)}` : '', d.contribution_min !== undefined ? ` → ${d.contribution_min >= 0 ? '+' : ''}${fmt(d.contribution_min, 1)} min` : '', d.note ? ` (${d.note})` : '')))))),
      h('div', { class: 'panel stack' }, h('h2', null, 'Route and altitude advice'), h('ul', { class: 'plain adv' }, r.advice.map((a) => h('li', null, a))), routeTable(r.route),
        h('div', { class: 'row' }, kindBadge(r.route.label), kindBadge(r.bird.label)))),
    h('div', { class: 'panel' }, h('h2', null, 'Data behind this answer'), sourcesPanel(r.sources),
      h('p', { class: 'small', style: { marginTop: '8px' } }, 'Models: ', Object.entries(r.models).map(([k, m]) => `${k} (${m.kind}, ${m.trained_on})`).join(' · '))),
  ];
}

const statusText = (s) => (s.status === 'normal' ? 'no programme' : s.status === 'not covered' ? 'not covered (non-US)' : `${s.status.replace('_', ' ')}${s.delay_minutes ? ` ${s.delay_minutes} min` : ''}`);

function routeTable(route) {
  if (!route.feasible) return h('p', { class: 'small' }, 'No feasible chain of legs for this aircraft class.');
  return h('div', { class: 'tscroll' }, h('table', { class: 'tbl' },
    h('thead', null, h('tr', null, h('th', null, 'Leg'), h('th', { class: 'r' }, 'km'))),
    h('tbody', null, route.legs.map((l) => h('tr', null, h('td', { class: 'mono' }, `${l.from} → ${l.to}`), h('td', { class: 'r mono' }, fmt(l.km)))),
      h('tr', null, h('td', null, `Total${route.stops.length ? ` (${route.stops.length} stop${route.stops.length > 1 ? 's' : ''}, +${route.detour_pct}%)` : ' (direct)'}`), h('td', { class: 'r mono' }, fmt(route.total_km))))));
}
