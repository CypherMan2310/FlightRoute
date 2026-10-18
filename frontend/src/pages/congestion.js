// Congestion: live ranked airports with a 0-100 stress index (hatch density = severity).
import { api } from '../api.js';
import { h, clear, badge, trainedBadge, sourcesPanel, stressBar, trafficLegend, fmt, errorBox, loading, put } from '../ui.js';

export default async function mount(root) {
  let payload = null;
  let timer = null;
  const q = h('input', { type: 'search', placeholder: 'filter: JFK, london, US…', 'aria-label': 'Filter airports' });
  const usOnly = h('input', { type: 'checkbox', id: 'us', checked: true });
  const body = h('div', null, loading('loading airports'));
  const meta = h('div', { class: 'row' });
  root.append(h('section', { class: 'page' },
    h('div', { class: 'hero' }, h('h1', null, 'Airport congestion'), h('span', { class: 'lede' }, 'Ranked by stress: the percentile of predicted airport-hour delay among training hours. Bar colour shows the traffic: green free, yellow moderate, red heavy.')),
    h('div', { class: 'row', style: { marginBottom: '12px' } }, h('label', { class: 'fld' }, 'Filter', q), h('label', { class: 'row small', for: 'us' }, usOnly, 'US airports only: where the models are validated and FAA status exists. Untick to include world airports (lower confidence).'), meta),
    trafficLegend(),
    h('div', { class: 'panel' }, body)));

  const draw = () => {
    if (!payload) return;
    const term = q.value.trim().toLowerCase();
    let rows = payload.airports.filter((a) => (!usOnly.checked || a.country === 'US' || a.country === 'PR') && (!term || `${a.iata} ${a.icao} ${a.name} ${a.city} ${a.country}`.toLowerCase().includes(term)));
    put(clear(body), 
      h('div', { class: 'tscroll' }, h('table', { class: 'tbl' },
        h('thead', null, h('tr', null, ['#', 'Airport', 'Stress', 'Pred. dep. delay', 'Congestion model', 'Weather', 'FAA status', 'Confidence'].map((t, i) => h('th', { class: i > 2 && i < 5 ? 'r' : '' }, t)))),
        h('tbody', null, rows.slice(0, 80).map((a, i) => h('tr', null,
          h('td', { class: 'mono mute' }, i + 1),
          h('td', null, h('a', { href: `#/search?from=${a.iata}`, class: 'mono' }, a.iata), ' ', h('span', { class: 'mute' }, `${a.name}${a.country ? ` · ${a.country}` : ''}`)),
          h('td', null, h('span', { class: 'row', style: { gap: '8px', flexWrap: 'nowrap' } }, stressBar(a.stress), h('span', { class: 'mono' }, fmt(a.stress)))),
          h('td', { class: 'r mono' }, `${fmt(a.pred_dep_delay_min, 1)} min`), h('td', { class: 'r mono' }, `${fmt(a.pred_nas_min, 1)} min`),
          h('td', { class: 'mono' }, a.weather_category),
          h('td', { class: 'mono' }, a.nas.status === 'normal' ? '–' : a.nas.status === 'not covered' ? h('span', { class: 'mute' }, 'n/a') : `${a.nas.status.replace('_', ' ')}${a.nas.delay_minutes ? ` ${a.nas.delay_minutes}m` : ''}`),
          h('td', null, a.confidence === 'low' ? h('span', { class: 'badge b-low', title: 'Trained on US airports only' }, 'low') : h('span', { class: 'mute small' }, 'normal'))))),
      )),
      rows.length > 80 ? h('p', { class: 'small' }, `Showing the top 80 of ${rows.length}; refine the filter.`) : null,
      h('h2', { style: { margin: '16px 0 8px' } }, 'Data behind this ranking'), sourcesPanel(payload.sources),
      h('p', { class: 'small', style: { marginTop: '8px' } }, payload.note),
    );
  };

  async function load() {
    try {
      payload = await api.congestion(155);
      put(clear(meta), badge(payload.source, payload.sources.metar), trainedBadge(payload.trained_on), h('span', { class: 'small mono' }, `horizon ${payload.horizon_h} h model`));
      draw();
    } catch (e) { put(clear(body), errorBox(e)); }
    timer = setTimeout(load, 60000);
  }
  q.addEventListener('input', draw);
  usOnly.addEventListener('change', draw);
  load();
  return () => clearTimeout(timer);
}
