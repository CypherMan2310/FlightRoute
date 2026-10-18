// Home: spinning black-and-white globe with live aircraft telemetry and airport stress.
import { api, openStream } from '../api.js';
import { h, clear, badge, trainedBadge, stressBar, fmt, ago, errorBox, loading, put } from '../ui.js';

export default async function mount(root) {
  const stage = h('div', { class: 'globe-stage' });
  const gl = h('div', { class: 'gl', role: 'img', 'aria-label': 'Globe of airports sized by congestion stress and live aircraft' });
  const left = h('aside', { class: 'hud left' }, loading('connecting'));
  const right = h('aside', { class: 'hud right', hidden: true });
  stage.append(gl, left, right);
  root.append(stage);

  const { createGlobe } = await import('../globe.js');
  let selected = null;
  const g = createGlobe(gl, { onAirport: (row) => select(row) });
  let cong = null;
  let flights = null;

  function select(row) {
    selected = row;
    g.focus(row.lat, row.lon, 1.5);
    right.hidden = false;
    put(clear(right), 
      h('h3', null, `${row.iata} · ${row.name}`),
      h('p', { class: 'small' }, `${row.city}, ${row.country} · hub tier ${row.tier}`),
      h('dl', { class: 'kv' },
        h('dt', null, 'stress'), h('dd', null, `${fmt(row.stress)} / 100`),
        h('dt', null, 'pred. dep. delay'), h('dd', null, `${fmt(row.pred_dep_delay_min, 1)} min`),
        h('dt', null, 'congestion model'), h('dd', null, `${fmt(row.pred_nas_min, 1)} min`),
        h('dt', null, 'weather'), h('dd', null, row.weather_category),
        h('dt', null, 'FAA status'), h('dd', null, row.nas.status + (row.nas.delay_minutes ? ` (${row.nas.delay_minutes} min)` : ''))),
      row.confidence === 'low' ? h('p', { class: 'small' }, h('span', { class: 'badge b-low' }, 'lower confidence'), ' trained on US airports only') : null,
      h('div', { class: 'row', style: { marginTop: '10px' } },
        h('a', { class: 'badge', href: `#/search?from=${row.iata}` }, 'predict from here'),
        h('a', { class: 'badge', href: `#/whatif?airport=${row.iata}` }, 'what-if')),
    );
  }

  const narrow = window.matchMedia('(max-width: 640px)').matches;
  let collapsed = narrow;
  function render() {
    clear(left);
    left.classList.toggle('collapsed', collapsed);
    put(left,
      h('div', { class: 'hud-head row between' },
        h('h1', null, 'AEROGRID'),
        h('button', { class: 'ghost', type: 'button', 'aria-expanded': String(!collapsed), onclick: () => { collapsed = !collapsed; render(); } }, collapsed ? 'show data' : 'hide')),
      h('p', { class: 'small' }, 'Delay and congestion, with the provenance of every number.'),
    );
    if (cong) {
      const top = cong.airports.slice(0, 8);
      put(left, 
        h('div', { class: 'row', style: { marginBottom: '8px' } }, badge(cong.source, cong.sources.metar), trainedBadge(cong.trained_on)),
        h('h2', null, 'Most stressed now'),
        h('div', { class: 'mini', style: { margin: '8px 0 12px' } }, top.flatMap((a) => [
          h('a', { href: '#/', onclick: (e) => { e.preventDefault(); select(a); }, class: 'mono' }, `${a.iata} ${fmt(a.stress)}`), stressBar(a.stress)])),
        h('div', { class: 'legend' },
          h('span', null, h('i', { class: 'sev h0' }), 'calm'), h('span', null, h('i', { class: 'sev h2' }), 'busy'), h('span', null, h('i', { class: 'sev h4' }), 'severe'),
          h('span', null, 'size and height = stress')),
      );
    }
    if (flights) {
      put(left, 
        h('h2', { style: { marginTop: '14px' } }, 'Aircraft'),
        h('div', { class: 'row', style: { margin: '6px 0' } }, badge(flights.source, flights), h('span', { class: 'mono' }, `${flights.count} shown`, flights.regions ? ` · ${flights.regions} regions` : '')),
        h('p', { class: 'small' }, flights.source === 'simulated' ? 'Live ADS-B is unavailable: these aircraft are SIMULATED on a modelled route network, not real traffic.' : ['Positions from ', flights.provider, String(flights.provider).includes('adsb.fi') ? [' (', h('a', { href: 'https://adsb.fi', rel: 'noopener' }, 'adsb.fi'), ', personal non-commercial use)'] : null, flights.age_s ? `, ${ago(flights.age_s)}` : '', '.']),
      );
    }
    if (cong && cong.trained_on === 'synthetic') put(left, h('div', { class: 'notice warn', style: { marginTop: '12px' } }, h('b', null, 'Demo models'), 'Stress comes from models trained on SYNTHETIC data.'));
  }

  try {
    cong = await api.congestion(155);
    g.setAirports(cong.airports);
    render();
  } catch (e) { put(clear(left), h('h1', null, 'AEROGRID'), errorBox(e)); }
  try { flights = await api.flights(); g.setAircraft(flights.flights); render(); } catch { /* shown as missing */ }

  const stop = openStream({
    flights: (f) => { flights = f; g.setAircraft(f.flights); render(); },
    congestion: async () => { try { cong = await api.congestion(155); g.setAirports(cong.airports); render(); } catch { /* keep last */ } },
  });

  return () => { stop(); g.dispose(); };
}
