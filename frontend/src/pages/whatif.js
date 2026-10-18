// What-if: drop a storm or close runway capacity at an airport and watch the cascade through the graph.
import { api } from '../api.js';
import { h, clear, trainedBadge, kindBadge, stressBar, airportInput, code, fmt, errorBox, loading, notice, put } from '../ui.js';

export default async function mount(root) {
  const params = new URLSearchParams(location.hash.split('?')[1] || '');
  const ap = airportInput(api, { id: 'wi-ap', value: params.get('airport') || 'ATL', placeholder: 'e.g. ATL' });
  const kind = h('select', { id: 'wi-kind' }, h('option', { value: 'storm' }, 'Storm over the airport'), h('option', { value: 'runway_closure' }, 'Runway closure (capacity loss)'));
  const sev = h('input', { id: 'wi-sev', type: 'range', min: 0, max: 1, step: 0.05, value: 0.8 });
  const sevVal = h('span', { class: 'mono' }, '0.80');
  sev.addEventListener('input', () => { sevVal.textContent = Number(sev.value).toFixed(2); });
  const go = h('button', { type: 'submit' }, 'Run scenario');
  const out = h('div', { class: 'stack' });
  const globeBox = h('div', { class: 'globe-small' }, h('div', { class: 'gl' }));
  let g = null;
  let created = null;

  root.append(h('section', { class: 'page' },
    h('div', { class: 'hero' }, h('h1', null, 'What-if'), h('span', { class: 'lede' }, 'Edit one airport, re-run the congestion model and the graph network, and see who is affected. A model-based scenario, not a forecast.')),
    h('div', { class: 'panel' }, h('form', { class: 'f', onsubmit: run },
      h('label', { class: 'fld' }, 'Airport', ap.input, ap.list), h('label', { class: 'fld' }, 'Scenario', kind),
      h('label', { class: 'fld' }, h('span', null, 'Severity ', sevVal), sev), go)),
    h('div', { class: 'grid two' }, globeBox, out)));

  async function ensureGlobe() {
    if (g) return g;
    created = import('../globe.js').then(({ createGlobe }) => { g = createGlobe(globeBox.firstChild, { autoRotate: false, initial: { lat: 35, lng: -40, altitude: 2.7 } }); return g; });
    return created;
  }

  async function run(ev) {
    ev.preventDefault();
    go.disabled = true;
    put(clear(out), loading('running scenario'));
    try {
      const res = await api.whatif({ airport: code(ap.input.value), kind: kind.value, severity: Number(sev.value) });
      const gl = await ensureGlobe();
      const src = res.deltas.find((d) => d.iata === res.airport);
      const hit = res.affected.filter((d) => d.iata !== res.airport && Math.abs(d.delta_dep_min) > 0.05).slice(0, 12);
      gl.setAirports(res.deltas.map((d) => ({ iata: d.iata, name: d.name, lat: d.lat, lon: d.lon, stress: d.stress_after, pred_dep_delay_min: d.dep_after })), { labelTop: 0, ringAt: 999 });
      gl.setRings([{ lat: src.lat, lng: src.lon, maxR: 3 + res.severity * 7, period: 1400 }]);
      gl.setArcs(hit.map((d) => ({ slat: src.lat, slng: src.lon, elat: d.lat, elng: d.lon, stroke: 0.15 + Math.min(1, Math.abs(d.delta_dep_min) / 15) * 0.7, strong: Math.abs(d.delta_dep_min) > 3, label: `${res.airport} → ${d.iata}: ${d.delta_dep_min >= 0 ? '+' : ''}${fmt(d.delta_dep_min, 1)} min` })));
      gl.focus(src.lat, src.lon, 2.2, 700);
      put(clear(out), 
        h('div', { class: 'row' }, h('h2', null, `${res.airport} · ${kind.options[kind.selectedIndex].text}`), trainedBadge(res.trained_on), kindBadge(res.label)),
        notice('Assumption', res.assumption),
        h('div', { class: 'panel' }, h('div', { class: 'row between' },
          h('div', { class: 'stat' }, h('span', { class: 'k' }, `${res.airport} predicted departure delay`), h('span', { class: 'v sm' }, `${fmt(src.dep_before, 1)} → ${fmt(src.dep_after, 1)} min`)),
          h('div', { class: 'stat' }, h('span', { class: 'k' }, 'stress'), h('span', { class: 'v sm' }, `${fmt(src.stress_before)} → ${fmt(src.stress_after)}`)))),
        h('div', { class: 'panel' }, h('h2', null, 'Most affected airports'), h('div', { class: 'tscroll' }, h('table', { class: 'tbl' },
          h('thead', null, h('tr', null, ['Airport', 'Hops', 'Δ departure delay', 'Stress after'].map((t, i) => h('th', { class: i === 2 ? 'r' : '' }, t)))),
          h('tbody', null, res.affected.map((d) => h('tr', null,
            h('td', null, h('span', { class: 'mono' }, d.iata), ' ', h('span', { class: 'mute' }, d.name)), h('td', { class: 'mono' }, d.hops === null ? '–' : d.hops),
            h('td', { class: 'r mono' }, `${d.delta_dep_min >= 0 ? '+' : ''}${fmt(d.delta_dep_min, 1)} min`), h('td', null, stressBar(d.stress_after))))))),
          h('p', { class: 'small' }, `Network used: ${res.inference_graph}.`)),
      );
    } catch (e) { put(clear(out), errorBox(e)); } finally { go.disabled = false; }
  }
  if (params.get('airport')) run({ preventDefault() {} });
  return () => { if (g) g.dispose(); else if (created) created.then((x) => x.dispose()); };
}
