// Black-and-white globe (globe.gl). Severity is shown by marker size, column height and pulsing rings; the only hue is the airport column colour
// (green free, yellow moderate, red heavy traffic, see traffic() in ui.js).
import Globe from 'globe.gl';
import { feature } from 'topojson-client';
import landTopo from 'world-atlas/land-110m.json';
import { esc, traffic } from './ui.js';

const LAND = feature(landTopo, landTopo.objects.land).features;

export function createGlobe(el, { onAirport, autoRotate = true, initial = { lat: 30, lng: -30, altitude: 2.4 } } = {}) {
  const g = Globe({ animateIn: false, rendererConfig: { antialias: true, alpha: false } })(el)
    .backgroundColor('#000000')
    .showAtmosphere(false)
    .showGraticules(true)
    .polygonsData(LAND)
    .polygonCapColor(() => '#171717')
    .polygonSideColor(() => 'rgba(0,0,0,0)')
    .polygonStrokeColor(() => '#b8b8b8')
    .polygonAltitude(0.003)
    .pointLat('lat').pointLng('lng')
    .pointAltitude((d) => d.alt)
    .pointRadius((d) => d.radius)
    .pointColor((d) => (d.kind === 'aircraft' ? '#d0d0d0' : traffic(d.row.stress).color))
    .pointResolution(8)
    .pointLabel((d) => `<div class="tip">${esc(d.label)}</div>`)
    .onPointClick((d) => { if (d.kind === 'airport' && onAirport) onAirport(d.row); })
    .ringLat('lat').ringLng('lng').ringMaxRadius('maxR').ringPropagationSpeed(1.4).ringRepeatPeriod('period')
    .ringColor(() => (t) => `rgba(255,255,255,${(1 - t).toFixed(3)})`)
    .labelLat('lat').labelLng('lng').labelText('text').labelSize(0.55).labelDotRadius(0.18).labelColor(() => '#ffffff').labelResolution(2).labelAltitude((d) => d.alt)
    .arcStartLat('slat').arcStartLng('slng').arcEndLat('elat').arcEndLng('elng')
    .arcColor((d) => (d.strong ? ['#ffffff', '#ffffff'] : ['#9a9a9a', '#5a5a5a']))
    .arcStroke((d) => d.stroke).arcAltitudeAutoScale(0.35).arcDashLength(0.45).arcDashGap(0.25).arcDashAnimateTime(2200)
    .arcLabel((d) => `<div class="tip">${esc(d.label)}</div>`);

  const mat = g.globeMaterial();
  mat.color.set('#000000');
  mat.emissive.set('#000000');
  mat.shininess = 0;
  const controls = g.controls();
  controls.autoRotate = autoRotate;
  controls.autoRotateSpeed = 0.35;
  controls.minDistance = 130;
  g.pointOfView(initial, 0);

  const size = () => { g.width(el.clientWidth || 600).height(el.clientHeight || 400); };
  size();
  const ro = new ResizeObserver(size);
  ro.observe(el);

  let airportPoints = [];
  let aircraftPoints = [];
  const refreshPoints = () => g.pointsData([...airportPoints, ...aircraftPoints]);

  return {
    globe: g,
    /** rows: [{iata,name,lat,lon,stress,...}] from /api/congestion */
    setAirports(rows, { labelTop = 12, ringAt = 60 } = {}) {
      airportPoints = rows.map((r) => ({
        kind: 'airport', row: r, lat: r.lat, lng: r.lon,
        alt: 0.004 + (r.stress / 100) * 0.13, radius: 0.2 + (r.stress / 100) * 0.55,
        label: `${r.iata} · ${r.name} · stress ${Math.round(r.stress)} (${traffic(r.stress).label} traffic) · ~${r.pred_dep_delay_min} min`,
      }));
      refreshPoints();
      const ranked = [...rows].sort((a, b) => b.stress - a.stress);
      g.ringsData(ranked.filter((r) => r.stress >= ringAt).slice(0, 25).map((r) => ({ lat: r.lat, lng: r.lon, maxR: 1.8 + (r.stress / 100) * 5, period: Math.max(900, 2600 - r.stress * 16) })));
      g.labelsData(ranked.slice(0, labelTop).map((r) => ({ lat: r.lat, lng: r.lon, text: r.iata, alt: 0.01 + (r.stress / 100) * 0.13 })));
    },
    setAircraft(flights) {
      aircraftPoints = flights.filter((f) => !f.on_ground).map((f) => ({
        kind: 'aircraft', lat: f.lat, lng: f.lon, alt: 0.006 + Math.min(1, (f.alt_ft || 0) / 45000) * 0.04, radius: 0.07,
        label: `${f.callsign || f.icao24} · ${f.alt_ft ?? '?'} ft · ${f.gs_kt ?? '?'} kt`,
      }));
      refreshPoints();
    },
    setArcs(arcs) { g.arcsData(arcs); },
    setRings(rings) { g.ringsData(rings); },
    focus(lat, lng, altitude = 1.6, ms = 900) { controls.autoRotate = false; g.pointOfView({ lat, lng, altitude }, ms); },
    rotate(on) { controls.autoRotate = on; },
    dispose() {
      ro.disconnect();
      try { g.pauseAnimation(); if (g._destructor) g._destructor(); } catch { /* already gone */ }
      el.replaceChildren();
    },
  };
}
