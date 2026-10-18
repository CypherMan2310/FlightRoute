import './style.css';
import { api } from './api.js';
import { h, clear, trainedBadge, errorBox, loading, put } from './ui.js';

const ROUTES = [
  ['', 'Globe', () => import('./pages/home.js')],
  ['search', 'Search', () => import('./pages/search.js')],
  ['congestion', 'Congestion', () => import('./pages/congestion.js')],
  ['whatif', 'What-if', () => import('./pages/whatif.js')],
  ['scorecard', 'Scorecard', () => import('./pages/scorecard.js')],
  ['models', 'Models', () => import('./pages/models.js')],
];

const app = document.getElementById('app');
const status = h('div', { class: 'nav-status', 'aria-live': 'polite' });
const links = h('nav', { class: 'nav-links', 'aria-label': 'Pages' }, ROUTES.map(([path, label]) => h('a', { href: `#/${path}`, 'data-route': path }, label)));
const nav = h('header', { class: 'nav' }, h('a', { class: 'brand', href: '#/' }, 'AEROGRID'), links, status);
const view = h('main', { id: 'view' });
const foot = h(
  'footer',
  { class: 'foot' },
  h('div', null, 'Free data only. Every figure carries its provenance: LIVE, CACHED or SIMULATED, and the model that produced it.'),
  h('ul', null,
    h('li', null, 'Flights: ', h('a', { href: 'https://adsb.lol', rel: 'noopener' }, 'adsb.lol'), ' (ODbL 1.0)'),
    h('li', null, 'Weather: ', h('a', { href: 'https://aviationweather.gov', rel: 'noopener' }, 'aviationweather.gov'), ' (NOAA/NWS), ', h('a', { href: 'https://open-meteo.com', rel: 'noopener' }, 'Open-Meteo.com'), ' (CC BY 4.0), ', h('a', { href: 'https://api.met.no', rel: 'noopener' }, 'MET Norway'), ' (CC BY 4.0, forecast fallback)'),
    h('li', null, 'Delay programmes: ', h('a', { href: 'https://nasstatus.faa.gov', rel: 'noopener' }, 'FAA NAS Status')),
    h('li', null, 'Birds: ', h('a', { href: 'https://ebird.org', rel: 'noopener' }, 'eBird.org'), ' (when a key is configured)'),
    h('li', null, 'Training data: US BTS On-Time Performance, IEM ASOS')),
);
app.append(nav, view, foot);

let cleanup = null;
let token = 0;

async function route() {
  const path = (location.hash.replace(/^#\/?/, '') || '').split('?')[0];
  const entry = ROUTES.find(([p]) => p === path) || ROUTES[0];
  const mine = ++token;
  if (cleanup) { try { cleanup(); } catch { /* ignore */ } cleanup = null; }
  for (const a of links.children) {
    if (a.dataset.route === entry[0]) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  }
  put(clear(view), loading('loading page'));
  document.title = entry[1] === 'Globe' ? 'AEROGRID' : `${entry[1]} · AEROGRID`;
  try {
    const mod = await entry[2]();
    if (mine !== token) return;
    clear(view);
    cleanup = (await mod.default(view)) || null;
    if (mine !== token && cleanup) { cleanup(); cleanup = null; }
    window.scrollTo(0, 0);
  } catch (e) {
    if (mine === token) put(clear(view), errorBox(e));
  }
}

async function refreshStatus() {
  try {
    const hl = await api.health();
    put(clear(status), 
      h('span', { title: 'API reachable' }, '● API'),
      hl.models.status === 'trained' ? trainedBadge(hl.models.trained_on) : h('span', { class: 'badge b-low' }, 'models not trained'),
      hl.__offline ? h('span', { class: 'badge b-offline' }, 'offline copy') : null,
    );
  } catch {
    put(clear(status), h('span', { title: 'API unreachable' }, '○ API offline'));
  }
}

window.addEventListener('hashchange', route);
route();
refreshStatus();
setInterval(refreshStatus, 30000);

if ('serviceWorker' in navigator && import.meta.env.PROD) {
  window.addEventListener('load', () => navigator.serviceWorker.register('/sw.js').catch(() => {}));
}
