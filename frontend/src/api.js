// Thin API client. Base URL is empty in production (same origin) and proxied to FastAPI in dev.
const BASE = import.meta.env.VITE_API_BASE || '';
const TIMEOUT_MS = 25000;

async function request(path, opts = {}) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(BASE + path, { ...opts, signal: ctl.signal, headers: { accept: 'application/json', ...(opts.headers || {}) } });
    const offline = res.headers.get('x-aerogrid-offline') === '1';
    let body = null;
    try { body = await res.json(); } catch { /* non-JSON error page */ }
    if (!res.ok) {
      const d = body && body.detail;
      const msg = typeof d === 'string' ? d : Array.isArray(d) ? d.map((x) => x.msg).join('; ') : `HTTP ${res.status}`;
      const err = new Error(msg);
      err.status = res.status;
      throw err;
    }
    if (offline && body && typeof body === 'object') Object.defineProperty(body, '__offline', { value: true, enumerable: false });
    return body;
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('The server did not answer in time.');
    if (e instanceof TypeError) throw new Error('Cannot reach the AEROGRID API.');
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

const qs = (params) => {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params || {})) if (v !== undefined && v !== null && v !== '') u.set(k, v);
  const s = u.toString();
  return s ? `?${s}` : '';
};

export const api = {
  health: () => request('/api/health'),
  models: () => request('/api/models'),
  airports: (q, limit = 8) => request(`/api/airports${qs({ q, limit })}`),
  flights: (p) => request(`/api/flights${qs(p)}`),
  congestion: (limit = 155) => request(`/api/congestion${qs({ limit })}`),
  predict: (body) => request('/api/predict', { method: 'POST', body: JSON.stringify(body), headers: { 'content-type': 'application/json' } }),
  advisor: (p) => request(`/api/advisor${qs(p)}`),
  whatif: (body) => request('/api/whatif', { method: 'POST', body: JSON.stringify(body), headers: { 'content-type': 'application/json' } }),
  scorecard: () => request('/api/scorecard'),
};

/** Server-sent events. Falls back to polling if the stream cannot be opened. Returns a stop function. */
export function openStream(handlers, { pollMs = 30000 } = {}) {
  let es = null;
  let timer = null;
  let stopped = false;
  const poll = async () => {
    if (stopped) return;
    try {
      if (handlers.flights) handlers.flights(await api.flights());
      if (handlers.congestion) handlers.congestion(await api.congestion(20));
    } catch (e) { if (handlers.error) handlers.error(e); }
    timer = setTimeout(poll, pollMs);
  };
  if (typeof EventSource === 'undefined') { poll(); return () => { stopped = true; clearTimeout(timer); }; }
  es = new EventSource(`${BASE}/api/stream`);
  for (const name of ['flights', 'congestion']) {
    es.addEventListener(name, (ev) => { try { if (handlers[name]) handlers[name](JSON.parse(ev.data)); } catch (e) { if (handlers.error) handlers.error(e); } });
  }
  es.onerror = () => { if (!timer && !stopped) { es.close(); poll(); } };
  return () => { stopped = true; if (es) es.close(); clearTimeout(timer); };
}
