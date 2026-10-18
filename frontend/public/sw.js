// AEROGRID service worker: offline app shell + network-first API with an explicit "offline copy" marker.
// Live streams (/api/stream) are never cached. API answers served from cache carry the header x-aerogrid-offline: 1
// so the UI can say it is showing the last copy instead of live data.
const SHELL = 'aerogrid-shell-v1';
const API = 'aerogrid-api-v1';

self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(caches.open(SHELL).then((c) => c.addAll(['/', '/manifest.webmanifest', '/icon.svg', '/icon-192.png'])));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => ![SHELL, API].includes(k)).map((k) => caches.delete(k)))).then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin || url.pathname === '/api/stream') return;

  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(req)
        .then((res) => {
          if (res.ok) caches.open(API).then((c) => c.put(req, res.clone()));
          return res;
        })
        .catch(async () => {
          const hit = await caches.match(req, { cacheName: API });
          if (!hit) return new Response(JSON.stringify({ detail: 'offline and no saved copy' }), { status: 503, headers: { 'content-type': 'application/json' } });
          const headers = new Headers(hit.headers);
          headers.set('x-aerogrid-offline', '1');
          return new Response(await hit.blob(), { status: hit.status, headers });
        }),
    );
    return;
  }

  event.respondWith(
    caches.match(req).then((hit) => {
      const network = fetch(req)
        .then((res) => {
          if (res.ok) caches.open(SHELL).then((c) => c.put(req, res.clone()));
          return res;
        })
        .catch(() => hit || (req.mode === 'navigate' ? caches.match('/') : Response.error()));
      return hit || network;
    }),
  );
});
