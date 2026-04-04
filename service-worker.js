// Vyakt PWA Service Worker
// Keeps caching logic minimal and safe for Flask static routing.

const CACHE_NAME = 'vyakt-cache-v1';

// Requested core assets + current real global assets.
const ASSETS_TO_CACHE = [
  '/',
  '/static/styles.css',
  '/static/app.js',
  '/static/css/style.css',
  '/static/js/main.js',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) =>
      // Add each asset independently so one missing file does not break install.
      Promise.allSettled(ASSETS_TO_CACHE.map((asset) => cache.add(asset)))
    )
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((oldKey) => caches.delete(oldKey))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') {
    return;
  }

  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(event.request);
    })
  );
});
