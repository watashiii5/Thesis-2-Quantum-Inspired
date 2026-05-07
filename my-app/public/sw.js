const CACHE_NAME = 'qtime-cache-v2'; // Bump version to invalidate old cache
const ASSETS = ['/', '/manifest.webmanifest', '/icon.svg'];

// URLs that should NEVER be cached (API routes, Supabase, dynamic data)
const NO_CACHE_PATTERNS = [
  '/api/',
  'supabase.co',
  'supabase.io',
  '/rest/v1/',
  '/auth/v1/',
  '/realtime/',
  '/storage/v1/'
];

function shouldCache(url) {
  // Never cache API or Supabase requests
  return !NO_CACHE_PATTERNS.some(pattern => url.includes(pattern));
}

function isHttpUrl(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
}

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
    )).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  
  const url = event.request.url;

  // Ignore non-HTTP(S) requests (e.g., chrome-extension://) to avoid cache.put errors.
  if (!isHttpUrl(url)) return;
  
  // For API/Supabase requests: ALWAYS go to network (no caching)
  if (!shouldCache(url)) {
    event.respondWith(fetch(event.request));
    return;
  }
  
  // For static assets: cache-first strategy
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((response) => {
        const responseClone = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, responseClone)).catch(() => {});
        return response;
      }).catch(() => caches.match('/'));
    })
  );
});
