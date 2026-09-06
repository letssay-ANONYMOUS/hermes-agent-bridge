const CACHE_NAME = 'hermes-voice-shell-v50'
const SHELL_ASSETS = [
  './',
  'index.html',
  'manifest.webmanifest',
  'icon-180.png',
  'icon-512.png'
]

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(SHELL_ASSETS))
      .then(() => self.skipWaiting())
      .catch(() => self.skipWaiting())
  )
})

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))))
      .then(() => self.clients.claim())
  )
})

self.addEventListener('fetch', event => {
  const request = event.request
  const url = new URL(request.url)
  if (request.method !== 'GET') return
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/ws/')) return
  event.respondWith(
    fetch(request)
      .then(response => {
        const copy = response.clone()
        caches.open(CACHE_NAME).then(cache => cache.put(request, copy)).catch(() => {})
        return response
      })
      .catch(() => caches.match(request).then(cached => cached || caches.match('index.html')))
  )
})
