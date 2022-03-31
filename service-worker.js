"use strict";

const CACHE = "default";

self.addEventListener("install", function (event) {
  event.waitUntil(
    caches.open(CACHE).then(function (cache) {
      return (
        cache
          .addAll([
            /* {%- for file in site.static_files -%}
                {%- if file.path contains '/assets/static' %} */
            // "{{ file.path | relative_url }}",
            /*  {%- endif -%}
               {%- endfor %} */
            /* The following files are templated, so they won't appear in 'static_files'.  */
            // "/assets/static/page.css",
            "/assets/static/page.js",
            // "/site.webmanifest",
            // "/offline",
          ])
          /* Don't wait for active clients to close before updating */
          .then(self.skipWaiting())
      );
    })
  );
});

self.addEventListener("activate", function (event) {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", function (event) {
  /* only intercept requests from self origin or github content. */
  if (
    !event.request.url.startsWith(self.location.origin) &&
    !event.request.url.startsWith("https://raw.githubusercontent.com")
  ) {
    return;
  }

  event.respondWith(
    event.request.url.startsWith(`${self.location.origin}/assets/static`)
      ? serveCacheThenRevalidate(event)
      : serveNetworkAndUpdateCache(event)
  );
});

function serveCacheThenRevalidate(event) {
  return caches.match(event.request).then(function (response) {
    if (response) {
      event.waitUntil(updateCache(event.request));
      return response;
    } else {
      return updateCache(event.request);
    }
  });
}

function serveNetworkAndUpdateCache(event) {
  return updateCache(event.request).catch(function () {
    return caches.match(event.request).then(function (response) {
      return response || caches.match("/offline");
    });
  });
}

function updateCache(request) {
  return caches.open(CACHE).then(function (cache) {
    return fetch(request).then(function (response) {
      if (!response.ok) {
        return response;
      }

      return cache.put(request, response.clone()).then(function () {
        return response;
      });
    });
  });
}