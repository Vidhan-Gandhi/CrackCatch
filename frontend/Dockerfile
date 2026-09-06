# CrackCatch dashboard - built with Vite, served by nginx.
FROM node:22-alpine AS build

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY frontend/ ./
# Same-origin by default: nginx proxies /api and /ws to the backend, so the
# browser needs no CORS and the WebSocket upgrade works unchanged.
ARG VITE_API_BASE=""
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

FROM nginx:1.27-alpine
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
# Use 127.0.0.1, not localhost: inside the container `localhost` can resolve
# to ::1 first, and nginx here listens only on IPv4, so the check would fail
# with "connection refused" while the site serves perfectly from outside.
HEALTHCHECK --interval=30s --timeout=4s --start-period=10s --retries=3 \
    CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
CMD ["nginx", "-g", "daemon off;"]
