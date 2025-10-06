FROM nginx:1.27-alpine
# static files
COPY frontend /usr/share/nginx/html
# nginx reverse-proxy config (no need to mount volumes on server)
COPY nginx/nginx.conf /etc/nginx/nginx.conf
