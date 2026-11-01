# Geharde omgekeerde proxy voor het signaalsysteem (ADR 0002 F, ADR 0016).
#
# De proef masquerade (F2-10) is onder Docker Desktop niet geslaagd: ook met
# enable_ip_masquerade=false heeft een container op `toegang` internet. Daarom
# wordt elke proxy gehard: geen shell en geen pakketbeheer in het image, de
# configuratie in het image (geen volumes, geen secrets), alleen het eigen pad
# naar de eigen doeldienst. Compose zet read_only, cap_drop ALL,
# no-new-privileges en gebruiker 101 (nginx-unprivileged, poort 8080).
# Eén image per proxy: --build-arg CONF=<bestand>.
FROM nginxinc/nginx-unprivileged:1.29.3-alpine-slim@sha256:df9d99123d80117ea348470175f20abe85befd1ed361ea6207eaf49ab5a30ed9
ARG CONF
COPY nginx.conf /etc/nginx/nginx.conf
COPY ${CONF} /etc/nginx/conf.d/default.conf
USER 0
# Shell, busybox-hulpprogramma's, apk en het startscript weg; busybox als laatste.
RUN ["/bin/busybox", "sh", "-c", "B=/bin/busybox; $B rm -rf /docker-entrypoint.sh /docker-entrypoint.d /sbin/apk /etc/apk /lib/apk /usr/share/apk /usr/bin/envsubst && for f in $($B find / -xdev -type l); do case $($B readlink $f) in *busybox*) $B rm -f $f;; esac; done && $B rm -f /bin/busybox"]
USER 101
ENTRYPOINT ["/usr/sbin/nginx", "-g", "daemon off;"]
