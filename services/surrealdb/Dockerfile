# SurrealDB voor het signaalsysteem (B3, ADR 0021, ADR 0002 F).
#
# Het image van SurrealDB heeft geen shell. De inloggegevens staan als Docker
# secret in /run/secrets; SurrealDB leest alleen SURREAL_USER en SURREAL_PASS
# uit de omgeving. Daarom een statische busybox, alleen om start.sh te draaien.
FROM busybox:1.37.0-musl@sha256:f6a95d611e0a8780fabdb2aec43e31275b2720dfefa3e41b16835bfc42454478 AS busybox

FROM surrealdb/surrealdb:v2.7.0@sha256:e5d7387e3727bbe2f7cd6d869e48cd34777272258572ead37c9656b25f745723
COPY --from=busybox /bin/busybox /busybox
COPY start.sh /start.sh
USER 0
# Het lege volume signaal_surreal_data neemt eigenaar en rechten van /data over.
RUN ["/busybox", "sh", "-c", "/busybox mkdir -p /data && /busybox chown 65532:65532 /data"]
USER 65532
ENTRYPOINT ["/busybox", "sh", "/start.sh"]
