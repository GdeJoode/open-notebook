# Afgeleid llama-swap-image met de Docker-client (docs/signaalsysteem/proeven.md, P1 en P7, stap 7a).
# Alleen nodig als P7 laat zien dat het basisimage geen docker-client heeft.
# De configuratie en wacht_op_dienst.sh zitten in het image (geen bind mount naar de repository).
FROM docker:29.9.2-cli AS cli

FROM ghcr.io/mostlygeek/llama-swap:v268-cuda13-b11977
ARG CONFIG=p1/llama-swap.yaml
COPY --from=cli /usr/local/bin/docker /usr/local/bin/docker
COPY ${CONFIG} /app/config.yaml
COPY p1/wacht_op_dienst.sh /app/wacht_op_dienst.sh
