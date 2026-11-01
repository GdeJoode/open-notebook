# llama-swap met de Docker-client (F2-11; ADR 0006, 0009; P1 en P7).
#
# Het basisimage heeft geen `docker` (P7, K1). De client praat alleen met de
# socket-proxy (DOCKER_HOST=tcp://socket-proxy:2375, configuratie B); de
# Docker-socket zelf komt nooit in deze container.
# De configuratie zit alleen-lezen in het image, zonder -watch-config: wijzigen
# gaat met een commit en `scripts/signaalsysteem/start.sh --build`. controle/gpu.sh
# vergelijkt de configuratie in de container met die in HEAD.
FROM docker:29.9.2-cli@sha256:5e6baf67325a79f64ed3bed6863954d5e05d0f3d4e57cbdb14809346814bbfdb AS cli

# Eist cuda>=13.4; zolang de driver 13.3 levert, zet Compose
# NVIDIA_DISABLE_REQUIRE en LD_LIBRARY_PATH (ADR 0006, proeven.md §8 punt 23; F2-25).
FROM ghcr.io/mostlygeek/llama-swap:v268-cuda13-b11977@sha256:b6159c6d225576ade4e3eb70f92b22741b06202235f04d6fff092e5191a76dfb
COPY --from=cli /usr/local/bin/docker /usr/local/bin/docker
COPY --chmod=444 config.yaml /app/config.yaml
COPY --chmod=555 wacht_op_dienst.sh /app/wacht_op_dienst.sh
ENTRYPOINT ["/app/llama-swap", "-config", "/app/config.yaml"]
