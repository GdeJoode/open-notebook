# P2: Langflow 1.13.0 met de component TickLogger (docs/signaalsysteem/proeven.md, P2).
# Geen bind mount naar de repository: de component zit in het image.
FROM langflowai/langflow:1.13.0
USER root
RUN mkdir -p /proef/components/proef /proef/ticks && chmod -R a+rwX /proef
COPY p2/components/tick_logger.py /proef/components/proef/tick_logger.py
RUN chmod -R a+rX /proef/components
# Oorspronkelijke gebruiker van het image (uid 1000, "user").
USER 1000
