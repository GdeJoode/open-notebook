"""Foutmeldingen zonder inhoud (F2-05, ADR 0002 C).

Een uitzondering kan tekst uit een intern record bevatten. Daarom gaat de tekst
van een fout nooit in een antwoord: de aanroeper krijgt een vaste melding met
een fout-id, en de volledige fout gaat alleen naar het lokale log, onder dat id.
``tests/test_geen_uitzonderingstekst_in_detail.py`` bewaakt dat routers geen
uitzonderingstekst in ``detail`` zetten.

Gebruik in een router::

    except Exception as e:
        raise fout_zonder_inhoud(e, "Zoeken mislukt")

Een fout die de app zelf al goed afhandelt, gaat ongewijzigd door: een
``HTTPException``, een classificatieconflict (409) of ongeldige classificatie
(422) en een onbereikbare database (503). Zo maakt een router met
``except Exception`` van een ``ClassificationConflictError`` geen 500.
"""

from __future__ import annotations

import uuid

from fastapi import HTTPException
from loguru import logger
from surrealdb_service.connection import (
    ClassificationConflictError,
    ClassificationValueError,
    DatabaseUnavailableError,
)

#: Fouten met een eigen afhandeling in ``app.py`` of een eigen statuscode.
DOORLATEN = (
    HTTPException,
    ClassificationConflictError,
    ClassificationValueError,
    DatabaseUnavailableError,
)


def nieuw_fout_id() -> str:
    """Een kort, willekeurig id dat antwoord en logregel verbindt."""
    return uuid.uuid4().hex[:12]


def melding_met_id(melding: str, fout_id: str) -> str:
    """De tekst voor ``detail``: de vaste melding en het fout-id."""
    return f"{melding} (fout-id {fout_id})"


def log_fout(fout: BaseException, melding: str, *, status_code: int = 500) -> str:
    """Schrijf de volledige fout naar het lokale log; geef het fout-id terug."""
    fout_id = nieuw_fout_id()
    if status_code >= 500:
        logger.opt(exception=fout).error(f"{melding} [fout-id {fout_id}]")
    else:
        logger.warning(f"{melding} [fout-id {fout_id}]: {type(fout).__name__}: {fout}")
    return fout_id


def fout_zonder_inhoud(
    fout: BaseException, melding: str = "Interne fout", *, status_code: int = 500
) -> HTTPException:
    """Een ``HTTPException`` met alleen ``melding`` en een fout-id.

    Een fout uit :data:`DOORLATEN` wordt opnieuw opgeworpen in plaats van
    omgezet. Roep aan binnen het ``except``-blok, als ``raise
    fout_zonder_inhoud(e, ...)``.
    """
    if isinstance(fout, DOORLATEN):
        raise fout
    fout_id = log_fout(fout, melding, status_code=status_code)
    return HTTPException(status_code=status_code, detail=melding_met_id(melding, fout_id))


__all__ = [
    "DOORLATEN",
    "fout_zonder_inhoud",
    "log_fout",
    "melding_met_id",
    "nieuw_fout_id",
]
