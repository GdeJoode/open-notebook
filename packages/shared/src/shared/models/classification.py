"""Classificatiebereik van een sleutel (F2-05, ADR 0002 C, plan §5.3).

Het recht van een sleutel bepaalt een :class:`ClassificationScope`. De
leesmethoden van de repositories krijgen dat bereik als verplicht argument,
zonder standaardwaarde, en zetten het met :func:`classifications_for` om in de
lijst die de query filtert (``classification IN $classifications``).
"""

from __future__ import annotations

from enum import Enum
from typing import List


class ClassificationScope(str, Enum):
    """Welke classificaties een sleutel mag zien (ADR 0002 C, plan §5.3)."""

    EXTERN = "EXTERN"
    ALLES = "ALLES"
    LEEG = "LEEG"


_PER_SCOPE = {
    ClassificationScope.EXTERN: ("extern",),
    ClassificationScope.ALLES: ("intern", "extern"),
    ClassificationScope.LEEG: (),
}


def classifications_for(scope: ClassificationScope) -> List[str]:
    """De classificaties die ``scope`` ziet, voor ``$classifications`` in een query.

    Een waarde die geen :class:`ClassificationScope` is, geeft een fout: er is
    geen terugval op 'alles'.
    """
    if not isinstance(scope, ClassificationScope):
        raise TypeError(f"scope moet een ClassificationScope zijn, niet {type(scope).__name__}")
    return list(_PER_SCOPE[scope])


__all__ = ["ClassificationScope", "classifications_for"]
