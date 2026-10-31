"""Het importeindpunt (F2-14, ADR 0010, plan §5.1).

``POST /api/v1/import`` (alleen recht ``pijplijn``; een ``beheerder`` krijgt 403 en
corrigeert via de handmatige run, ADR 0005): 201 bij een nieuwe versie, 200
bij ``ongewijzigd`` of een herhaald verzoek. Een 409 en een 422 hebben een
vaste ``code`` en melding, zonder inhoud van het verzoek (ADR 0002 C).
"""

from fastapi import APIRouter, Depends, HTTPException, Response
from shared.models.agents import Right

from app_main.api.agent_auth import AgentKeyContext, require_right
from app_main.api.fouten import fout_zonder_inhoud
from app_main.api.schemas import ImportRequest, ImportResponse
from app_main.dependencies import get_import_service
from app_main.services.import_service import ImportService, ImportWeigering
from app_main.services.inname_binding import BindingError, ItemStatusError

router = APIRouter(prefix="/v1/import", tags=["import"])


@router.post("", status_code=201, response_model=ImportResponse)
async def importeer(
    body: ImportRequest,
    response: Response,
    key: AgentKeyContext = Depends(require_right(Right.PIJPLIJN)),
    service: ImportService = Depends(get_import_service),
) -> ImportResponse:
    if key.permission != Right.PIJPLIJN:
        raise HTTPException(status_code=403, detail="Alleen de pijplijn importeert")
    try:
        uitkomst = await service.importeer(body, key_id=key.key_id)
    except ImportWeigering as e:
        raise HTTPException(status_code=e.status, detail=e.detail) from None
    except BindingError:
        raise HTTPException(status_code=409, detail={
            "code": "niet_gebonden",
            "melding": "Run, regel, batch, item of bron horen niet bij elkaar"}) from None
    except ItemStatusError:
        raise HTTPException(status_code=409, detail={
            "code": "verkeerde_status", "melding": "Status staat dit niet toe"}) from None
    except Exception as e:  # noqa: BLE001 - zonder inhoud naar de aanroeper
        raise fout_zonder_inhoud(e, "Import mislukt") from None
    if not uitkomst.nieuw:
        response.status_code = 200
    return ImportResponse(
        status=uitkomst.status, source_id=uitkomst.source_id,
        source_version_id=uitkomst.source_version_id, version_no=uitkomst.version_no,
        run_id=uitkomst.run_id, classification=uitkomst.classification,
        chunk_ids=uitkomst.chunk_ids,
    )
