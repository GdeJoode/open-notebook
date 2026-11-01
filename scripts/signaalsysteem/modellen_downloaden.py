"""Modellen voor llama-swap en docling-serve: downloaden en controleren (F2-11).

Uit ``scripts/signaalsysteem/proeven/p1/modellen_downloaden.py``, met vaste
revisies en de volledige SHA-256 (``proeven.md`` 1.7; de ingekorte waarden daar
kloppen met het begin en einde hieronder, en met de LFS-hash op Hugging Face).
Alleen de standaardbibliotheek; draait in een container met de volumes
gekoppeld (``scripts/signaalsysteem/modellen_laden.sh``).

    modellen_downloaden.py gguf <map> [--bron <map>]
        Per GGUF-bestand: staat het er met de juiste SHA-256, dan niets doen;
        staat het er met een andere, dan weigeren (niets overschrijven); anders
        kopiëren uit --bron (een bestand met dezelfde SHA-256, onder welke naam
        ook) of downloaden op de vaste revisie, en daarna controleren.
    modellen_downloaden.py controle <gguf-map> <docling-map>
        Alleen controleren, zonder netwerk: de GGUF-bestanden en de boom van de
        Docling-modellen (``DOCLING``). Die boom laadt ``docling-tools models
        download`` uit het docling-serve-image; dit script downloadt hem niet.

Exitcodes: 0 in orde, 1 een SHA-256 wijkt af, 2 een bestand ontbreekt of is
niet te halen, 3 scriptfout. Het verslag (JSON) gaat naar stdout.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

#: Bestandsnaam in /modellen (de naam op Hugging Face), repo, revisie, SHA-256, grootte.
GGUF = [
    {
        "bestand": "Qwen3.5-9B-Q4_K_M.gguf",
        "repo": "unsloth/Qwen3.5-9B-GGUF",
        "revisie": "3885219b6810b007914f3a7950a8d1b469d598a5",
        "sha256": "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
        "grootte": 5680522464,
    },
    {
        "bestand": "mxbai-embed-large-v1.Q8_0.gguf",
        "repo": "ChristianAzinn/mxbai-embed-large-v1-gguf",
        "revisie": "3ec6d46af11ba2b982d8c6a1e11183c4995f76a7",
        "sha256": "bcdebca12aa16c0e51d166d97e4776efd46f905952b0c9acb968976eba2619f3",
        "grootte": 358235712,
    },
]

#: De Docling-modellen van docling-serve-cu128:v1.38.2 (``docling-tools models
#: download``, 30-10-2026): SHA-256 over de regels ``<sha256>  ./<pad>`` van alle
#: bestanden, op pad gesorteerd (gelijk aan ``find . -type f | sort | xargs
#: sha256sum | sha256sum``).
DOCLING = {
    "sha256_boom": "41b04b963c87513ae6cfb338e8df8142e850c553cd1870f4e32cdc69f948670c",
    "bestanden": 142,
}

UA = "signaalsysteem/F2-11 (modellen)"


def sha256(pad: Path) -> str:
    h = hashlib.sha256()
    with open(pad, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def boom_sha256(map_: Path) -> tuple[str, int]:
    paden = sorted((p for p in map_.rglob("*") if p.is_file() and not p.is_symlink()),
                   key=lambda p: p.relative_to(map_).as_posix().encode())
    h = hashlib.sha256()
    for p in paden:
        h.update(f"{sha256(p)}  ./{p.relative_to(map_).as_posix()}\n".encode())
    return h.hexdigest(), len(paden)


def _uit_bron(item: dict, bron: Path, doel: Path) -> bool:
    for kandidaat in sorted(bron.iterdir()):
        if (kandidaat.is_file() and kandidaat.stat().st_size == item["grootte"]
                and sha256(kandidaat) == item["sha256"]):
            shutil.copyfile(kandidaat, doel)
            return True
    return False


def _download(item: dict, doel: Path) -> None:
    url = f"https://huggingface.co/{item['repo']}/resolve/{item['revisie']}/{item['bestand']}"
    deel = doel.with_name(doel.name + ".deel")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r, open(deel, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    if sha256(deel) != item["sha256"]:
        deel.unlink()
        raise ValueError("sha256")
    os.replace(deel, doel)


def gguf(map_: Path, bron: Path | None, mag_halen: bool = True) -> tuple[int, list[dict]]:
    code, verslag = 0, []
    for item in GGUF:
        doel = map_ / item["bestand"]
        regel = {"bestand": item["bestand"], "repo": item["repo"], "revisie": item["revisie"]}
        if not doel.exists() and mag_halen:
            if bron is not None and _uit_bron(item, bron, doel):
                regel["herkomst"] = f"kopie uit {bron}"
            else:
                try:
                    _download(item, doel)
                    regel["herkomst"] = "download"
                except ValueError:
                    regel["fout"] = "sha256 van de download wijkt af"
                    code = max(code, 1)
                    verslag.append(regel)
                    continue
                except OSError as e:
                    regel["fout"] = f"niet te halen ({type(e).__name__})"
                    code = max(code, 2)
                    verslag.append(regel)
                    continue
        if not doel.exists():
            regel["fout"] = "ontbreekt"
            code = max(code, 2)
        else:
            som = sha256(doel)
            regel["sha256"] = som
            regel["geslaagd"] = som == item["sha256"]
            if not regel["geslaagd"]:
                code = max(code, 1)
        verslag.append(regel)
    return code, verslag


def docling(map_: Path) -> tuple[int, dict]:
    som, aantal = boom_sha256(map_)
    if aantal == 0:
        return 2, {"fout": "geen Docling-modellen", "map": str(map_)}
    geslaagd = som == DOCLING["sha256_boom"] and aantal == DOCLING["bestanden"]
    return (0 if geslaagd else 1), {"sha256_boom": som, "bestanden": aantal, "geslaagd": geslaagd}


def main(argv: list[str]) -> int:
    if argv[:1] == ["gguf"] and len(argv) == 2:
        code, verslag = gguf(Path(argv[1]), None)
    elif argv[:1] == ["gguf"] and len(argv) == 4 and argv[2] == "--bron":
        code, verslag = gguf(Path(argv[1]), Path(argv[3]))
    elif argv[:1] == ["controle"] and len(argv) == 3:
        c1, v1 = gguf(Path(argv[1]), None, mag_halen=False)
        c2, v2 = docling(Path(argv[2]))
        print(json.dumps({"gguf": v1, "docling": v2}, indent=2))
        return max(c1, c2)
    else:
        print(__doc__, file=sys.stderr)
        return 3
    print(json.dumps({"gguf": verslag}, indent=2))
    return code


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as e:  # scriptfout
        print(f"scriptfout: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(3)
