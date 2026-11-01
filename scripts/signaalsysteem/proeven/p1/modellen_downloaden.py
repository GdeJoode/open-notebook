"""Voorbereiden van P1 en P3: modellen en testaudio downloaden (met internet).

Zie docs/signaalsysteem/proeven.md 1.6 en 1.7. Draait in de container
p1-modellen-voorbereiden op het netwerk "uitgaand" met de volumes:
  /models  GGUF-bestanden onder vaste lokale namen
  /hf      HF_HOME van Kev (adapter kev-4b en basismodel)
  /audio   testaudio D4 (nieuwste aflevering De Dag), D7 (nl) en D8 (en)

De SHA-256 van de GGUF-bestanden wordt gecontroleerd tegen het begin en het einde
uit proeven.md 1.7 (daar ingekort weergegeven). Schrijft een JSON-verslag naar stdout.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request

import feedparser
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

GGUF = [
    # (lokale naam, repo, revisie, patroon, sha256-begin, sha256-einde)
    ("werkmodel-qwen3.5-9b-q4_k_m.gguf", "unsloth/Qwen3.5-9B-GGUF", None, r"^Qwen3\.5-9B-Q4_K_M\.gguf$", "03b74727", "b7e8"),
    ("embed-mxbai-q8_0.gguf", "ChristianAzinn/mxbai-embed-large-v1-gguf",
     "3ec6d46af11ba2b982d8c6a1e11183c4995f76a7", r"(?i)q8_0\.gguf$", None, None),
    ("embed-bge-m3-q8_0.gguf", "gpustack/bge-m3-GGUF", None, r"^bge-m3-Q8_0\.gguf$", "950f4a8e", "7821167"),
    ("rerank-bge-reranker-v2-m3-q8_0.gguf", "gpustack/bge-reranker-v2-m3-GGUF", None,
     r"^bge-reranker-v2-m3-Q8_0\.gguf$", "a43c7c9b", "aaa1d3"),
]
KEV = ("jaredpalmer/kev-4b", "6cfce5c2fa4b4bd64026336ab649c5ca78857d52")
BASIS = ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b")
AUDIO = {
    "D7-nl.ogg": "https://upload.wikimedia.org/wikipedia/commons/5/58/Nl-Fractievoorzitter.ogg",
    "D8-en.ogg": "https://upload.wikimedia.org/wikipedia/commons/4/47/Spoken_Wikipedia_en_Elongate_carpet_shark.ogg",
}
D4_FEED = "https://podcast.npo.nl/feed/de-dag.xml"
UA = "signaal-proef/0.1 (openbare testdata)"


def sha256(pad: str) -> str:
    h = hashlib.sha256()
    with open(pad, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def download(url: str, doel: str) -> None:
    if os.path.exists(doel) and os.path.getsize(doel) > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r, open(doel + ".deel", "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(doel + ".deel", doel)


def main() -> int:
    verslag: dict = {"gguf": [], "kev": {}, "audio": {}}
    fout = False
    for naam, repo, rev, patroon, begin, einde in GGUF:
        doel = os.path.join("/models", naam)
        bestanden = [b for b in list_repo_files(repo, revision=rev) if re.search(patroon, b)]
        if len(bestanden) != 1:
            verslag["gguf"].append({"naam": naam, "fout": f"{len(bestanden)} bestanden passen: {bestanden}"})
            fout = True
            continue
        if not os.path.exists(doel):
            pad = hf_hub_download(repo, bestanden[0], revision=rev, cache_dir="/models/.hf-cache")
            shutil.copyfile(pad, doel)
        som = sha256(doel)
        ok = (begin is None or som.startswith(begin)) and (einde is None or som.endswith(einde))
        fout |= not ok
        verslag["gguf"].append({"naam": naam, "repo": repo, "bestand": bestanden[0], "sha256": som,
                                "grootte": os.path.getsize(doel), "controle": ok})
    shutil.rmtree("/models/.hf-cache", ignore_errors=True)

    # HF_HOME wordt bij het importeren van huggingface_hub gelezen; daarom de cachemap
    # expliciet: /hf/hub is de cache die Kev met HF_HOME=/hf offline gebruikt.
    adapter = snapshot_download(KEV[0], revision=KEV[1], local_dir="/hf/kev-4b")
    basis = snapshot_download(BASIS[0], revision=BASIS[1], cache_dir="/hf/hub")
    verslag["kev"] = {"adapter": adapter, "basis": basis}

    for naam, url in AUDIO.items():
        download(url, os.path.join("/audio", naam))
        verslag["audio"][naam] = {"url": url, "sha256": sha256(os.path.join("/audio", naam))}
    feed = feedparser.parse(urllib.request.urlopen(urllib.request.Request(D4_FEED, headers={"User-Agent": UA})).read())
    bijlage = next(l["href"] for l in feed.entries[0].links if l.get("rel") == "enclosure")
    download(bijlage, "/audio/D4-de-dag.mp3")
    verslag["audio"]["D4-de-dag.mp3"] = {"url": bijlage, "titel": feed.entries[0].title,
                                         "sha256": sha256("/audio/D4-de-dag.mp3")}
    print(json.dumps(verslag, indent=2))
    return 1 if fout else 0


if __name__ == "__main__":
    sys.exit(main())
