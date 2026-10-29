"""P6. Feedcontrole voor een YouTube-kanaal en een Spotify-podcast.

Zie docs/signaalsysteem/proeven.md, P6. Draait in een container met internet.
Schrijft één JSON-object naar stdout met metingen en criteria K1-K5, en eindigt
met exitcode 0 (alle criteria gehaald) of 1 (minstens één niet gehaald).
Haalt van YouTube alleen de feed op: geen video, audio of ondertitels (FO 5).
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import feedparser

D3 = "https://www.youtube.com/feeds/videos.xml?channel_id=UCEqXS8PoH57XaizXKMTL_UQ"
D4 = "https://podcast.npo.nl/feed/de-dag.xml"
D5 = "https://open.spotify.com/show/2Oeh3L5u0y9R54jEz8Mgty"
UA = "signaal-proef/0.1 (feedcontrole; openbare feeds)"
WACHT_S = int(os.environ.get("P6_WACHT_S", "60"))

uit: dict = {"metingen": {}, "criteria": {}}


def ophalen(url: str, headers: dict | None = None) -> tuple[int, dict, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, e.read() if e.fp else b""


def tijd(struct) -> datetime | None:
    if not struct:
        return None
    return datetime(*struct[:6], tzinfo=timezone.utc)


def crit(naam: str, geslaagd: bool | None, toelichting: str) -> None:
    stand = "vastgelegd" if geslaagd is None else ("geslaagd" if geslaagd else "mislukt")
    uit["criteria"][naam] = {"stand": stand, "toelichting": toelichting}


def items(feed, soort: str) -> list[dict]:
    res = []
    for e in feed.entries:
        if soort == "youtube":
            ident = e.get("yt_videoid")
        else:
            ident = e.get("id") or e.get("guid")
        res.append(
            {
                "id": ident,
                "titel": e.get("title"),
                "gepubliceerd": tijd(e.get("published_parsed")),
                "beschrijving": bool(e.get("summary") or e.get("media_description")),
                "bijlagen": [
                    {"url": l.get("href"), "type": l.get("type"), "lengte": l.get("length")}
                    for l in e.get("links", [])
                    if l.get("rel") == "enclosure"
                ],
            }
        )
    return res


def nieuw(huidig: list[dict], toestand: set[str]) -> list[dict]:
    return [i for i in huidig if i["id"] and i["id"] not in toestand]


def controle(naam: str, url: str, soort: str) -> dict:
    m: dict = {}
    status, hdr, body = ophalen(url)
    eerste_tijd = datetime.now(timezone.utc)
    m["http_status"] = status
    m["content_type"] = hdr.get("content-type")
    m["etag"] = hdr.get("etag")
    m["last_modified"] = hdr.get("last-modified")
    feed = feedparser.parse(body)
    lijst = items(feed, soort)
    m["aantal_items"] = len(lijst)
    m["zonder_id"] = sum(1 for i in lijst if not i["id"])
    m["zonder_datum"] = sum(1 for i in lijst if not i["gepubliceerd"])
    toestand = {i["id"] for i in lijst if i["id"]}

    # Tweede ophaling na WACHT_S seconden.
    time.sleep(WACHT_S)
    status2, _, body2 = ophalen(url)
    lijst2 = items(feedparser.parse(body2), soort)
    n2 = nieuw(lijst2, toestand)
    m["tweede_ophaling_status"] = status2
    m["tweede_ophaling_nieuw"] = len(n2)
    m["tweede_ophaling_nieuw_na_eerste"] = sum(
        1 for i in n2 if i["gepubliceerd"] and i["gepubliceerd"] > eerste_tijd
    )

    # Kunstmatig: de k nieuwste id's uit de toestand verwijderen.
    op_datum = sorted(
        [i for i in lijst if i["id"]],
        key=lambda i: i["gepubliceerd"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    m["k_proef"] = {}
    for k in (1, 3):
        weg = {i["id"] for i in op_datum[:k]}
        m["k_proef"][str(k)] = len(nieuw(lijst, toestand - weg))

    # Voorwaardelijke aanvraag (alleen vastleggen).
    vw = {}
    if m["etag"]:
        vw["If-None-Match"] = m["etag"]
    if m["last_modified"]:
        vw["If-Modified-Since"] = m["last_modified"]
    if vw:
        s3, _, b3 = ophalen(url, vw)
        m["voorwaardelijk"] = {"headers": sorted(vw), "status": s3, "bytes": len(b3)}
    else:
        m["voorwaardelijk"] = "geen ETag of Last-Modified"

    m["_lijst"] = lijst
    m["_n2"] = n2
    m["_eerste_tijd"] = eerste_tijd
    return m


def normaliseer(s: str | None) -> str:
    s = html.unescape(s or "").lower()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def spotify_naar_rss() -> dict:
    m: dict = {}
    status, _, body = ophalen(D5, {"Accept-Language": "en"})
    m["spotify_status"] = status
    titel = re.search(rb"<title[^>]*>(.*?)</title>", body or b"", re.S)
    pagina_titel = html.unescape(titel.group(1).decode("utf-8", "replace")).strip() if titel else ""
    m["spotify_paginatitel"] = pagina_titel
    shownaam = pagina_titel.split(" | ")[0].strip() if pagina_titel else ""
    m["shownaam"] = shownaam

    s_oe, _, b_oe = ophalen("https://open.spotify.com/oembed?url=" + urllib.parse.quote(D5, safe=""))
    m["oembed_status"] = s_oe
    try:
        oembed_titel = json.loads(b_oe).get("title")
    except ValueError:
        oembed_titel = None
    m["oembed_titel"] = oembed_titel

    zoek = "https://itunes.apple.com/search?" + urllib.parse.urlencode(
        {"term": shownaam, "entity": "podcast", "country": "NL"}
    )
    s_it, _, b_it = ophalen(zoek)
    m["itunes_status"] = s_it
    kandidaten = []
    try:
        kandidaten = [r.get("feedUrl") for r in json.loads(b_it).get("results", []) if r.get("feedUrl")]
    except ValueError:
        pass
    m["itunes_kandidaten"] = kandidaten[:10]
    gevonden = None
    controles = []
    for feed_url in kandidaten[:10]:
        s_f, _, b_f = ophalen(feed_url)
        titels = [e.get("title") for e in feedparser.parse(b_f).entries[:3]]
        treffer = normaliseer(oembed_titel) in {normaliseer(t) for t in titels} if oembed_titel else False
        controles.append({"feed": feed_url, "status": s_f, "nieuwste_titels": titels, "treffer": treffer})
        if treffer:
            gevonden = feed_url
            break
    m["controles"] = controles
    m["gevonden_feed"] = gevonden

    if os.environ.get("PODCASTINDEX_KEY") and os.environ.get("PODCASTINDEX_SECRET"):
        import hashlib

        ts = str(int(time.time()))
        key, sec = os.environ["PODCASTINDEX_KEY"], os.environ["PODCASTINDEX_SECRET"]
        auth = hashlib.sha1((key + sec + ts).encode()).hexdigest()
        s_pi, _, b_pi = ophalen(
            "https://api.podcastindex.org/api/1.0/search/byterm?" + urllib.parse.urlencode({"q": shownaam}),
            {"X-Auth-Key": key, "X-Auth-Date": ts, "Authorization": auth},
        )
        try:
            m["podcastindex_feeds"] = [f.get("url") for f in json.loads(b_pi).get("feeds", [])][:10]
        except ValueError:
            m["podcastindex_feeds"] = None
        m["podcastindex_status"] = s_pi
    else:
        m["podcastindex"] = "overgeslagen: PODCASTINDEX_KEY en PODCASTINDEX_SECRET niet gezet"
    return m


def main() -> int:
    yt = controle("youtube", D3, "youtube")
    pc = controle("podcast", D4, "podcast")

    # K1
    k1 = (
        yt["http_status"] == 200
        and 1 <= yt["aantal_items"] <= 15
        and yt["zonder_id"] == 0
        and yt["zonder_datum"] == 0
    )
    crit("K1", k1, f"YouTube: HTTP {yt['http_status']}, {yt['aantal_items']} items, "
         f"{yt['zonder_id']} zonder videoId, {yt['zonder_datum']} zonder datum")

    # K2
    def k2_ok(m):
        return m["tweede_ophaling_nieuw"] == m["tweede_ophaling_nieuw_na_eerste"]
    crit("K2", k2_ok(yt) and k2_ok(pc),
         f"tweede ophaling nieuw: YouTube {yt['tweede_ophaling_nieuw']} "
         f"(waarvan na eerste ophaling {yt['tweede_ophaling_nieuw_na_eerste']}), podcast "
         f"{pc['tweede_ophaling_nieuw']} (waarvan na eerste {pc['tweede_ophaling_nieuw_na_eerste']})")

    # K3
    k3 = all(yt["k_proef"][str(k)] == k and pc["k_proef"][str(k)] == k for k in (1, 3))
    crit("K3", k3, f"na verwijderen van k id's: YouTube {yt['k_proef']}, podcast {pc['k_proef']}")

    # K4
    lijst = pc["_lijst"]
    zonder_guid = sum(1 for i in lijst if not i["id"])
    zonder_audio = sum(
        1 for i in lijst if not any((b.get("type") or "").startswith("audio/") for b in i["bijlagen"])
    )
    crit("K4", pc["http_status"] == 200 and len(lijst) > 0 and zonder_guid == 0 and zonder_audio == 0,
         f"podcast: HTTP {pc['http_status']}, {len(lijst)} items, {zonder_guid} zonder guid, "
         f"{zonder_audio} zonder audio-bijlage")
    if lijst:
        pc["nieuwste_bijlage"] = lijst[0]["bijlagen"][:1]

    # K5
    sp = spotify_naar_rss()
    crit("K5", sp["gevonden_feed"] == D4,
         f"gevonden feed {sp['gevonden_feed']!r}; oEmbed-titel {sp['oembed_titel']!r}; shownaam {sp['shownaam']!r}")

    # 6d, optioneel
    if os.environ.get("YOUTUBE_API_KEY"):
        q = urllib.parse.urlencode(
            {"part": "snippet", "playlistId": "UU" + D3.split("UC", 1)[1], "maxResults": 15,
             "key": os.environ["YOUTUBE_API_KEY"]}
        )
        s_yd, _, b_yd = ophalen("https://www.googleapis.com/youtube/v3/playlistItems?" + q)
        try:
            n = len(json.loads(b_yd).get("items", []))
        except ValueError:
            n = None
        uit["metingen"]["6d"] = {"status": s_yd, "items": n}
    else:
        uit["metingen"]["6d"] = "overgeslagen: YOUTUBE_API_KEY niet gezet"

    for m in (yt, pc):
        for k in ("_lijst", "_n2", "_eerste_tijd"):
            m.pop(k, None)
    uit["metingen"]["6a_youtube"] = yt
    uit["metingen"]["6b_podcast"] = pc
    uit["metingen"]["6c_spotify"] = sp
    print(json.dumps(uit, default=str, ensure_ascii=False, indent=2))
    return 0 if all(c["stand"] != "mislukt" for c in uit["criteria"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
