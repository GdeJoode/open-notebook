"""
Zotero integration service — bidirectional sync with cloud file upload.

Wraps pyzotero for the Zotero Web API v3. All public methods are
synchronous; the FastAPI layer wraps them in asyncio.to_thread().
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pyzotero import zotero

from zotero_integration.dedup import DuplicateDetector


# -------------------------------------------------------------------------
# Item type mapping
# -------------------------------------------------------------------------

_SOURCE_TO_ZOTERO_TYPE = {
    "paper": "journalArticle",
    "article": "journalArticle",
    "book": "book",
    "chapter": "bookSection",
    "report": "report",
    "thesis": "thesis",
    "webpage": "webpage",
    "presentation": "presentation",
    "document": "document",
    "podcast": "podcast",
    "video": "videoRecording",
    "transcript": "document",
    "news_article": "newspaperArticle",
}

_ZOTERO_TO_SOURCE_TYPE = {v: k for k, v in _SOURCE_TO_ZOTERO_TYPE.items()}
_ZOTERO_TO_SOURCE_TYPE.update({
    "conferencePaper": "paper",
    "preprint": "paper",
    "newspaperArticle": "news_article",
    "letter": "correspondence",
    "email": "correspondence",
})


class ZoteroService:
    """Bidirectional Zotero synchronization with cloud file upload."""

    def __init__(
        self,
        library_id: str,
        api_key: str,
        library_type: str = "user",
    ):
        self._zot = zotero.Zotero(library_id, library_type, api_key)
        self._dedup = DuplicateDetector(self._zot)
        self._library_version: Optional[int] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def test_connection(self) -> Dict[str, Any]:
        """Validate credentials and return library info."""
        try:
            # A lightweight call that exercises the API key
            self._zot.items(limit=1)
            version = int(
                self._zot.request.headers.get("Last-Modified-Version", 0)
            )
            self._library_version = version
            return {
                "status": "connected",
                "library_version": version,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @property
    def library_version(self) -> int:
        """Current library version from the most recent API response."""
        if self._library_version is None:
            self.test_connection()
        return self._library_version or 0

    # ------------------------------------------------------------------
    # Push: App → Zotero
    # ------------------------------------------------------------------

    def push_source(
        self,
        source: Dict[str, Any],
        file_path: Optional[str] = None,
        collection_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Push a source to Zotero with duplicate detection.

        Creates a Zotero item from source metadata. If file_path is
        provided, uploads the file to Zotero cloud storage.

        Returns:
            Dict with zotero_key, zotero_version, action.
        """
        metadata = self._source_to_zotero(source)

        # Check for duplicates
        existing = self._dedup.find_duplicate(metadata)
        if existing:
            return {
                "zotero_key": existing["key"],
                "zotero_version": existing["version"],
                "action": "duplicate_found",
                "title": existing.get("data", {}).get("title", ""),
            }

        # Add to collection if specified
        if collection_key:
            metadata["collections"] = [collection_key]

        # Create the item
        result = self._zot.create_items([metadata])

        if "0" not in result.get("successful", {}):
            failed = result.get("failed", {})
            raise RuntimeError(f"Failed to create Zotero item: {failed}")

        created = result["successful"]["0"]
        item_key = created["key"]
        item_version = created["version"]

        logger.info(f"Created Zotero item: {item_key} ({metadata.get('title', '')[:50]})")

        # Upload file to Zotero cloud storage
        if file_path and os.path.exists(file_path):
            try:
                self._zot.attachment_simple([file_path], parentid=item_key)
                logger.info(f"Uploaded file to Zotero cloud: {os.path.basename(file_path)}")
            except Exception as e:
                logger.warning(f"File upload failed (item created without attachment): {e}")

        self._library_version = int(
            self._zot.request.headers.get("Last-Modified-Version", item_version)
        )

        return {
            "zotero_key": item_key,
            "zotero_version": item_version,
            "action": "created",
            "title": metadata.get("title", ""),
        }

    # ------------------------------------------------------------------
    # Pull: Zotero → App
    # ------------------------------------------------------------------

    def pull_changes(self, since_version: int) -> List[Dict[str, Any]]:
        """Get items modified in Zotero since the given library version.

        Returns list of dicts with zotero_key, zotero_version, metadata,
        and action (modified or new).
        """
        modified = self._zot.items(since=since_version)
        self._library_version = int(
            self._zot.request.headers.get("Last-Modified-Version", since_version)
        )

        results = []
        for item in modified:
            if item["data"].get("itemType") == "attachment":
                continue  # Skip attachments, handle via parent
            results.append({
                "zotero_key": item["key"],
                "zotero_version": item["version"],
                "metadata": self._zotero_to_source(item),
                "action": "modified",
            })

        return results

    def pull_deleted(self, since_version: int) -> List[str]:
        """Get keys of items deleted from Zotero since version."""
        try:
            deleted = self._zot.deleted(since=since_version)
            return deleted.get("items", [])
        except Exception as e:
            logger.warning(f"Failed to fetch deleted items: {e}")
            return []

    # ------------------------------------------------------------------
    # Sync: bidirectional
    # ------------------------------------------------------------------

    def sync(
        self,
        local_sources: List[Dict[str, Any]],
        last_sync_version: int,
    ) -> Dict[str, Any]:
        """Bidirectional sync between app and Zotero.

        Args:
            local_sources: Sources from the app database. Each should have
                zotero_key (if previously synced), modified_at, zotero_synced_at.
            last_sync_version: Library version from last sync.

        Returns:
            Dict with pushed, pulled, conflicts, deleted, new_version.
        """
        result: Dict[str, Any] = {
            "pushed": [],
            "pulled": [],
            "conflicts": [],
            "deleted": [],
            "new_version": last_sync_version,
        }

        # 1. Pull remote changes
        remote_changes = self.pull_changes(last_sync_version)
        remote_deleted = self.pull_deleted(last_sync_version)
        remote_by_key = {r["zotero_key"]: r for r in remote_changes}

        # 2. Categorize local sources
        local_modified = [
            s for s in local_sources
            if s.get("zotero_key")
            and s.get("modified_at", "") > (s.get("zotero_synced_at") or "")
        ]
        local_new = [s for s in local_sources if not s.get("zotero_key")]

        # 3. Detect conflicts and push local-only changes
        for local in local_modified:
            zkey = local["zotero_key"]
            if zkey in remote_by_key:
                # Modified on both sides
                result["conflicts"].append({
                    "local": local,
                    "remote": remote_by_key.pop(zkey),
                })
            else:
                # Only modified locally — push update
                try:
                    self._update_zotero_item(zkey, local)
                    result["pushed"].append({
                        "zotero_key": zkey,
                        "action": "updated",
                    })
                except Exception as e:
                    logger.warning(f"Failed to push update for {zkey}: {e}")

        # 4. Remaining remote changes are pull-only
        for remote in remote_by_key.values():
            result["pulled"].append(remote)

        # 5. Push new local items
        for new_source in local_new:
            try:
                push_result = self.push_source(
                    new_source,
                    file_path=new_source.get("file_path"),
                )
                result["pushed"].append(push_result)
            except Exception as e:
                logger.warning(f"Failed to push new source: {e}")

        # 6. Remote deletions
        for deleted_key in remote_deleted:
            result["deleted"].append(deleted_key)

        result["new_version"] = self.library_version
        return result

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    def get_or_create_collection(self, name: str) -> str:
        """Get collection key by name, creating if it doesn't exist."""
        collections = self._zot.collections()
        for col in collections:
            if col["data"]["name"] == name:
                return col["key"]

        result = self._zot.create_collections([{"name": name}])
        return result["successful"]["0"]["key"]

    def list_collections(self) -> List[Dict[str, str]]:
        """List all collections."""
        collections = self._zot.collections()
        return [
            {"key": c["key"], "name": c["data"]["name"]}
            for c in collections
        ]

    # ------------------------------------------------------------------
    # Item listing
    # ------------------------------------------------------------------

    def list_items(self, limit: int = 25, start: int = 0) -> List[Dict[str, Any]]:
        """List items from the Zotero library."""
        items = self._zot.items(limit=limit, start=start)
        return [
            {
                "zotero_key": item["key"],
                "zotero_version": item["version"],
                "metadata": self._zotero_to_source(item),
            }
            for item in items
            if item["data"].get("itemType") != "attachment"
        ]

    # ------------------------------------------------------------------
    # Metadata mapping
    # ------------------------------------------------------------------

    def _source_to_zotero(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Map app source format to Zotero item template."""
        source_type = source.get("type", source.get("source_type", "document"))
        zotero_type = _SOURCE_TO_ZOTERO_TYPE.get(source_type, "document")
        template = self._zot.item_template(zotero_type)

        template["title"] = source.get("title", "Untitled")

        if doi := source.get("doi") or source.get("DOI"):
            template["DOI"] = doi
        if isbn := source.get("isbn") or source.get("ISBN"):
            template["ISBN"] = isbn
        if abstract := source.get("abstract") or source.get("summary"):
            template["abstractNote"] = abstract[:10000]  # Zotero limit
        if url := source.get("url"):
            template["url"] = url
        if date := source.get("date") or source.get("publication_date"):
            template["date"] = str(date)

        # Map authors
        authors = source.get("authors", [])
        if authors:
            creators = []
            for a in authors:
                if isinstance(a, dict):
                    creators.append({
                        "creatorType": "author",
                        "firstName": a.get("first_name", a.get("firstName", "")),
                        "lastName": a.get("last_name", a.get("lastName", "")),
                    })
                elif isinstance(a, str):
                    parts = a.rsplit(" ", 1)
                    creators.append({
                        "creatorType": "author",
                        "firstName": parts[0] if len(parts) > 1 else "",
                        "lastName": parts[-1],
                    })
            if creators:
                template["creators"] = creators

        # Tags from topics
        topics = source.get("topics", [])
        if topics:
            template["tags"] = [{"tag": t} for t in topics[:50]]

        return template

    def _zotero_to_source(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Map Zotero item to app source metadata."""
        data = item.get("data", {})
        return {
            "title": data.get("title", ""),
            "doi": data.get("DOI", ""),
            "isbn": data.get("ISBN", ""),
            "abstract": data.get("abstractNote", ""),
            "url": data.get("url", ""),
            "date": data.get("date", ""),
            "authors": [
                {
                    "first_name": c.get("firstName", ""),
                    "last_name": c.get("lastName", ""),
                }
                for c in data.get("creators", [])
                if c.get("creatorType") == "author"
            ],
            "type": _ZOTERO_TO_SOURCE_TYPE.get(
                data.get("itemType", ""), "document"
            ),
            "tags": [t["tag"] for t in data.get("tags", [])],
            "zotero_item_type": data.get("itemType", ""),
        }

    def _update_zotero_item(
        self, item_key: str, source: Dict[str, Any]
    ) -> None:
        """Update an existing Zotero item with app source data."""
        item = self._zot.item(item_key)
        metadata = self._source_to_zotero(source)

        # Only update fields that have values in the source
        for field in ("title", "DOI", "ISBN", "abstractNote", "url", "date"):
            if metadata.get(field):
                item["data"][field] = metadata[field]

        if metadata.get("creators"):
            item["data"]["creators"] = metadata["creators"]
        if metadata.get("tags"):
            item["data"]["tags"] = metadata["tags"]

        self._zot.update_item(item)
