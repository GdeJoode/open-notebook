"""
Ontology Registry for managing and caching ontology definitions.

Provides a singleton registry with lookup order: Cache -> DB -> File
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from ontology_manager.schema import Ontology


class OntologyRegistry:
    """
    Singleton registry for ontology management.

    Lookup order:
    1. In-memory cache (fastest)
    2. Database (version-controlled)
    3. File system (ontologies/ directory)
    """

    _instance: Optional["OntologyRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "OntologyRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if OntologyRegistry._initialized:
            return
        OntologyRegistry._initialized = True

        self._cache: Dict[str, Ontology] = {}
        self._ontology_dir = self._find_ontology_dir()
        logger.info(f"OntologyRegistry initialized with dir: {self._ontology_dir}")

    def _find_ontology_dir(self) -> Path:
        """Find the ontologies directory.

        Search order:
        1. Relative to the package root (packages/ontology-manager/ontologies/)
        2. Current working directory (cwd/ontologies/)
        3. ONTOLOGY_DIR environment variable
        4. Falls back to creating the package-relative directory
        """
        # Try relative to the package root:
        # __file__ is src/ontology_manager/registry.py
        # .parent    -> src/ontology_manager/
        # .parent    -> src/
        # .parent    -> packages/ontology-manager/  (package root)
        package_root = Path(__file__).parent.parent.parent
        ontology_dir = package_root / "ontologies"
        if ontology_dir.exists():
            return ontology_dir

        # Try from current working directory
        cwd_dir = Path.cwd() / "ontologies"
        if cwd_dir.exists():
            return cwd_dir

        # Try from ONTOLOGY_DIR environment variable
        env_dir = os.environ.get("ONTOLOGY_DIR")
        if env_dir:
            env_path = Path(env_dir)
            if env_path.exists():
                return env_path

        # Default to package directory (create if needed)
        ontology_dir.mkdir(parents=True, exist_ok=True)
        return ontology_dir

    async def get(
        self,
        name: str,
        version: Optional[str] = None,
        resolve_inheritance: bool = True,
    ) -> Optional[Ontology]:
        """
        Get an ontology by name, optionally with specific version.

        Args:
            name: Ontology name (e.g., "base", "general")
            version: Specific version to retrieve (None = active/latest)
            resolve_inheritance: Whether to merge with parent ontology

        Returns:
            Ontology instance or None if not found
        """
        # OE.0: the key names whether inheritance was resolved. Without it, a
        # `resolve_inheritance=False` call after a resolved one returned the
        # MERGED ontology (and the reverse) — and storing a merged ontology
        # would bake its parents' types into the child.
        cache_key = f"{name}:{version or 'active'}:{'merged' if resolve_inheritance else 'own'}"

        # 1. Check cache
        if cache_key in self._cache:
            logger.debug(f"Ontology '{cache_key}' found in cache")
            return self._cache[cache_key]

        # 2. Try database
        ontology = await self._load_from_db(name, version)
        if ontology:
            logger.debug(f"Ontology '{name}' loaded from database")
            if resolve_inheritance and ontology.metadata.extends:
                ontology = await self._resolve_inheritance(ontology)
            self._cache[cache_key] = ontology
            return ontology

        # 3. Try file system
        ontology = self._load_from_file(name)
        if ontology:
            logger.debug(f"Ontology '{name}' loaded from file")
            if resolve_inheritance and ontology.metadata.extends:
                ontology = await self._resolve_inheritance(ontology)
            self._cache[cache_key] = ontology
            return ontology

        logger.warning(f"Ontology '{name}' not found")
        return None

    async def _load_from_db(
        self, name: str, version: Optional[str] = None
    ) -> Optional[Ontology]:
        """Load ontology from database."""
        try:
            from surrealdb_service.connection import execute_query

            if version:
                query = """
                    SELECT yaml_content FROM ontology_version
                    WHERE name = $name AND version = $version
                    LIMIT 1
                """
                params = {"name": name, "version": version}
            else:
                # Get active version
                query = """
                    SELECT yaml_content FROM ontology_version
                    WHERE name = $name AND is_active = true
                    LIMIT 1
                """
                params = {"name": name}

            result = await execute_query(query, params)
            if result and len(result) > 0:
                yaml_content = result[0].get("yaml_content")
                if yaml_content:
                    return Ontology.from_yaml_string(yaml_content)
        except Exception as e:
            logger.debug(f"Could not load ontology from DB: {e}")
        return None

    def _load_from_file(self, name: str) -> Optional[Ontology]:
        """Load ontology from file system."""
        yaml_path = self._ontology_dir / f"{name}.yaml"
        if yaml_path.exists():
            try:
                return Ontology.from_yaml(yaml_path)
            except Exception as e:
                logger.error(f"Error loading ontology from {yaml_path}: {e}")
        return None

    async def _resolve_inheritance(self, ontology: Ontology) -> Ontology:
        """Resolve ontology inheritance by merging with parent and includes."""
        # Resolve 'extends' (single parent)
        if ontology.metadata.extends:
            parent_name = ontology.metadata.extends
            parent = await self.get(parent_name, resolve_inheritance=True)
            if parent:
                ontology = ontology.merge_with(parent)
            else:
                logger.warning(
                    f"Parent ontology '{parent_name}' not found for '{ontology.metadata.name}'"
                )

        # Resolve 'includes' (multiple siblings)
        if ontology.metadata.includes:
            for include_name in ontology.metadata.includes:
                # Skip if already merged via extends chain
                included = await self.get(include_name, resolve_inheritance=True)
                if included:
                    ontology = ontology.merge_with(included)
                else:
                    logger.warning(
                        f"Included ontology '{include_name}' not found for '{ontology.metadata.name}'"
                    )

        return ontology

    async def register(
        self,
        ontology: Ontology,
        activate: bool = False,
        note: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> bool:
        """Store ``ontology`` as a version in ``ontology_version`` (migration 86).

        ``ontology`` must be the ontology's OWN content, not an
        inheritance-resolved one: get it with ``get(name,
        resolve_inheritance=False)`` or ``Ontology.from_yaml``. A resolved
        ontology would bake its parents' types into this version, and a later
        edit of the parent would never reach it.

        OE.0: stored losslessly — ``exclude_unset`` writes back what the
        source set, including the keys the model has no field for, and no
        defaults it never said; ``mode="json"`` writes enums as their values.
        Activating deactivates the other versions in the same transaction, so
        a failure cannot leave the ontology with none. Every cached ontology is
        dropped, because any ontology that ``extends`` this one cached a merged
        copy of it.

        Returns True on success; errors are logged and return False.
        """
        try:
            import yaml
            from surrealdb_service.connection import execute_query

            yaml_content = yaml.safe_dump(
                ontology.model_dump(mode="json", exclude_unset=True),
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
            await execute_query(
                """
                BEGIN TRANSACTION;
                IF $activate {
                    UPDATE ontology_version SET is_active = false WHERE name = $name;
                };
                INSERT INTO ontology_version {
                    name: $name, version: $version, yaml_content: $yaml_content,
                    is_active: $activate, note: $note, created_by: $created_by
                } ON DUPLICATE KEY UPDATE
                    yaml_content = $input.yaml_content,
                    is_active = $input.is_active,
                    note = $input.note,
                    created_by = $input.created_by;
                COMMIT TRANSACTION;
                """,
                {
                    "name": ontology.metadata.name,
                    "version": ontology.metadata.version,
                    "yaml_content": yaml_content,
                    "activate": activate,
                    "note": note,
                    "created_by": created_by,
                },
            )
            self.clear_cache()
            logger.info(
                f"Registered ontology '{ontology.metadata.name}' v{ontology.metadata.version}"
                + (" (active)" if activate else "")
            )
            return True

        except Exception as e:
            logger.error(f"Error registering ontology: {e}")
            return False

    async def activate_version(self, name: str, version: str) -> bool:
        """Make ``version`` the active version of ``name``.

        OE.0: one transaction that refuses an unknown version BEFORE touching
        the active one — activating a typo used to deactivate every version and
        activate none. Every cached ontology is dropped (see ``register``).
        """
        try:
            from surrealdb_service.connection import execute_query

            await execute_query(
                """
                BEGIN TRANSACTION;
                IF count(SELECT id FROM ontology_version
                         WHERE name = $name AND version = $version) = 0 {
                    THROW "no such ontology version: " + $name + " " + $version;
                };
                UPDATE ontology_version SET is_active = (version = $version)
                    WHERE name = $name;
                COMMIT TRANSACTION;
                """,
                {"name": name, "version": version},
            )
            self.clear_cache()
            logger.info(f"Activated ontology '{name}' version {version}")
            return True

        except Exception as e:
            logger.error(f"Error activating ontology version: {e}")
            return False

    async def list_versions(self, name: str) -> List[Dict]:
        """List all versions of an ontology."""
        try:
            from surrealdb_service.connection import execute_query

            result = await execute_query(
                """
                SELECT name, version, is_active, created_at
                FROM ontology_version
                WHERE name = $name
                ORDER BY created_at DESC
                """,
                {"name": name},
            )
            return result if result else []
        except Exception as e:
            logger.error(f"Error listing ontology versions: {e}")
            return []

    async def list_ontologies(self) -> List[str]:
        """List all available ontologies (from DB and files)."""
        ontologies = set()

        # From database
        try:
            from surrealdb_service.connection import execute_query

            # SurrealDB doesn't support `SELECT DISTINCT` — dedupe client-side.
            result = await execute_query(
                "SELECT name FROM ontology_version", {}
            )
            if result:
                for row in result:
                    name = row.get("name")
                    if name:
                        ontologies.add(name)
        except Exception:
            pass

        # From files
        if self._ontology_dir.exists():
            for yaml_file in self._ontology_dir.glob("*.yaml"):
                ontologies.add(yaml_file.stem)

        return sorted(list(ontologies))

    def clear_cache(self, name: Optional[str] = None) -> None:
        """Clear the ontology cache."""
        if name:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{name}:")]
            for key in keys_to_remove:
                del self._cache[key]
            logger.debug(f"Cleared cache for ontology '{name}'")
        else:
            self._cache.clear()
            logger.debug("Cleared entire ontology cache")


# Module-level singleton accessor
_registry: Optional[OntologyRegistry] = None


def get_ontology_registry() -> OntologyRegistry:
    """Get the singleton ontology registry instance."""
    global _registry
    if _registry is None:
        _registry = OntologyRegistry()
    return _registry
