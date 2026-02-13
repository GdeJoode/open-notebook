"""MCP server for file-manager access."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from file_manager.operations import list_files, read_file
from surrealdb_service.repositories.file_tracking import (
    KnowledgeBaseRepository,
    ProjectRepository,
    TrackedFileRepository,
)


def _get_storage_root() -> str:
    return os.environ.get("STORAGE_ROOT", os.path.expanduser("~/.open-notebook"))


def create_server() -> FastMCP:
    """Create and configure the file-manager MCP server."""
    mcp = FastMCP(
        name="file-manager",
        instructions="File management and workspace access for open-notebook.",
    )

    @mcp.tool()
    async def get_workspace_stats() -> str:
        """Get workspace overview: project count, KB count, temp files, total size."""
        storage_root = _get_storage_root()
        projects_path = Path(storage_root) / "projects"
        kb_path = Path(storage_root) / "knowledgebases"
        temp_path = Path(storage_root) / "temp"

        projects_count = 0
        if projects_path.exists():
            projects_count = sum(1 for d in projects_path.iterdir() if d.is_dir())

        kb_count = 0
        if kb_path.exists():
            kb_count = sum(1 for d in kb_path.iterdir() if d.is_dir())

        temp_count = 0
        total_size = 0
        if temp_path.exists():
            for f in temp_path.rglob("*"):
                if f.is_file():
                    temp_count += 1
                    total_size += f.stat().st_size

        for d in [projects_path, kb_path]:
            if d.exists():
                for f in d.rglob("*"):
                    if f.is_file():
                        total_size += f.stat().st_size

        stats = {
            "storage_root": storage_root,
            "projects_count": projects_count,
            "knowledgebases_count": kb_count,
            "temp_files_count": temp_count,
            "total_size_bytes": total_size,
        }
        return json.dumps(stats, indent=2)

    @mcp.tool()
    async def list_directory(
        path: str, pattern: str = "*", recursive: bool = False
    ) -> str:
        """List files in a directory. Returns file paths with sizes and modified times."""
        files = await list_files(path, pattern, recursive)
        result = []
        for f in files:
            stat = f.stat()
            result.append({
                "name": f.name,
                "path": str(f),
                "size_bytes": stat.st_size,
                "modified": stat.st_mtime,
            })
        return json.dumps(result, default=str, indent=2)

    @mcp.tool()
    async def read_file_content(path: str, max_bytes: int = 50000) -> str:
        """Read text file content (truncated at max_bytes for safety)."""
        content = await read_file(path)
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        if len(content) > max_bytes:
            return content[:max_bytes] + f"\n... [truncated at {max_bytes} bytes]"
        return content

    @mcp.tool()
    async def list_knowledge_bases() -> str:
        """List all knowledge bases."""
        repo = KnowledgeBaseRepository()
        kbs = await repo.get_all()
        return json.dumps([kb.model_dump() for kb in kbs], default=str, indent=2)

    @mcp.tool()
    async def list_projects() -> str:
        """List all projects."""
        repo = ProjectRepository()
        projects = await repo.get_all()
        return json.dumps([p.model_dump() for p in projects], default=str, indent=2)

    @mcp.tool()
    async def list_tracked_files(
        status: Optional[str] = None, limit: int = 50
    ) -> str:
        """List tracked files, optionally filtered by status."""
        from shared.types.enums import FileStatus

        repo = TrackedFileRepository()
        if status:
            file_status = FileStatus(status)
            files = await repo.get_by_status(file_status, limit=limit)
        else:
            files = await repo.get_all(limit=limit)
        return json.dumps([f.model_dump() for f in files], default=str, indent=2)

    @mcp.tool()
    async def get_tracked_file(
        file_id: Optional[str] = None, path: Optional[str] = None
    ) -> str:
        """Get tracked file details by ID or file path."""
        repo = TrackedFileRepository()
        if file_id:
            f = await repo.get(file_id)
        elif path:
            f = await repo.get_by_path(path)
        else:
            return json.dumps({"error": "Provide file_id or path"})
        if not f:
            return json.dumps({"error": "File not found"})
        return json.dumps(f.model_dump(), default=str, indent=2)

    return mcp


def main():
    """Run the MCP server."""
    parser = argparse.ArgumentParser(description="File Manager MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
    )
    parser.add_argument("--port", type=int, default=8201)
    args = parser.parse_args()

    server = create_server()
    transport_kwargs: Dict[str, Any] = {}
    if args.transport != "stdio":
        transport_kwargs["port"] = args.port
    server.run(transport=args.transport, **transport_kwargs)
