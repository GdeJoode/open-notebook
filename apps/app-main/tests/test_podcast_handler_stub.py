"""Tests for the podcast handler stub."""

import ast
import inspect
from pathlib import Path

import pytest


class TestPodcastHandlerStub:
    @pytest.mark.asyncio
    async def test_returns_disabled_error(self):
        from app_main.handlers import handle_generate_podcast

        result = await handle_generate_podcast({"episode_name": "test"})
        assert result["success"] is False
        assert "temporarily disabled" in result["error"].lower()

    def test_no_monolith_imports_in_handlers(self):
        """Ensure handlers.py has zero 'from open_notebook' imports."""
        handlers_path = (
            Path(__file__).parent.parent
            / "src"
            / "app_main"
            / "handlers.py"
        )
        source = handlers_path.read_text()
        tree = ast.parse(source)

        monolith_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("open_notebook"):
                    monolith_imports.append(node.module)

        assert monolith_imports == [], (
            f"Monolith imports still present: {monolith_imports}"
        )
