"""_resolve_asset_pdf — environment-independent source PDF path resolution."""
from pathlib import Path

from app_main.services.references.reference_extraction_service import (
    _resolve_asset_pdf,
)


def test_returns_path_as_is_when_it_exists(tmp_path: Path) -> None:
    pdf = tmp_path / "uploads" / "paper.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.6\n")
    assert _resolve_asset_pdf(str(pdf)) == str(pdf)


def test_env_override_root_used_when_as_is_missing(tmp_path: Path, monkeypatch) -> None:
    real = tmp_path / "host_uploads" / "convenant.pdf"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"%PDF-1.6\n")
    monkeypatch.setenv("OPEN_NOTEBOOK_UPLOADS_ROOT", str(real.parent))
    # the stored (container-relative) path does not exist as-is
    resolved = _resolve_asset_pdf("data/uploads/convenant.pdf")
    assert resolved == str(real)


def test_mount_inverse_data_to_notebook_data(tmp_path: Path, monkeypatch) -> None:
    # simulate a repo-root run: data/uploads absent, notebook_data/uploads present
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "notebook_data" / "uploads" / "convenant.pdf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"%PDF-1.6\n")
    assert _resolve_asset_pdf("data/uploads/convenant.pdf") == (
        "notebook_data/uploads/convenant.pdf"
    )


def test_returns_original_when_nothing_resolves(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPEN_NOTEBOOK_UPLOADS_ROOT", raising=False)
    original = "data/uploads/missing.pdf"
    # nothing on disk → original returned (GROBID then reports not-found, pass → [])
    assert _resolve_asset_pdf(original) == original
