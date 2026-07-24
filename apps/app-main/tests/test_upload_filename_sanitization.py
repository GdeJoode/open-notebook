"""Security regression: upload filename sanitization (path traversal).

`generate_unique_filename` takes the UNTRUSTED multipart filename. A crafted
`../../etc/cron.d/x` must NOT write outside the uploads folder. Reached by both
the UI upload route and the G.3b agent upload endpoints, so this guards both.
"""

from __future__ import annotations

from pathlib import Path

from app_main.api.routers.sources_upload import generate_unique_filename


def _uploads(tmp_path) -> str:
    return str(tmp_path / "uploads")


def test_strips_path_traversal(tmp_path):
    uploads = _uploads(tmp_path)
    root = (tmp_path / "uploads").resolve()
    for evil in [
        "../../etc/cron.d/pwn",
        "../../../tmp/escape.txt",
        "/etc/passwd",
        "foo/../../bar.txt",
        "a/b/c/deep.pdf",
    ]:
        result = Path(generate_unique_filename(evil, uploads)).resolve()
        # The write lands directly inside the uploads folder — never above it.
        assert result.parent == root, f"{evil!r} escaped to {result}"
        assert ".." not in result.parts


def test_empty_or_dot_basename_falls_back(tmp_path):
    uploads = _uploads(tmp_path)
    for weird in ["..", ".", "../", "/", ""]:
        result = Path(generate_unique_filename(weird, uploads))
        assert result.name == "upload"
        assert result.parent.resolve() == (tmp_path / "uploads").resolve()


def test_normal_filename_preserved(tmp_path):
    uploads = _uploads(tmp_path)
    result = Path(generate_unique_filename("Research Paper.pdf", uploads))
    assert result.name == "Research Paper.pdf"
    assert result.parent.resolve() == (tmp_path / "uploads").resolve()


def test_counter_variant_stays_contained(tmp_path):
    uploads = _uploads(tmp_path)
    Path(uploads).mkdir(parents=True, exist_ok=True)
    (Path(uploads) / "dup.txt").write_text("x")
    result = Path(generate_unique_filename("dup.txt", uploads))
    assert result.name == "dup (1).txt"  # counter appended, still a basename
    assert result.parent.resolve() == (tmp_path / "uploads").resolve()
