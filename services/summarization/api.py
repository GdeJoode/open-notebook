"""
Summarization service.

Standalone HTTP API that accepts a document file (markdown, plain text,
or pre-chunked JSON/XML) and produces two summary files in the same
directory using RAPTOR and TreeKG strategies.

LLM and embedding endpoints are configured via environment variables
and can be overridden per request.
"""

import json
import os
import re
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from esperanto import OllamaEmbeddingModel, OpenAIEmbeddingModel
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from summarization.config import LLMConfig, RaptorConfig, SummarizationConfig, TreeKGConfig
from summarization.models.result import ChunkInput, SummarizationResult
from summarization.raptor.strategy import RaptorStrategy
from summarization.treekg.strategy import TreeKGStrategy

from presets import get_preset_prompts, list_presets

app = FastAPI(
    title="Summarization Service",
    description="Document summarization via RAPTOR and TreeKG strategies",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Environment-based defaults
# ---------------------------------------------------------------------------

DEFAULT_LLM_PROVIDER = os.getenv("SUMMARIZATION_PROVIDER", "ollama")
DEFAULT_LLM_MODEL = os.getenv("SUMMARIZATION_MODEL", "qwen2.5:14b")
DEFAULT_LLM_BASE_URL = os.getenv("SUMMARIZATION_BASE_URL", "http://host.docker.internal:11434")
DEFAULT_LLM_TEMPERATURE = float(os.getenv("SUMMARIZATION_TEMPERATURE", "0.3"))
DEFAULT_LLM_MAX_TOKENS = int(os.getenv("SUMMARIZATION_MAX_TOKENS", "500"))

DEFAULT_EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
DEFAULT_EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://host.docker.internal:11434")

# Optional API key for cloud providers (Anthropic, OpenAI, etc.)
DEFAULT_API_KEY = os.getenv("LLM_API_KEY", "")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class LLMOverride(BaseModel):
    """Optional per-request LLM endpoint override."""

    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class EmbeddingOverride(BaseModel):
    """Optional per-request embedding endpoint override."""

    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class SummarizeRequest(BaseModel):
    file_path: str = Field(
        ...,
        description="Path to input file (md, txt, json, or xml) inside the container",
    )
    preset: Optional[str] = Field(
        default="auto",
        description="Preset name ('meeting_notes', 'academic_paper', 'policy_brief', 'report'), "
                    "'auto' for auto-detection, or null/empty to use defaults",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt (overrides preset system_prompt)",
    )
    output_instructions: Optional[str] = Field(
        default=None,
        description="Custom output instructions appended to every LLM prompt (overrides preset)",
    )
    llm: Optional[LLMOverride] = Field(
        default=None,
        description="Override default LLM endpoint for this request",
    )
    embedding: Optional[EmbeddingOverride] = Field(
        default=None,
        description="Override default embedding endpoint for this request",
    )


class StrategySummary(BaseModel):
    strategy: str
    success: bool
    output_file: Optional[str] = None
    document_summary: Optional[str] = None
    num_nodes: int = 0
    num_layers: int = 0
    error: Optional[str] = None
    processing_seconds: float = 0.0


class SummarizeResponse(BaseModel):
    input_file: str
    output_dir: str
    strategies: List[StrategySummary]
    total_seconds: float


# ---------------------------------------------------------------------------
# Chunking helpers for raw text input
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)", re.MULTILINE)


def _chunks_from_markdown(text: str) -> List[ChunkInput]:
    """Split markdown into chunks based on headings.

    Each heading starts a new chunk. The section_path is built from
    the heading hierarchy so TreeKG can use it.
    """
    lines = text.split("\n")
    chunks: List[ChunkInput] = []
    current_text_lines: List[str] = []
    heading_stack: List[tuple] = []  # (level, title)
    order = 0

    def _flush():
        nonlocal order
        content = "\n".join(current_text_lines).strip()
        if not content:
            return
        section_path = [h[1] for h in heading_stack] if heading_stack else None
        section_level = heading_stack[-1][0] if heading_stack else None
        chunks.append(
            ChunkInput(
                text=content,
                chunk_id=str(uuid.uuid4()),
                order=order,
                section_path=section_path,
                section_level=section_level,
            )
        )
        order += 1

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            _flush()
            current_text_lines = []
            level = len(m.group(1))
            title = m.group(2).strip()
            # Pop headings at same or deeper level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_text_lines.append(line)
        else:
            current_text_lines.append(line)

    _flush()

    # Fallback: if no headings found, split by double newlines
    if not chunks:
        chunks = _chunks_from_plain_text(text)

    return chunks


def _chunks_from_plain_text(text: str) -> List[ChunkInput]:
    """Split plain text into chunks by double newlines (paragraphs)."""
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks = []
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
        chunks.append(
            ChunkInput(
                text=para,
                chunk_id=str(uuid.uuid4()),
                order=i,
            )
        )
    return chunks


def _chunks_from_json(raw: str) -> List[ChunkInput]:
    """Parse JSON input — supports both pre-chunked arrays and docling document.json.

    Docling format detection: dict with "pages" key containing elements with
    "content", "section_path", "type" fields.

    Pre-chunked format: JSON array of objects with "text" field, or wrapped
    in {"chunks": [...]}.
    """
    data = json.loads(raw)

    # Detect docling document.json format
    if isinstance(data, dict) and "pages" in data:
        return _chunks_from_docling_json(data)

    # Pre-chunked array format
    if isinstance(data, dict):
        data = data.get("chunks", data.get("items", [data]))
    if not isinstance(data, list):
        raise ValueError("JSON input must be an array or contain a 'chunks' array")

    chunks = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            chunks.append(ChunkInput(text=item, chunk_id=str(uuid.uuid4()), order=i))
        elif isinstance(item, dict):
            chunks.append(
                ChunkInput(
                    text=item.get("text", item.get("content", "")),
                    chunk_id=item.get("chunk_id", item.get("id", str(uuid.uuid4()))),
                    order=item.get("order", i),
                    section_path=item.get("section_path"),
                    section_level=item.get("section_level"),
                    metadata=item.get("metadata", {}),
                )
            )
        else:
            raise ValueError(f"Unexpected item type in JSON array: {type(item)}")
    return chunks


def _chunks_from_docling_json(doc: dict) -> List[ChunkInput]:
    """Parse a docling document.json into section-grouped chunks.

    Groups consecutive elements by section_path, concatenating their text
    into one chunk per section. This preserves the real document hierarchy
    for TreeKG instead of page-based splitting.
    """
    # Flatten all elements across pages in order
    all_elements = []
    for page in doc.get("pages", []):
        for elem in page.get("elements", []):
            content = (elem.get("content") or "").strip()
            if not content:
                continue
            all_elements.append(elem)

    if not all_elements:
        raise ValueError("No content elements found in docling JSON")

    # Group consecutive elements by section_path
    chunks: List[ChunkInput] = []
    current_section: Optional[List[str]] = None
    current_texts: List[str] = []
    current_level: Optional[int] = None
    order = 0

    def _flush_section():
        nonlocal order
        if not current_texts:
            return
        combined = "\n\n".join(current_texts)
        chunks.append(
            ChunkInput(
                text=combined,
                chunk_id=str(uuid.uuid4()),
                order=order,
                section_path=current_section,
                section_level=current_level,
            )
        )
        order += 1

    for elem in all_elements:
        section_path = elem.get("section_path") or []
        section_level = elem.get("section_level")
        content = elem.get("content", "").strip()
        elem_type = elem.get("type", "")

        # Skip non-content elements
        if elem_type in ("page_header", "page_footer", "footnote"):
            continue

        # New section detected
        if section_path != current_section:
            _flush_section()
            current_section = section_path if section_path else None
            current_texts = []
            current_level = section_level

        current_texts.append(content)

    _flush_section()

    logger.info(
        f"Docling JSON: {len(all_elements)} elements grouped into "
        f"{len(chunks)} section chunks"
    )
    return chunks


def _chunks_from_xml(raw: str) -> List[ChunkInput]:
    """Parse pre-chunked XML input.

    Expected format:
    <chunks>
      <chunk id="..." order="0">
        <text>...</text>
        <section_path>Heading 1/Heading 2</section_path>
      </chunk>
    </chunks>
    """
    root = ET.fromstring(raw)
    chunks = []

    elements = root.findall(".//chunk")
    if not elements:
        # Maybe the root itself contains text
        elements = root.findall(".//*[text]") or [root]

    for i, elem in enumerate(elements):
        text_el = elem.find("text")
        text = (text_el.text or "").strip() if text_el is not None else (elem.text or "").strip()
        if not text:
            continue

        chunk_id = elem.get("id", str(uuid.uuid4()))
        order = int(elem.get("order", str(i)))

        section_path = None
        sp_el = elem.find("section_path")
        if sp_el is not None and sp_el.text:
            section_path = [s.strip() for s in sp_el.text.split("/")]

        section_level = None
        sl_el = elem.find("section_level")
        if sl_el is not None and sl_el.text:
            section_level = int(sl_el.text)

        chunks.append(
            ChunkInput(
                text=text,
                chunk_id=chunk_id,
                order=order,
                section_path=section_path,
                section_level=section_level,
            )
        )
    return chunks


def load_chunks(file_path: Path) -> List[ChunkInput]:
    """Load and chunk a file based on its extension."""
    raw = file_path.read_text(encoding="utf-8")
    suffix = file_path.suffix.lower()

    if suffix == ".json":
        return _chunks_from_json(raw)
    elif suffix == ".xml":
        return _chunks_from_xml(raw)
    elif suffix == ".md":
        return _chunks_from_markdown(raw)
    else:
        # .txt or anything else: try markdown first, fall back to plain text
        chunks = _chunks_from_markdown(raw)
        if not chunks:
            chunks = _chunks_from_plain_text(raw)
        return chunks


# ---------------------------------------------------------------------------
# LLM / embedding factory
# ---------------------------------------------------------------------------


def _resolve_llm_config(override: Optional[LLMOverride] = None) -> LLMConfig:
    """Build LLMConfig from env defaults + optional per-request override."""
    return LLMConfig(
        model_name=override.model if override and override.model else DEFAULT_LLM_MODEL,
        provider=override.provider if override and override.provider else DEFAULT_LLM_PROVIDER,
        base_url=override.base_url if override and override.base_url else DEFAULT_LLM_BASE_URL,
        temperature=override.temperature if override and override.temperature is not None else DEFAULT_LLM_TEMPERATURE,
        max_tokens=override.max_tokens if override and override.max_tokens is not None else DEFAULT_LLM_MAX_TOKENS,
    )


def _make_embedding_fn(override: Optional[EmbeddingOverride] = None):
    """Create an async embedding function for RAPTOR."""
    provider = override.provider if override and override.provider else DEFAULT_EMBEDDING_PROVIDER
    model = override.model if override and override.model else DEFAULT_EMBEDDING_MODEL
    base_url = override.base_url if override and override.base_url else DEFAULT_EMBEDDING_BASE_URL
    api_key = override.api_key if override and override.api_key else DEFAULT_API_KEY

    if provider == "ollama":
        em = OllamaEmbeddingModel(model_name=model, base_url=base_url)
    elif provider in ("openai", "anthropic"):
        em = OpenAIEmbeddingModel(model_name=model, api_key=api_key, base_url=base_url or None)
    else:
        # Fall back to Ollama
        em = OllamaEmbeddingModel(model_name=model, base_url=base_url)

    async def embedding_fn(texts: List[str]) -> List[List[float]]:
        result = await em.aembed(texts)
        return [r.embedding for r in result]

    return embedding_fn


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------


MAX_CHUNK_CHARS = 2000  # Safe limit for embedding models (~500 tokens)


def _split_large_chunks(chunks: List[ChunkInput]) -> List[ChunkInput]:
    """Split chunks that exceed the embedding model's context limit.

    Preserves section_path metadata on all sub-chunks so they can still
    be traced back to their source section.
    """
    result = []
    order = 0
    for chunk in chunks:
        if len(chunk.text) <= MAX_CHUNK_CHARS:
            result.append(chunk.model_copy(update={"order": order}))
            order += 1
        else:
            # Split by paragraphs, accumulating until limit
            paragraphs = chunk.text.split("\n\n")
            current_parts: List[str] = []
            current_len = 0
            for para in paragraphs:
                if current_len + len(para) > MAX_CHUNK_CHARS and current_parts:
                    result.append(
                        ChunkInput(
                            text="\n\n".join(current_parts),
                            chunk_id=str(uuid.uuid4()),
                            order=order,
                            section_path=chunk.section_path,
                            section_level=chunk.section_level,
                        )
                    )
                    order += 1
                    current_parts = []
                    current_len = 0
                current_parts.append(para)
                current_len += len(para)
            if current_parts:
                result.append(
                    ChunkInput(
                        text="\n\n".join(current_parts),
                        chunk_id=str(uuid.uuid4()),
                        order=order,
                        section_path=chunk.section_path,
                        section_level=chunk.section_level,
                    )
                )
                order += 1
    return result


async def _run_raptor(
    chunks: List[ChunkInput],
    llm_config: LLMConfig,
    embedding_override: Optional[EmbeddingOverride],
    system_prompt: str = "",
    output_instructions: str = "",
) -> SummarizationResult:
    """Run RAPTOR strategy."""
    config = SummarizationConfig(
        strategy="raptor", llm=llm_config,
        system_prompt=system_prompt, output_instructions=output_instructions,
    )
    embedding_fn = _make_embedding_fn(embedding_override)
    strategy = RaptorStrategy(config, embedding_fn=embedding_fn)
    raptor_chunks = _split_large_chunks(chunks)
    logger.info(f"RAPTOR: {len(chunks)} section chunks split into {len(raptor_chunks)} embedding chunks")
    return await strategy.summarize(raptor_chunks)


async def _run_treekg(
    chunks: List[ChunkInput],
    llm_config: LLMConfig,
    system_prompt: str = "",
    output_instructions: str = "",
) -> SummarizationResult:
    """Run TreeKG strategy."""
    config = SummarizationConfig(
        strategy="treekg", llm=llm_config,
        system_prompt=system_prompt, output_instructions=output_instructions,
    )
    strategy = TreeKGStrategy(config)
    return await strategy.summarize(chunks)


def _write_summary(result: SummarizationResult, output_path: Path) -> None:
    """Write a summary result to a markdown file."""
    parts = []
    parts.append(f"# Summary ({result.strategy.value.upper()})")
    parts.append("")
    parts.append(f"> Strategy: {result.strategy.value}")
    parts.append(f"> Input chunks: {result.num_input_chunks}")
    parts.append(f"> Summary nodes: {result.node_count}")
    parts.append(f"> Layers: {result.num_layers}")
    parts.append("")

    if result.document_summary:
        parts.append("## Document Summary")
        parts.append("")
        parts.append(result.document_summary)
        parts.append("")

    if result.summary_nodes and result.has_hierarchy:
        parts.append("## Summary Tree")
        parts.append("")
        for node in result.summary_nodes:
            indent = "  " * node.layer
            title = f" ({node.section_title})" if node.section_title else ""
            parts.append(f"{indent}- **Layer {node.layer}{title}**: {node.text}")
        parts.append("")

    output_path.write_text("\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "service": "summarization"}


@app.get("/presets")
def get_presets():
    """List available summarization presets."""
    presets = []
    for name in list_presets():
        try:
            from presets import load_preset
            p = load_preset(name)
            presets.append({
                "name": name,
                "description": p.get("description", ""),
            })
        except Exception:
            presets.append({"name": name, "description": ""})
    return {"presets": presets, "auto_detect": True}


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    """Summarize a file using both RAPTOR and TreeKG strategies.

    Output files are written to the same directory as the input file.
    Preset auto-detection classifies the input to pick the right prompt template.
    """
    file_path = Path(req.file_path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {req.file_path}")

    # Load and chunk
    try:
        chunks = load_chunks(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse input file: {e}")

    if not chunks:
        raise HTTPException(status_code=400, detail="No content found in input file")

    logger.info(f"Loaded {len(chunks)} chunks from {file_path.name}")

    # Resolve prompt customization: explicit > preset > defaults
    full_text = "\n".join(c.text for c in chunks[:20])  # Sample for detection
    has_speakers = any("SPEAKER_" in c.text for c in chunks[:20])

    preset_system, preset_instructions = get_preset_prompts(
        preset_name=req.preset,
        text=full_text,
        has_speakers=has_speakers,
    )

    # Explicit per-request overrides take precedence over preset
    system_prompt = req.system_prompt or preset_system
    output_instructions = req.output_instructions or preset_instructions

    if system_prompt:
        logger.info(f"Using system_prompt ({len(system_prompt)} chars)")
    if output_instructions:
        logger.info(f"Using output_instructions ({len(output_instructions)} chars)")

    llm_config = _resolve_llm_config(req.llm)
    output_dir = file_path.parent
    stem = file_path.stem
    t0 = time.time()
    strategies: List[StrategySummary] = []

    # Run TreeKG
    treekg_result = await _run_strategy(
        name="treekg",
        runner=_run_treekg(chunks, llm_config, system_prompt, output_instructions),
        output_path=output_dir / f"{stem}_summary_treekg.md",
    )
    strategies.append(treekg_result)

    # Run RAPTOR
    raptor_result = await _run_strategy(
        name="raptor",
        runner=_run_raptor(chunks, llm_config, req.embedding, system_prompt, output_instructions),
        output_path=output_dir / f"{stem}_summary_raptor.md",
    )
    strategies.append(raptor_result)

    total = time.time() - t0

    return SummarizeResponse(
        input_file=str(file_path),
        output_dir=str(output_dir),
        strategies=strategies,
        total_seconds=round(total, 2),
    )


async def _run_strategy(
    name: str,
    runner,
    output_path: Path,
) -> StrategySummary:
    """Run a single strategy, catch errors, write output."""
    t0 = time.time()
    try:
        result: SummarizationResult = await runner
        _write_summary(result, output_path)
        elapsed = time.time() - t0
        return StrategySummary(
            strategy=name,
            success=True,
            output_file=str(output_path),
            document_summary=result.document_summary,
            num_nodes=result.node_count,
            num_layers=result.num_layers,
            processing_seconds=round(elapsed, 2),
        )
    except Exception as e:
        logger.error(f"Strategy {name} failed: {e}")
        return StrategySummary(
            strategy=name,
            success=False,
            error=str(e),
            processing_seconds=round(time.time() - t0, 2),
        )
