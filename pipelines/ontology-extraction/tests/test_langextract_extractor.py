"""Tests for LangExtract extractor and ExampleBuilder.

These tests mock langextract to avoid requiring a running model server.
We force-reimport the extractor modules inside each test so they pick up
the fake langextract from the patched sys.modules.
"""

import importlib
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from ontology_manager.schema import (
    DataType,
    EntityTypeDefinition,
    Ontology,
    OntologyMetadata,
    PropertyDefinition,
)
from shared.models.extraction import ExtractionResult


# ---------------------------------------------------------------------------
# Helpers: fake langextract data classes for mocking
# ---------------------------------------------------------------------------

_FAKE_LX = types.ModuleType("langextract")
_FAKE_DATA = types.ModuleType("langextract.data")
_FAKE_IO = types.ModuleType("langextract.io")


class FakeExtraction:
    def __init__(
        self,
        extraction_class: str = "",
        extraction_text: str = "",
        attributes: dict[str, Any] | None = None,
        char_interval: Any = None,
    ):
        self.extraction_class = extraction_class
        self.extraction_text = extraction_text
        self.attributes = attributes
        self.char_interval = char_interval


class FakeExampleData:
    def __init__(self, text: str = "", extractions: list[Any] | None = None):
        self.text = text
        self.extractions = extractions or []


class FakeCharInterval:
    def __init__(self, start_pos: int, end_pos: int):
        self.start_pos = start_pos
        self.end_pos = end_pos


class FakeAnnotatedDocument:
    def __init__(
        self,
        extractions: list[Any] | None = None,
        document_id: str | None = None,
    ):
        self.extractions = extractions or []
        self.document_id = document_id


class FakeDocument:
    def __init__(
        self,
        text: str = "",
        document_id: str | None = None,
        additional_context: str | None = None,
    ):
        self.text = text
        self.document_id = document_id
        self.additional_context = additional_context


_FAKE_DATA.Extraction = FakeExtraction  # type: ignore[attr-defined]
_FAKE_DATA.ExampleData = FakeExampleData  # type: ignore[attr-defined]
_FAKE_DATA.CharInterval = FakeCharInterval  # type: ignore[attr-defined]
_FAKE_DATA.AnnotatedDocument = FakeAnnotatedDocument  # type: ignore[attr-defined]
_FAKE_DATA.Document = FakeDocument  # type: ignore[attr-defined]
_FAKE_LX.data = _FAKE_DATA  # type: ignore[attr-defined]
_FAKE_IO.save_annotated_documents = MagicMock()  # type: ignore[attr-defined]
_FAKE_LX.io = _FAKE_IO  # type: ignore[attr-defined]
_FAKE_LX.visualize = MagicMock()  # type: ignore[attr-defined]

# Module keys we need to evict so they reimport with patched langextract
_MODULES_TO_EVICT = [
    "ontology_extraction.extractors.example_builder",
    "ontology_extraction.extractors.langextract_extractor",
]


def _patch_and_import():
    """Inject fake langextract into sys.modules, evict cached extractor modules,
    and reimport them so they see ``lx = langextract`` instead of ``lx = None``.

    Returns (ExampleBuilder, LangExtractExtractor) classes.
    """
    sys.modules["langextract"] = _FAKE_LX
    sys.modules["langextract.data"] = _FAKE_DATA
    sys.modules["langextract.io"] = _FAKE_IO
    for mod_key in _MODULES_TO_EVICT:
        sys.modules.pop(mod_key, None)

    eb_mod = importlib.import_module(
        "ontology_extraction.extractors.example_builder"
    )
    le_mod = importlib.import_module(
        "ontology_extraction.extractors.langextract_extractor"
    )
    return eb_mod.ExampleBuilder, le_mod.LangExtractExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_ontology() -> Ontology:
    """A minimal ontology with two entity types that have examples."""
    return Ontology(
        metadata=OntologyMetadata(name="test", version="1.0"),
        entity_types={
            "Person": EntityTypeDefinition(
                name="Person",
                description="a person",
                examples=["Alice", "Bob"],
                extraction_hints=["Look for proper names of people."],
                properties=[
                    PropertyDefinition(
                        name="role",
                        data_type=DataType.STRING,
                        examples=["researcher"],
                    ),
                ],
            ),
            "Organization": EntityTypeDefinition(
                name="Organization",
                description="an organization",
                examples=["MIT", "Google"],
            ),
        },
    )


@pytest.fixture()
def empty_ontology() -> Ontology:
    """An ontology with entity types that lack examples."""
    return Ontology(
        metadata=OntologyMetadata(name="empty", version="1.0"),
        entity_types={
            "Thing": EntityTypeDefinition(name="Thing", description="a thing"),
        },
    )


@pytest.fixture()
def yaml_override_file(tmp_path: Path) -> Path:
    """Create a temporary YAML override file."""
    data = {
        "prompt_description": "Custom prompt from YAML.",
        "examples": [
            {
                "text": "Dr. Smith works at MIT.",
                "extractions": [
                    {
                        "extraction_class": "Person",
                        "extraction_text": "Dr. Smith",
                        "attributes": {"role": "doctor"},
                    },
                    {
                        "extraction_class": "Organization",
                        "extraction_text": "MIT",
                    },
                ],
            },
        ],
    }
    path = tmp_path / "test_langextract_examples.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# ExampleBuilder Tests
# ---------------------------------------------------------------------------


class TestExampleBuilder:
    """Tests for ExampleBuilder generating ExampleData from ontology + YAML."""

    def test_examples_from_ontology_with_entity_examples(
        self, simple_ontology: Ontology
    ):
        """Entity types with examples produce ExampleData objects."""
        ExampleBuilder, _ = _patch_and_import()
        builder = ExampleBuilder()
        prompt, examples = builder.build_examples(simple_ontology)

        assert "Person" in prompt
        assert "Organization" in prompt
        assert len(examples) >= 2

        person_example = next(
            (
                e
                for e in examples
                if any(ext.extraction_class == "Person" for ext in e.extractions)
            ),
            None,
        )
        assert person_example is not None
        assert "Alice" in person_example.text

    def test_examples_from_ontology_without_examples_returns_empty(
        self, empty_ontology: Ontology
    ):
        """Entity types without examples produce no ExampleData."""
        ExampleBuilder, _ = _patch_and_import()
        builder = ExampleBuilder()
        prompt, examples = builder.build_examples(empty_ontology)

        assert isinstance(prompt, str)
        assert len(examples) == 0

    def test_yaml_override_loading(
        self, simple_ontology: Ontology, yaml_override_file: Path
    ):
        """YAML overrides are loaded and merged with ontology examples."""
        ExampleBuilder, _ = _patch_and_import()
        builder = ExampleBuilder()
        prompt, examples = builder.build_examples(
            simple_ontology, override_file=yaml_override_file
        )

        assert prompt == "Custom prompt from YAML."
        assert len(examples) >= 3  # 1 from YAML + 2 from ontology

    def test_yaml_auto_discovery(self, simple_ontology: Ontology, tmp_path: Path):
        """ExampleBuilder auto-discovers YAML files by ontology name."""
        data = {
            "prompt_description": "Auto-discovered.",
            "examples": [
                {
                    "text": "Sample text.",
                    "extractions": [
                        {
                            "extraction_class": "Person",
                            "extraction_text": "Sample",
                        }
                    ],
                }
            ],
        }
        yaml_path = tmp_path / "test_langextract_examples.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)

        ExampleBuilder, _ = _patch_and_import()
        builder = ExampleBuilder(examples_dir=tmp_path)
        prompt, examples = builder.build_examples(simple_ontology)

        assert prompt == "Auto-discovered."

    def test_invalid_yaml_file_handled_gracefully(
        self, simple_ontology: Ontology, tmp_path: Path
    ):
        """A malformed YAML file doesn't crash; falls back to ontology examples."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("::not valid yaml[[[")

        ExampleBuilder, _ = _patch_and_import()
        builder = ExampleBuilder()
        prompt, examples = builder.build_examples(
            simple_ontology, override_file=bad_yaml
        )

        assert isinstance(prompt, str)
        assert len(examples) >= 2


# ---------------------------------------------------------------------------
# LangExtractExtractor Tests
# ---------------------------------------------------------------------------


def _make_extractor(confidence_threshold: float = 0.5, **kwargs):
    """Create a LangExtractExtractor with defaults, optionally overriding fields."""
    _, LangExtractExtractor = _patch_and_import()
    extractor = LangExtractExtractor.__new__(LangExtractExtractor)
    extractor._confidence_threshold = confidence_threshold
    extractor._model_id = kwargs.get("model_id", "test")
    extractor._model_url = kwargs.get("model_url", None)
    extractor._extraction_passes = kwargs.get("extraction_passes", 1)
    extractor._max_workers = kwargs.get("max_workers", 4)
    extractor._max_char_buffer = kwargs.get("max_char_buffer", 5000)
    extractor._example_builder = kwargs.get("example_builder", None)
    # New parameters with defaults
    extractor._batch_length = kwargs.get("batch_length", None)
    extractor._temperature = kwargs.get("temperature", None)
    extractor._max_output_tokens = kwargs.get("max_output_tokens", None)
    extractor._top_p = kwargs.get("top_p", None)
    extractor._top_k = kwargs.get("top_k", None)
    extractor._use_schema_constraints = kwargs.get("use_schema_constraints", True)
    extractor._fence_output = kwargs.get("fence_output", True)
    extractor._api_key = kwargs.get("api_key", None)
    extractor._provider = kwargs.get("provider", None)
    extractor._provider_kwargs = kwargs.get("provider_kwargs", None)
    extractor._language_model_params = kwargs.get("language_model_params", None)
    extractor._save_jsonl = kwargs.get("save_jsonl", False)
    extractor._jsonl_output_dir = kwargs.get("jsonl_output_dir", None)
    extractor._visualize = kwargs.get("visualize", False)
    extractor._visualize_output_dir = kwargs.get("visualize_output_dir", None)
    return extractor


class TestLangExtractExtractorResultMapping:
    """Tests for LangExtractExtractor._map_result and _map_single_document."""

    def test_map_result_with_grounding(self):
        """Extractions with char_interval produce source_grounding in entities."""
        extractor = _make_extractor()
        doc = FakeAnnotatedDocument(
            extractions=[
                FakeExtraction(
                    extraction_class="Person",
                    extraction_text="Dr. Smith",
                    attributes={"role": "researcher"},
                    char_interval=FakeCharInterval(0, 9),
                ),
                FakeExtraction(
                    extraction_class="Organization",
                    extraction_text="MIT",
                    char_interval=FakeCharInterval(24, 27),
                ),
            ]
        )

        result = extractor._map_result(doc)

        assert isinstance(result, ExtractionResult)
        assert len(result.entities) == 2
        assert result.entities[0].text == "Dr. Smith"
        assert result.entities[0].label == "Person"
        assert result.entities[0].properties == {"role": "researcher"}
        assert result.entities[0].confidence == 1.0
        assert result.entities[0].source_grounding == {
            "start_pos": 0,
            "end_pos": 9,
        }
        assert result.entities[1].text == "MIT"
        assert result.entities[1].source_grounding == {
            "start_pos": 24,
            "end_pos": 27,
        }
        assert len(result.relations) == 0
        assert result.metadata["extractor"] == "langextract"

    def test_map_result_without_grounding(self):
        """Extractions without char_interval get lower confidence and no grounding."""
        extractor = _make_extractor()
        doc = FakeAnnotatedDocument(
            extractions=[
                FakeExtraction(
                    extraction_class="Topic",
                    extraction_text="AI research",
                    char_interval=None,
                ),
            ]
        )

        result = extractor._map_result(doc)

        assert len(result.entities) == 1
        assert result.entities[0].confidence == 0.6
        assert result.entities[0].source_grounding is None

    def test_confidence_filtering(self):
        """Extractions below confidence threshold are excluded."""
        extractor = _make_extractor(confidence_threshold=0.8)
        doc = FakeAnnotatedDocument(
            extractions=[
                FakeExtraction(
                    extraction_class="Person",
                    extraction_text="Alice",
                    char_interval=FakeCharInterval(0, 5),
                ),
                FakeExtraction(
                    extraction_class="Person",
                    extraction_text="Bob",
                    char_interval=None,
                ),
            ]
        )

        result = extractor._map_result(doc)

        assert len(result.entities) == 1
        assert result.entities[0].text == "Alice"

    def test_map_result_empty_extractions(self):
        """Empty extraction list produces empty result."""
        extractor = _make_extractor()
        doc = FakeAnnotatedDocument(extractions=[])
        result = extractor._map_result(doc)

        assert len(result.entities) == 0
        assert len(result.relations) == 0

    def test_map_result_with_document_id(self):
        """AnnotatedDocument with document_id includes it in metadata."""
        extractor = _make_extractor()
        doc = FakeAnnotatedDocument(
            extractions=[
                FakeExtraction(
                    extraction_class="Person",
                    extraction_text="Alice",
                    char_interval=FakeCharInterval(0, 5),
                ),
            ],
            document_id="chunk_42",
        )

        result = extractor._map_result(doc)

        assert result.metadata["document_id"] == "chunk_42"

    def test_map_result_list_of_documents(self):
        """_map_result handles a list of AnnotatedDocuments."""
        extractor = _make_extractor()
        docs = [
            FakeAnnotatedDocument(
                extractions=[
                    FakeExtraction(
                        extraction_class="Person",
                        extraction_text="Alice",
                        char_interval=FakeCharInterval(0, 5),
                    ),
                ],
                document_id="doc_1",
            ),
            FakeAnnotatedDocument(
                extractions=[
                    FakeExtraction(
                        extraction_class="Organization",
                        extraction_text="MIT",
                        char_interval=FakeCharInterval(0, 3),
                    ),
                ],
                document_id="doc_2",
            ),
        ]

        result = extractor._map_result(docs)

        assert len(result.entities) == 2
        assert result.metadata["document_count"] == 2
        assert result.entities[0].text == "Alice"
        assert result.entities[1].text == "MIT"


class TestLangExtractExtractorExtract:
    """Tests for the async extract() method with mocked lx.extract()."""

    @pytest.mark.asyncio
    async def test_extract_calls_lx_and_maps_result(
        self, simple_ontology: Ontology
    ):
        """extract() builds examples, calls lx.extract, and maps the result."""
        fake_doc = FakeAnnotatedDocument(
            extractions=[
                FakeExtraction(
                    extraction_class="Person",
                    extraction_text="Alice",
                    attributes={"role": "researcher"},
                    char_interval=FakeCharInterval(0, 5),
                ),
            ]
        )

        _FAKE_LX.extract = MagicMock(return_value=fake_doc)  # type: ignore[attr-defined]
        ExampleBuilder, LangExtractExtractor = _patch_and_import()

        extractor = _make_extractor(
            model_id="qwen2.5:latest",
            model_url="http://localhost:11434",
        )
        extractor._example_builder = ExampleBuilder()

        result = await extractor.extract(
            "Alice is a researcher.", simple_ontology
        )

        assert len(result.entities) == 1
        assert result.entities[0].text == "Alice"
        assert result.entities[0].label == "Person"
        _FAKE_LX.extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_with_no_examples_returns_warning(
        self, empty_ontology: Ontology
    ):
        """extract() with no examples returns a warning result."""
        ExampleBuilder, LangExtractExtractor = _patch_and_import()

        extractor = _make_extractor()
        extractor._example_builder = ExampleBuilder()

        result = await extractor.extract("Some text.", empty_ontology)

        assert len(result.entities) == 0
        assert "warning" in result.metadata

    @pytest.mark.asyncio
    async def test_extract_with_schema_constraints_disabled(
        self, simple_ontology: Ontology
    ):
        """use_schema_constraints=False is forwarded to lx.extract."""
        fake_doc = FakeAnnotatedDocument(extractions=[])
        _FAKE_LX.extract = MagicMock(return_value=fake_doc)
        ExampleBuilder, _ = _patch_and_import()

        extractor = _make_extractor(use_schema_constraints=False)
        extractor._example_builder = ExampleBuilder()

        await extractor.extract("Some text.", simple_ontology)

        call_kwargs = _FAKE_LX.extract.call_args
        assert call_kwargs[1]["use_schema_constraints"] is False

    @pytest.mark.asyncio
    async def test_extract_with_fence_output_disabled(
        self, simple_ontology: Ontology
    ):
        """fence_output=False is forwarded to lx.extract."""
        fake_doc = FakeAnnotatedDocument(extractions=[])
        _FAKE_LX.extract = MagicMock(return_value=fake_doc)
        ExampleBuilder, _ = _patch_and_import()

        extractor = _make_extractor(fence_output=False)
        extractor._example_builder = ExampleBuilder()

        await extractor.extract("Some text.", simple_ontology)

        call_kwargs = _FAKE_LX.extract.call_args
        assert call_kwargs[1]["fence_output"] is False

    @pytest.mark.asyncio
    async def test_extract_forwards_provider_kwargs(
        self, simple_ontology: Ontology
    ):
        """temperature, api_key, top_p, top_k are forwarded to lx.extract."""
        fake_doc = FakeAnnotatedDocument(extractions=[])
        _FAKE_LX.extract = MagicMock(return_value=fake_doc)
        ExampleBuilder, _ = _patch_and_import()

        extractor = _make_extractor(
            temperature=0.1,
            api_key="sk-test-key",
            top_p=0.9,
            top_k=40,
            max_output_tokens=2048,
        )
        extractor._example_builder = ExampleBuilder()

        await extractor.extract("Some text.", simple_ontology)

        call_kwargs = _FAKE_LX.extract.call_args[1]
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["api_key"] == "sk-test-key"
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["top_k"] == 40
        assert call_kwargs["max_output_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_extract_forwards_batch_length(
        self, simple_ontology: Ontology
    ):
        """batch_length is forwarded to lx.extract when set."""
        fake_doc = FakeAnnotatedDocument(extractions=[])
        _FAKE_LX.extract = MagicMock(return_value=fake_doc)
        ExampleBuilder, _ = _patch_and_import()

        extractor = _make_extractor(batch_length=8)
        extractor._example_builder = ExampleBuilder()

        await extractor.extract("Some text.", simple_ontology)

        call_kwargs = _FAKE_LX.extract.call_args[1]
        assert call_kwargs["batch_length"] == 8

    @pytest.mark.asyncio
    async def test_extract_batch_length_not_sent_when_none(
        self, simple_ontology: Ontology
    ):
        """batch_length is not included in kwargs when None."""
        fake_doc = FakeAnnotatedDocument(extractions=[])
        _FAKE_LX.extract = MagicMock(return_value=fake_doc)
        ExampleBuilder, _ = _patch_and_import()

        extractor = _make_extractor(batch_length=None)
        extractor._example_builder = ExampleBuilder()

        await extractor.extract("Some text.", simple_ontology)

        call_kwargs = _FAKE_LX.extract.call_args[1]
        assert "batch_length" not in call_kwargs

    @pytest.mark.asyncio
    async def test_extract_forwards_provider(
        self, simple_ontology: Ontology
    ):
        """provider is forwarded to lx.extract when set."""
        fake_doc = FakeAnnotatedDocument(extractions=[])
        _FAKE_LX.extract = MagicMock(return_value=fake_doc)
        ExampleBuilder, _ = _patch_and_import()

        extractor = _make_extractor(provider="openai")
        extractor._example_builder = ExampleBuilder()

        await extractor.extract("Some text.", simple_ontology)

        call_kwargs = _FAKE_LX.extract.call_args[1]
        assert call_kwargs["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_extract_constructs_document_with_chunk_id(
        self, simple_ontology: Ontology
    ):
        """When chunk_id is passed, lx.data.Document is used instead of plain text."""
        fake_doc = FakeAnnotatedDocument(extractions=[])
        _FAKE_LX.extract = MagicMock(return_value=fake_doc)
        ExampleBuilder, _ = _patch_and_import()

        extractor = _make_extractor()
        extractor._example_builder = ExampleBuilder()

        await extractor.extract(
            "Alice works at MIT.", simple_ontology,
            chunk_id="chunk_42",
        )

        call_kwargs = _FAKE_LX.extract.call_args[1]
        doc_input = call_kwargs["text_or_documents"]
        assert isinstance(doc_input, FakeDocument)
        assert doc_input.text == "Alice works at MIT."
        assert doc_input.document_id == "chunk_42"

    @pytest.mark.asyncio
    async def test_extract_constructs_document_with_additional_context(
        self, simple_ontology: Ontology
    ):
        """additional_context is passed to Document when provided."""
        fake_doc = FakeAnnotatedDocument(extractions=[])
        _FAKE_LX.extract = MagicMock(return_value=fake_doc)
        ExampleBuilder, _ = _patch_and_import()

        extractor = _make_extractor()
        extractor._example_builder = ExampleBuilder()

        await extractor.extract(
            "Alice works at MIT.", simple_ontology,
            chunk_id="chunk_42",
            additional_context="From a research paper about AI.",
        )

        call_kwargs = _FAKE_LX.extract.call_args[1]
        doc_input = call_kwargs["text_or_documents"]
        assert isinstance(doc_input, FakeDocument)
        assert doc_input.additional_context == "From a research paper about AI."

    @pytest.mark.asyncio
    async def test_extract_plain_string_without_chunk_id(
        self, simple_ontology: Ontology
    ):
        """Without chunk_id, plain string is passed (backward compat)."""
        fake_doc = FakeAnnotatedDocument(extractions=[])
        _FAKE_LX.extract = MagicMock(return_value=fake_doc)
        ExampleBuilder, _ = _patch_and_import()

        extractor = _make_extractor()
        extractor._example_builder = ExampleBuilder()

        await extractor.extract("Plain text.", simple_ontology)

        call_kwargs = _FAKE_LX.extract.call_args[1]
        assert call_kwargs["text_or_documents"] == "Plain text."

    @pytest.mark.asyncio
    async def test_extract_calls_save_jsonl(
        self, simple_ontology: Ontology
    ):
        """lx.io.save_annotated_documents is called when save_jsonl=True."""
        fake_doc = FakeAnnotatedDocument(extractions=[])
        _FAKE_LX.extract = MagicMock(return_value=fake_doc)
        _FAKE_IO.save_annotated_documents = MagicMock()
        ExampleBuilder, _ = _patch_and_import()

        extractor = _make_extractor(save_jsonl=True, jsonl_output_dir="/tmp/jsonl")
        extractor._example_builder = ExampleBuilder()

        await extractor.extract("Some text.", simple_ontology)

        _FAKE_IO.save_annotated_documents.assert_called_once()
        call_args = _FAKE_IO.save_annotated_documents.call_args
        assert call_args[0][0] == [fake_doc]  # wrapped in list
        assert call_args[1]["output_dir"] == "/tmp/jsonl"

    @pytest.mark.asyncio
    async def test_extract_calls_visualize(
        self, simple_ontology: Ontology
    ):
        """lx.visualize is called when visualize=True."""
        fake_doc = FakeAnnotatedDocument(extractions=[])
        _FAKE_LX.extract = MagicMock(return_value=fake_doc)
        _FAKE_LX.visualize = MagicMock()
        ExampleBuilder, _ = _patch_and_import()

        extractor = _make_extractor(visualize=True, visualize_output_dir="/tmp/viz")
        extractor._example_builder = ExampleBuilder()

        await extractor.extract("Some text.", simple_ontology)

        _FAKE_LX.visualize.assert_called_once()
        call_args = _FAKE_LX.visualize.call_args
        assert call_args[0][0] == [fake_doc]
        assert call_args[1]["output_dir"] == "/tmp/viz"

    @pytest.mark.asyncio
    async def test_extract_jsonl_failure_does_not_crash(
        self, simple_ontology: Ontology
    ):
        """JSONL save failure is logged but doesn't crash extraction."""
        fake_doc = FakeAnnotatedDocument(extractions=[])
        _FAKE_LX.extract = MagicMock(return_value=fake_doc)
        _FAKE_IO.save_annotated_documents = MagicMock(
            side_effect=RuntimeError("disk full")
        )
        ExampleBuilder, _ = _patch_and_import()

        extractor = _make_extractor(save_jsonl=True)
        extractor._example_builder = ExampleBuilder()

        result = await extractor.extract("Some text.", simple_ontology)

        # Should succeed despite JSONL failure
        assert isinstance(result, ExtractionResult)

    @pytest.mark.asyncio
    async def test_extract_visualize_failure_does_not_crash(
        self, simple_ontology: Ontology
    ):
        """Visualization failure is logged but doesn't crash extraction."""
        fake_doc = FakeAnnotatedDocument(extractions=[])
        _FAKE_LX.extract = MagicMock(return_value=fake_doc)
        _FAKE_LX.visualize = MagicMock(side_effect=RuntimeError("no display"))
        ExampleBuilder, _ = _patch_and_import()

        extractor = _make_extractor(visualize=True)
        extractor._example_builder = ExampleBuilder()

        result = await extractor.extract("Some text.", simple_ontology)

        assert isinstance(result, ExtractionResult)

    @pytest.mark.asyncio
    async def test_extract_forwards_language_model_params(
        self, simple_ontology: Ontology
    ):
        """language_model_params is forwarded to lx.extract when set."""
        fake_doc = FakeAnnotatedDocument(extractions=[])
        _FAKE_LX.extract = MagicMock(return_value=fake_doc)
        ExampleBuilder, _ = _patch_and_import()

        params = {"batch_mode": True, "project_id": "my-project"}
        extractor = _make_extractor(language_model_params=params)
        extractor._example_builder = ExampleBuilder()

        await extractor.extract("Some text.", simple_ontology)

        call_kwargs = _FAKE_LX.extract.call_args[1]
        assert call_kwargs["language_model_params"] == params

    @pytest.mark.asyncio
    async def test_extract_forwards_custom_provider_kwargs(
        self, simple_ontology: Ontology
    ):
        """provider_kwargs dict is merged into lx.extract kwargs."""
        fake_doc = FakeAnnotatedDocument(extractions=[])
        _FAKE_LX.extract = MagicMock(return_value=fake_doc)
        ExampleBuilder, _ = _patch_and_import()

        extractor = _make_extractor(
            provider_kwargs={"custom_param": "value", "seed": 42}
        )
        extractor._example_builder = ExampleBuilder()

        await extractor.extract("Some text.", simple_ontology)

        call_kwargs = _FAKE_LX.extract.call_args[1]
        assert call_kwargs["custom_param"] == "value"
        assert call_kwargs["seed"] == 42


class TestImportGuard:
    """Test that the LLM extractor works when langextract is not installed."""

    def test_llm_extractor_works_without_langextract(self):
        """LLMExtractor functions regardless of langextract availability."""
        from ontology_extraction.extractors.llm_extractor import LLMExtractor

        extractor = LLMExtractor()
        assert extractor._llm_model == "default"
