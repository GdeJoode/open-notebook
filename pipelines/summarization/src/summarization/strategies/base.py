"""Base class for all summarization strategies."""

import re
from abc import ABC, abstractmethod
from typing import List

from summarization.config import SummarizationConfig
from summarization.models.result import ChunkInput, SummarizationResult

# Regex to strip <think>...</think> blocks (thinking models like qwen3.5, deepseek-r1)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

# Default system message when no custom system_prompt is configured
DEFAULT_SYSTEM_PROMPT = (
    "You are a summarization assistant. "
    "Output ONLY the summary text. "
    "Do not include reasoning, thinking steps, meta-commentary, or preamble. "
    "Do not repeat the instructions. "
    "Be concise and preserve key ideas."
)


class BaseSummarizationStrategy(ABC):
    """Abstract base for summarization strategies.

    Subclasses implement ``summarize()`` which receives ordered chunks
    and returns a :class:`SummarizationResult`.
    """

    def __init__(self, config: SummarizationConfig, **kwargs) -> None:
        self.config = config

    @property
    def injected_model(self):
        """The app-injected routed LanguageModel, or ``None`` (Track J.4).

        Strategies call this from ``_get_model``: when non-``None`` they use it
        directly (privacy-aware routed model with failover already applied at
        injection time) instead of constructing one via ``AIFactory`` from the
        env-driven ``LLMConfig``. ``None`` -> the legacy env path (back-compat).
        """
        return getattr(self.config, "language_model", None)

    @abstractmethod
    async def summarize(self, chunks: List[ChunkInput]) -> SummarizationResult:
        """Run the strategy on the provided chunks."""
        ...

    def get_system_prompt(self) -> str:
        """Return the system prompt: custom if configured, otherwise default."""
        return self.config.system_prompt or DEFAULT_SYSTEM_PROMPT

    def apply_output_instructions(self, prompt: str) -> str:
        """Append output_instructions to a prompt if configured."""
        if self.config.output_instructions:
            return f"{prompt}\n\n{self.config.output_instructions}"
        return prompt

    @staticmethod
    def normalize_response(text: str) -> str:
        """Clean LLM response by stripping wrapper artifacts.

        Handles:
        - <think>...</think> blocks from thinking models (qwen3.5, deepseek-r1)
        - Leading/trailing whitespace
        """
        cleaned = _THINK_RE.sub("", text)
        return cleaned.strip()
