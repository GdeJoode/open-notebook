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
        """The app-injected language model, or ``None``.

        In the fork this is a model behind the model gate (F2-08, ADR 0006):
        every call goes via ``ModelGate`` to llama-swap.
        """
        return getattr(self.config, "language_model", None)

    def _poortmodel(self):
        """The injected model; without one there is no model call (F2-08).

        The strategies no longer build a model themselves: that would bypass the
        model gate (no register, no classification check, no audit row).
        """
        model = self.injected_model
        if model is None:
            raise RuntimeError(
                "Geen taalmodel: de aanroeper geeft een model van de modelpoort mee "
                "(SummarizationConfig.language_model)"
            )
        return model

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
