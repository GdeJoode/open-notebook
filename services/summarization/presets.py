"""
Preset management and auto-detection for summarization prompts.

Loads preset YAML files and scores input text against each preset's
detection rules to automatically select the best-matching preset.
"""

import re
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

PRESETS_DIR = Path(__file__).parent / "presets"


def load_preset(name: str) -> dict:
    """Load a preset by name from the presets directory."""
    path = PRESETS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Preset '{name}' not found. Available: {list_presets()}"
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_presets() -> list[str]:
    """List available preset names."""
    return sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))


def detect_preset(text: str, has_speakers: bool = False) -> Optional[str]:
    """Auto-detect the best preset for the given input text.

    Scores each preset against the text using keyword matches and
    regex pattern matches. Returns the preset name with the highest
    score, or None if no preset scores above the minimum threshold.

    Args:
        text: The input text (first ~5000 chars are used for detection).
        has_speakers: Whether the input contains speaker diarization
            (boosts meeting_notes score).
    """
    # Use first ~5000 chars for detection (enough for classification)
    sample = text[:5000].lower()

    best_name: Optional[str] = None
    best_score = 0.0
    min_threshold = 3.0  # Minimum score to consider a match

    for preset_path in PRESETS_DIR.glob("*.yaml"):
        with open(preset_path, encoding="utf-8") as f:
            preset = yaml.safe_load(f)

        detect = preset.get("detect", {})
        score = 0.0

        # Keyword scoring: +1 per keyword found
        for kw in detect.get("keywords", []):
            if kw.lower() in sample:
                score += 1.0

        # Pattern scoring: +2 per regex match (patterns are more specific)
        for pattern in detect.get("patterns", []):
            try:
                if re.search(pattern, sample, re.IGNORECASE):
                    score += 2.0
            except re.error:
                continue

        # Transcription boost: if input has speakers and preset wants it
        if has_speakers and detect.get("transcription_boost", False):
            score += 5.0

        preset_name = preset_path.stem
        if score > best_score:
            best_score = score
            best_name = preset_name

    if best_score >= min_threshold and best_name:
        logger.info(
            f"Auto-detected preset: {best_name} (score: {best_score:.1f})"
        )
        return best_name

    logger.info(f"No preset matched above threshold (best: {best_name}={best_score:.1f})")
    return None


def get_preset_prompts(
    preset_name: Optional[str] = None,
    text: str = "",
    has_speakers: bool = False,
) -> tuple[str, str]:
    """Get system_prompt and output_instructions from a preset.

    If preset_name is "auto", auto-detects from text.
    If preset_name is None or empty, returns empty strings (use defaults).

    Returns:
        Tuple of (system_prompt, output_instructions).
    """
    if not preset_name:
        return "", ""

    if preset_name == "auto":
        detected = detect_preset(text, has_speakers=has_speakers)
        if not detected:
            return "", ""
        preset_name = detected

    try:
        preset = load_preset(preset_name)
    except FileNotFoundError as e:
        logger.warning(str(e))
        return "", ""

    system_prompt = preset.get("system_prompt", "").strip()
    output_instructions = preset.get("output_instructions", "").strip()

    logger.info(f"Using preset '{preset_name}': system_prompt={len(system_prompt)} chars, "
                f"output_instructions={len(output_instructions)} chars")

    return system_prompt, output_instructions
