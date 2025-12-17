"""
RAPTOR Summarization Module

LLM-based cluster summarization using open-notebook's model infrastructure.
"""

from typing import List, Optional

from loguru import logger


DEFAULT_SUMMARIZE_PROMPT = """Write a concise summary of the following text passages, capturing the key themes, main arguments, and important details. Focus on synthesizing the information rather than listing points.

Text passages:
{context}

Summary:"""


async def summarize_texts(
    texts: List[str],
    max_tokens: int = 200,
    model_id: Optional[str] = None,
    prompt_template: Optional[str] = None,
) -> str:
    """
    Summarize a list of text passages into a single summary.

    Args:
        texts: List of text passages to summarize
        max_tokens: Maximum tokens for the summary
        model_id: Specific model to use (or app default)
        prompt_template: Custom prompt template (use {context} placeholder)

    Returns:
        Summary text
    """
    from open_notebook.domain.models import model_manager

    # Combine texts with separators
    context = "\n\n---\n\n".join(texts)

    # Use provided template or default
    template = prompt_template or DEFAULT_SUMMARIZE_PROMPT
    prompt = template.format(context=context)

    # Get LLM model
    try:
        if model_id:
            model = await model_manager.get_model(model_id)
        else:
            model = await model_manager.get_default_model()

        if not model:
            raise RuntimeError("No LLM model available for summarization")

        # Generate summary
        response = await model.ainvoke(prompt)

        # Handle different response types
        if hasattr(response, 'content'):
            summary = response.content
        elif isinstance(response, str):
            summary = response
        else:
            summary = str(response)

        return summary.strip()

    except Exception as e:
        logger.error(f"RAPTOR summarization failed: {e}")
        # Fallback: return truncated concatenation
        fallback = " [...] ".join(t[:100] for t in texts[:3])
        if len(fallback) > max_tokens * 4:
            fallback = fallback[:max_tokens * 4] + "..."
        return fallback


async def summarize_cluster(
    cluster_texts: List[str],
    max_tokens: int = 200,
    model_id: Optional[str] = None,
    max_input_tokens: int = 3500,
) -> str:
    """
    Summarize a cluster of texts, handling token limits.

    If the combined text exceeds max_input_tokens, it will be truncated.

    Args:
        cluster_texts: List of texts in the cluster
        max_tokens: Maximum tokens for output summary
        model_id: Model to use for summarization
        max_input_tokens: Maximum tokens for input context

    Returns:
        Summary text
    """
    import tiktoken

    # Get tokenizer
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Rough estimate: 4 chars per token
        tokenizer = None

    # Truncate if needed
    truncated_texts = []
    total_tokens = 0

    for text in cluster_texts:
        if tokenizer:
            text_tokens = len(tokenizer.encode(text))
        else:
            text_tokens = len(text) // 4

        if total_tokens + text_tokens > max_input_tokens:
            # Add partial text if possible
            remaining = max_input_tokens - total_tokens
            if remaining > 100:
                if tokenizer:
                    # Decode truncated tokens back to text
                    tokens = tokenizer.encode(text)[:remaining]
                    truncated_texts.append(tokenizer.decode(tokens) + "...")
                else:
                    truncated_texts.append(text[:remaining * 4] + "...")
            break

        truncated_texts.append(text)
        total_tokens += text_tokens

    if not truncated_texts:
        truncated_texts = [cluster_texts[0][:max_input_tokens * 4]]

    return await summarize_texts(truncated_texts, max_tokens, model_id)
