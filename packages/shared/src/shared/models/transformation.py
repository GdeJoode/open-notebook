"""
Transformation models.

Pure Pydantic models for text transformations and default prompts.
"""

from typing import ClassVar, Optional

from pydantic import Field

from shared.models.base import ObjectModel, RecordModel


class Transformation(ObjectModel):
    """
    Text transformation configuration.

    Defines a reusable prompt-based transformation that can be applied to source content.
    """

    table_name: ClassVar[str] = "transformation"

    name: str = Field(..., description="Transformation name identifier")
    title: str = Field(..., description="Display title for the transformation")
    description: str = Field(..., description="Description of what this transformation does")
    prompt: str = Field(..., description="The transformation prompt template")
    apply_default: bool = Field(
        False, description="Whether to apply this transformation by default"
    )


class DefaultPrompts(RecordModel):
    """
    Default prompt configuration singleton.

    Stores system-wide default prompts for various operations.
    """

    record_id: ClassVar[str] = "open_notebook:default_prompts"

    transformation_instructions: Optional[str] = Field(
        None, description="Instructions for executing a transformation"
    )
