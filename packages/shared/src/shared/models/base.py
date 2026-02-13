"""
Base model classes for the shared package.

These are pure Pydantic models without any database operations.
Database operations are handled by the surrealdb-service package.
"""

from datetime import datetime
from typing import Any, ClassVar, Dict, Optional

from pydantic import BaseModel, ConfigDict, field_validator


class ObjectModel(BaseModel):
    """
    Base model for all domain objects.

    Contains only data fields and validation logic.
    Database operations (save, delete, get) are handled by the repository layer.
    """
    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
    )

    id: Optional[str] = None
    table_name: ClassVar[str] = ""
    created: Optional[datetime] = None
    updated: Optional[datetime] = None

    @field_validator("created", "updated", mode="before")
    @classmethod
    def parse_datetime(cls, value: Any) -> Optional[datetime]:
        """Parse datetime from string or return as-is."""
        if value is None:
            return None
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    def to_dict(self, exclude_none: bool = True) -> Dict[str, Any]:
        """Convert model to dictionary, optionally excluding None values."""
        data = self.model_dump()
        if exclude_none:
            return {k: v for k, v in data.items() if v is not None}
        return data


class RecordModel(BaseModel):
    """
    Base model for singleton record types (settings, configuration).

    Identified by a unique record_id rather than auto-generated id.
    Database persistence is handled by the repository layer.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        extra="allow",
        from_attributes=True,
    )

    record_id: ClassVar[str] = ""

    def to_dict(self, exclude_none: bool = True) -> Dict[str, Any]:
        """Convert model to dictionary, optionally excluding None values."""
        data = self.model_dump()
        if exclude_none:
            return {k: v for k, v in data.items() if v is not None}
        return data
