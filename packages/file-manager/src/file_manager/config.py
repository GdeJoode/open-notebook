"""
Configuration for file manager service.
"""

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class FileManagerConfig(BaseSettings):
    """File manager configuration."""

    model_config = {
        "env_prefix": "FILE_MANAGER_",
        "env_file": ".env",
        "extra": "ignore",
    }

    # Directory paths
    upload_dir: str = Field(
        default="./data/uploads",
        description="Directory for uploaded files",
    )
    input_dir: str = Field(
        default="./data/input",
        description="Directory for organized input files",
    )
    output_dir: str = Field(
        default="./data/output",
        description="Directory for processed output files",
    )
    markdown_dir: str = Field(
        default="./data/markdown",
        description="Directory for markdown exports with assets",
    )
    temp_dir: str = Field(
        default="./data/temp",
        description="Directory for temporary files",
    )

    # File operation settings
    default_file_operation: Literal["copy", "move", "none"] = Field(
        default="copy",
        description="Default file operation after processing",
    )
    default_naming_scheme: Literal[
        "timestamp_prefix", "date_prefix", "datetime_suffix", "original"
    ] = Field(
        default="date_prefix",
        description="Default naming scheme for output files",
    )
    auto_delete_uploads: bool = Field(
        default=False,
        description="Automatically delete uploaded files after processing",
    )

    # Size limits
    max_file_size_mb: int = Field(
        default=100,
        description="Maximum file size in MB",
    )
    max_upload_size_mb: int = Field(
        default=500,
        description="Maximum total upload size in MB",
    )

    # Allowed file types
    allowed_extensions: str = Field(
        default=".pdf,.docx,.doc,.txt,.md,.html,.epub,.pptx,.xlsx",
        description="Comma-separated list of allowed file extensions",
    )

    @property
    def allowed_extensions_list(self) -> list[str]:
        """Get allowed extensions as a list."""
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def max_upload_size_bytes(self) -> int:
        """Get max upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create all configured directories if they don't exist."""
        for dir_path in [
            self.upload_dir,
            self.input_dir,
            self.output_dir,
            self.markdown_dir,
            self.temp_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Global config instance
_config: Optional[FileManagerConfig] = None


def get_config() -> FileManagerConfig:
    """Get the global file manager configuration."""
    global _config
    if _config is None:
        _config = FileManagerConfig()
    return _config


def set_config(config: FileManagerConfig) -> None:
    """Set the global file manager configuration."""
    global _config
    _config = config
