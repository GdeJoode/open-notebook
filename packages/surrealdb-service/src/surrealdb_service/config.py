"""
Configuration for SurrealDB service.
"""

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class SurrealDBConfig(BaseSettings):
    """SurrealDB connection configuration."""

    model_config = {
        "env_prefix": "SURREAL_",
        "env_file": ".env",
        "extra": "ignore",
    }

    # Connection settings
    url: str = Field(
        default="ws://localhost:8000/rpc",
        description="SurrealDB WebSocket URL",
    )
    username: str = Field(default="root", description="Database username")
    password: str = Field(default="root", description="Database password")
    namespace: str = Field(default="open_notebook", description="Database namespace")
    database: str = Field(default="open_notebook", description="Database name")

    # Connection pool settings
    pool_size: int = Field(default=5, description="Connection pool size")
    connection_timeout: int = Field(default=30, description="Connection timeout in seconds")

    @classmethod
    def from_env(cls) -> "SurrealDBConfig":
        """Create config from environment variables with backward compatibility."""
        # Handle backward-compatible URL construction
        url = os.getenv("SURREAL_URL")
        if not url:
            address = os.getenv("SURREAL_ADDRESS", "localhost")
            port = os.getenv("SURREAL_PORT", "8000")
            url = f"ws://{address}:{port}/rpc"

        # Handle backward-compatible password
        password = os.getenv("SURREAL_PASSWORD") or os.getenv("SURREAL_PASS", "root")

        return cls(
            url=url,
            username=os.getenv("SURREAL_USER", "root"),
            password=password,
            namespace=os.getenv("SURREAL_NAMESPACE", "open_notebook"),
            database=os.getenv("SURREAL_DATABASE", "open_notebook"),
        )


# Global config instance
_config: Optional[SurrealDBConfig] = None


def get_config() -> SurrealDBConfig:
    """Get the global SurrealDB configuration."""
    global _config
    if _config is None:
        _config = SurrealDBConfig.from_env()
    return _config


def set_config(config: SurrealDBConfig) -> None:
    """Set the global SurrealDB configuration."""
    global _config
    _config = config
