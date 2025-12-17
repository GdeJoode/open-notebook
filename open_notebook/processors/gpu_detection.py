"""
GPU Detection and Acceleration Setup for Document Processing.

Detects available GPU hardware (CUDA, MPS, CPU) and configures
the appropriate acceleration for spaCy and PyTorch models.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from loguru import logger


class GPUDevice(Enum):
    """Available GPU device types."""
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    CPU = "cpu"


@dataclass
class GPUConfig:
    """GPU configuration for document processing."""
    device: GPUDevice
    device_id: int = 0
    enabled: bool = True

    @property
    def torch_device(self) -> str:
        """Get PyTorch device string."""
        if self.device == GPUDevice.CUDA:
            return f"cuda:{self.device_id}"
        elif self.device == GPUDevice.MPS:
            return "mps"
        return "cpu"

    @property
    def spacy_gpu_id(self) -> int:
        """Get spaCy GPU ID (-1 for CPU)."""
        if self.device == GPUDevice.CPU or not self.enabled:
            return -1
        return self.device_id


def detect_gpu() -> GPUConfig:
    """
    Detect available GPU hardware and return optimal configuration.

    Priority: CUDA > MPS > CPU

    Returns:
        GPUConfig with detected device settings
    """
    # Try CUDA first
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"🚀 CUDA detected: {device_name} ({device_count} device(s))")
            return GPUConfig(device=GPUDevice.CUDA, device_id=0, enabled=True)
    except ImportError:
        logger.debug("PyTorch not available for CUDA detection")
    except Exception as e:
        logger.debug(f"CUDA detection failed: {e}")

    # Try MPS (Apple Silicon)
    try:
        import torch
        if torch.backends.mps.is_available():
            logger.info("🍎 MPS (Apple Silicon) detected")
            return GPUConfig(device=GPUDevice.MPS, device_id=0, enabled=True)
    except (ImportError, AttributeError):
        logger.debug("MPS not available")
    except Exception as e:
        logger.debug(f"MPS detection failed: {e}")

    # Fall back to CPU
    logger.info("💻 Using CPU for processing")
    return GPUConfig(device=GPUDevice.CPU, device_id=-1, enabled=False)


def setup_spacy_gpu(config: GPUConfig) -> bool:
    """
    Configure spaCy to use GPU if available.

    Args:
        config: GPU configuration

    Returns:
        True if GPU was successfully configured
    """
    if config.device == GPUDevice.CPU or not config.enabled:
        logger.debug("spaCy will use CPU")
        return False

    try:
        import spacy
        spacy.prefer_gpu(config.spacy_gpu_id)
        logger.info(f"✅ spaCy configured for GPU (device {config.spacy_gpu_id})")
        return True
    except Exception as e:
        logger.warning(f"Failed to configure spaCy GPU: {e}")
        return False


def get_optimal_config(
    user_preference: Optional[str] = None,
    gpu_enabled: bool = True
) -> GPUConfig:
    """
    Get optimal GPU configuration based on user preference and availability.

    Args:
        user_preference: User's device preference ('cuda', 'mps', 'cpu', 'auto')
        gpu_enabled: Whether GPU acceleration is enabled

    Returns:
        GPUConfig with optimal settings
    """
    if not gpu_enabled:
        logger.info("GPU disabled by user preference")
        return GPUConfig(device=GPUDevice.CPU, device_id=-1, enabled=False)

    if user_preference and user_preference.lower() != "auto":
        # User specified a device
        try:
            device = GPUDevice(user_preference.lower())
            if device == GPUDevice.CUDA:
                import torch
                if torch.cuda.is_available():
                    return GPUConfig(device=device, enabled=True)
                logger.warning("CUDA requested but not available, falling back to auto-detect")
            elif device == GPUDevice.MPS:
                import torch
                if torch.backends.mps.is_available():
                    return GPUConfig(device=device, enabled=True)
                logger.warning("MPS requested but not available, falling back to auto-detect")
            elif device == GPUDevice.CPU:
                return GPUConfig(device=GPUDevice.CPU, device_id=-1, enabled=False)
        except (ValueError, ImportError):
            logger.warning(f"Invalid device preference: {user_preference}, falling back to auto-detect")

    # Auto-detect
    return detect_gpu()


# Module-level cached config
_cached_config: Optional[GPUConfig] = None


def get_gpu_config(force_refresh: bool = False) -> GPUConfig:
    """
    Get cached GPU configuration (or detect if not cached).

    Args:
        force_refresh: Force re-detection of GPU

    Returns:
        Cached or newly detected GPUConfig
    """
    global _cached_config
    if _cached_config is None or force_refresh:
        _cached_config = detect_gpu()
    return _cached_config
