# Current legacy schema
from .asset import Asset, AssetCreate

# New OOP base classes
from .base_asset import BaseAsset, BaseAssetCreate

# Source-specific schemas
from .objaverse_asset import (
    ObjaverseAsset,
    ObjaverseAssetCreate,
)

from .polyhaven_asset import (
    PolyhavenAsset,
    PolyhavenAssetCreate,
)

# Convenience exports for clean imports
__all__ = [
    # Legacy (for backward compatibility)
    "Asset",
    "AssetCreate",
    # Base classes
    "BaseAsset",
    "BaseAssetCreate",
    # Objaverse
    "ObjaverseAsset",
    "ObjaverseAssetCreate",
    # Poly Haven
    "PolyhavenAsset",
    "PolyhavenAssetCreate",
]
