from typing import Optional, List, Dict, Any

from .base_asset import BaseAsset, BaseAssetCreate


class ObjaverseAsset(BaseAsset):
    """Objaverse 3D model asset with categorization"""

    viewer_url: str
    license: Optional[str] = None
    asset_category: Optional[str] = None  # "furniture", "vehicle", "character", etc.
    geometric_complexity: Optional[str] = None  # "simple", "moderate", "complex"
    has_textures: bool = False
    file_format: str = "glb"

    def get_source(self) -> str:
        return "objaverse"

    def categorize(self) -> Dict[str, Any]:
        """Categorize Objaverse asset based on tags"""
        tags_str = " ".join(self.tags or []).lower()

        # Simple categorization - default to furniture for interior design
        category = "furniture"
        if any(kw in tags_str for kw in ["car", "vehicle", "transport"]):
            category = "vehicle"
        elif any(kw in tags_str for kw in ["person", "character", "human"]):
            category = "character"

        # Simple complexity
        complexity = "moderate"
        if "simple" in tags_str or "basic" in tags_str:
            complexity = "simple"
        elif "complex" in tags_str or "detailed" in tags_str:
            complexity = "complex"

        # Check for textures
        has_textures = "texture" in tags_str or "material" in tags_str

        return {
            "asset_category": category,
            "geometric_complexity": complexity,
            "has_textures": has_textures,
            "file_format": "glb",
        }


class ObjaverseAssetCreate(ObjaverseAsset, BaseAssetCreate):
    """Objaverse asset for creation with auto-categorization"""

    def __init__(self, **data):
        super().__init__(**data)
        # Auto-categorize on creation
        categorization = self.categorize()
        for key, value in categorization.items():
            setattr(self, key, value)

    def to_db_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for database insertion"""
        return {
            "uid": self.uid,
            "viewer_url": self.url,  # Objaverse uses viewer_url
            "license": self.license,
            "tags": self.tags,
            "asset_category": self.asset_category,
            "geometric_complexity": self.geometric_complexity,
            "has_textures": self.has_textures,
            "file_format": self.file_format,
            "clip_embedding": self.clip_embedding,
            "sbert_embedding": self.sbert_embedding,
        }
