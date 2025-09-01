from typing import Optional, List, Dict, Any

from .base_asset import BaseAsset, BaseAssetCreate


class PolyhavenAsset(BaseAsset):
    """Poly Haven PBR material asset with categorization"""

    polyhaven_url: str
    asset_category: Optional[str] = None  # "wood", "metal", "fabric", "stone", etc.
    asset_subcategory: Optional[str] = None  # "oak", "steel", "leather", "marble", etc.
    surface_type: Optional[str] = None  # "rough", "smooth", "bumpy", "glossy"
    material_properties: Optional[
        List[str]
    ] = []  # ["reflective", "transparent", "rough"]
    resolution_available: Optional[List[str]] = []  # ["1k", "2k", "4k", "8k"]
    categories: Optional[List[str]] = []  # From Poly Haven API
    asset_type: str = "texture"  # "texture", "hdri"

    def get_source(self) -> str:
        return "polyhaven"

    def categorize(self) -> Dict[str, Any]:
        """Categorize Poly Haven asset based on name and tags"""
        text = f"{self.uid} {' '.join(self.tags or [])}".lower()

        # Simple material categorization
        category = "material"  # Default
        if any(kw in text for kw in ["wood", "oak", "pine"]):
            category = "wood"
        elif any(kw in text for kw in ["metal", "steel", "iron"]):
            category = "metal"
        elif any(kw in text for kw in ["stone", "concrete", "marble", "brick"]):
            category = "stone"
        elif any(kw in text for kw in ["fabric", "leather", "cloth"]):
            category = "fabric"

        # Simple surface type
        surface_type = "smooth"
        if any(kw in text for kw in ["rough", "bumpy"]):
            surface_type = "rough"
        elif any(kw in text for kw in ["glossy", "shiny"]):
            surface_type = "glossy"

        return {
            "asset_category": category,
            "asset_subcategory": self.uid.split("_")[0]
            if "_" in self.uid
            else self.uid,
            "surface_type": surface_type,
            "material_properties": [],
            "resolution_available": ["1k", "2k", "4k", "8k"],
        }


class PolyhavenAssetCreate(PolyhavenAsset, BaseAssetCreate):
    """Poly Haven asset for creation with auto-categorization"""

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
            "polyhaven_url": self.url,  # Poly Haven uses polyhaven_url
            "asset_category": self.asset_category,
            "asset_subcategory": self.asset_subcategory,
            "surface_type": self.surface_type,
            "material_properties": self.material_properties,
            "resolution_available": self.resolution_available,
            "tags": self.tags,
            "categories": self.categories,
            "asset_type": self.asset_type,
            "clip_embedding": self.clip_embedding,
            "sbert_embedding": self.sbert_embedding,
        }
