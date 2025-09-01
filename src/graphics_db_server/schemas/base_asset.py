from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime

import numpy as np
from pydantic import BaseModel, ConfigDict


class BaseAsset(BaseModel, ABC):
    """Abstract base class for all asset types"""

    uid: str
    url: str
    tags: Optional[List[str]] = []
    clip_embedding: Optional[List[float] | np.ndarray] = None
    sbert_embedding: Optional[List[float] | np.ndarray] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

    @abstractmethod
    def get_source(self) -> str:
        """Return the source identifier for this asset type"""
        pass

    @abstractmethod
    def categorize(self) -> Dict[str, Any]:
        """Perform source-specific categorization of the asset"""
        pass


class BaseAssetCreate(BaseAsset):
    """Base class for asset creation with required embeddings"""

    clip_embedding: List[float] | np.ndarray
    sbert_embedding: List[float] | np.ndarray

    @abstractmethod
    def to_db_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for database insertion"""
        pass
