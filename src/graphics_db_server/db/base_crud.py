"""
Abstract base class for CRUD operations with common interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import numpy as np


class BaseCRUD(ABC):
    """Abstract base class for source-specific CRUD operations"""

    @abstractmethod
    def get_table_name(self) -> str:
        """Return the table name for this CRUD implementation"""
        pass

    @abstractmethod
    def search_assets(
        self,
        conn,
        query_embedding_clip: np.ndarray,
        query_embedding_sbert: np.ndarray,
        top_k: int,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search assets in this source's table"""
        pass

    @abstractmethod
    def insert_assets(self, conn, assets: List[Any]) -> None:
        """Insert assets into this source's table"""
        pass

    @abstractmethod
    def get_asset_by_uid(self, conn, uid: str) -> Optional[Dict[str, Any]]:
        """Get a single asset by UID"""
        pass

    @abstractmethod
    def count_assets(self, conn) -> int:
        """Count total assets in this source's table"""
        pass

    def get_assets_by_category(
        self, conn, category: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get assets by category (optional override for source-specific logic)"""
        # Default implementation - can be overridden
        return []
