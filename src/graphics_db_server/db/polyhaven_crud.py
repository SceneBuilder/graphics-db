"""
CRUD operations for Poly Haven assets table.
"""

from typing import List, Dict, Any, Optional

import numpy as np
from psycopg.rows import dict_row

from .base_crud import BaseCRUD
from graphics_db_server.schemas import PolyhavenAssetCreate
from graphics_db_server.logging import logger


class PolyhavenCRUD(BaseCRUD):
    """CRUD operations for polyhaven_assets table"""

    def get_table_name(self) -> str:
        return "polyhaven_assets"

    def search_assets(
        self,
        conn,
        query_embedding_clip: np.ndarray,
        query_embedding_sbert: np.ndarray,
        top_k: int,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search Poly Haven assets with optional category filtering"""

        with conn.cursor(row_factory=dict_row) as cur:
            # Build query with optional category filter
            base_query = """
            SELECT
                uid,
                polyhaven_url as url,
                tags,
                categories,
                asset_category,
                asset_subcategory,
                surface_type,
                material_properties,
                resolution_available,
                asset_type,
                (1 - (clip_embedding <=> %(query_vector_clip)s)) + (1 - (sbert_embedding <=> %(query_vector_sbert)s)) as similarity_score
            FROM polyhaven_assets
            """

            where_clause = ""
            params = {
                "query_vector_clip": query_embedding_clip,
                "query_vector_sbert": query_embedding_sbert,
                "limit": top_k,
            }

            if category_filter:
                where_clause = "WHERE asset_category = %(category_filter)s"
                params["category_filter"] = category_filter

            order_clause = """
            ORDER BY (clip_embedding <=> %(query_vector_clip)s) + (sbert_embedding <=> %(query_vector_sbert)s)
            LIMIT %(limit)s
            """

            query = f"{base_query} {where_clause} {order_clause}"

            cur.execute(query, params)
            results = cur.fetchall()

        if not results:
            logger.warning("No Poly Haven results found.")

        return results

    def insert_assets(self, conn, assets: List[PolyhavenAssetCreate]) -> None:
        """Insert Poly Haven assets into the database"""

        data = []
        for asset in assets:
            db_dict = asset.to_db_dict()
            data.append(
                (
                    db_dict["uid"],
                    db_dict["polyhaven_url"],
                    db_dict["asset_category"],
                    db_dict["asset_subcategory"],
                    db_dict["surface_type"],
                    db_dict["material_properties"],
                    db_dict["resolution_available"],
                    db_dict["tags"],
                    db_dict["categories"],
                    db_dict["asset_type"],
                    db_dict["clip_embedding"],
                    db_dict["sbert_embedding"],
                )
            )

        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO polyhaven_assets 
                   (uid, polyhaven_url, asset_category, asset_subcategory, surface_type,
                    material_properties, resolution_available, tags, categories, asset_type,
                    clip_embedding, sbert_embedding) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                data,
            )
            conn.commit()

        logger.success(f"Inserted {len(assets)} Poly Haven assets.")

    def get_asset_by_uid(self, conn, uid: str) -> Optional[Dict[str, Any]]:
        """Get a single Poly Haven asset by UID"""

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT uid, polyhaven_url as url, tags, categories, asset_category,
                          asset_subcategory, surface_type, material_properties,
                          resolution_available, asset_type
                   FROM polyhaven_assets 
                   WHERE uid = %s""",
                (uid,),
            )
            result = cur.fetchone()

        return result

    def count_assets(self, conn) -> int:
        """Count total Poly Haven assets"""

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM polyhaven_assets")
            count = cur.fetchone()[0]

        return count

    def get_assets_by_category(
        self, conn, category: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get Poly Haven assets by category"""

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT uid, polyhaven_url as url, tags, categories, asset_category,
                          asset_subcategory, surface_type, material_properties,
                          resolution_available, asset_type
                   FROM polyhaven_assets 
                   WHERE asset_category = %s 
                   LIMIT %s""",
                (category, limit),
            )
            results = cur.fetchall()

        return results

    def get_categories(self, conn) -> List[Dict[str, Any]]:
        """Get all available categories with counts"""

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT asset_category, COUNT(*) as count
                   FROM polyhaven_assets 
                   WHERE asset_category IS NOT NULL
                   GROUP BY asset_category 
                   ORDER BY count DESC"""
            )
            results = cur.fetchall()

        return results

    def get_materials_by_surface_type(
        self, conn, surface_type: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get materials by surface type (Poly Haven specific)"""

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT uid, polyhaven_url as url, tags, asset_category,
                          asset_subcategory, surface_type, material_properties
                   FROM polyhaven_assets 
                   WHERE surface_type = %s 
                   LIMIT %s""",
                (surface_type, limit),
            )
            results = cur.fetchall()

        return results


# Singleton instance
polyhaven_crud = PolyhavenCRUD()
