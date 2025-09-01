"""
CRUD operations for Objaverse assets table.
"""

from typing import List, Dict, Any, Optional

import numpy as np
from psycopg.rows import dict_row

from .base_crud import BaseCRUD
from graphics_db_server.schemas import ObjaverseAssetCreate
from graphics_db_server.logging import logger


class ObjaverseCRUD(BaseCRUD):
    """CRUD operations for objaverse_assets table"""

    def get_table_name(self) -> str:
        return "objaverse_assets"

    def search_assets(
        self,
        conn,
        query_embedding_clip: np.ndarray,
        query_embedding_sbert: np.ndarray,
        top_k: int,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search Objaverse assets with optional category filtering"""

        with conn.cursor(row_factory=dict_row) as cur:
            # Build query with optional category filter
            base_query = """
            SELECT
                uid,
                viewer_url as url,
                tags,
                license,
                asset_category,
                geometric_complexity,
                has_textures,
                file_format,
                (1 - (clip_embedding <=> %(query_vector_clip)s)) + (1 - (sbert_embedding <=> %(query_vector_sbert)s)) as similarity_score
            FROM objaverse_assets
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
            logger.warning("No Objaverse results found.")

        return results

    def insert_assets(self, conn, assets: List[ObjaverseAssetCreate]) -> None:
        """Insert Objaverse assets into the database"""

        data = []
        for asset in assets:
            db_dict = asset.to_db_dict()
            data.append(
                (
                    db_dict["uid"],
                    db_dict["viewer_url"],
                    db_dict["license"],
                    db_dict["tags"],
                    db_dict["asset_category"],
                    db_dict["geometric_complexity"],
                    db_dict["has_textures"],
                    db_dict["file_format"],
                    db_dict["clip_embedding"],
                    db_dict["sbert_embedding"],
                )
            )

        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO objaverse_assets 
                   (uid, viewer_url, license, tags, asset_category, geometric_complexity, 
                    has_textures, file_format, clip_embedding, sbert_embedding) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                data,
            )
            conn.commit()

        logger.success(f"Inserted {len(assets)} Objaverse assets.")

    def get_asset_by_uid(self, conn, uid: str) -> Optional[Dict[str, Any]]:
        """Get a single Objaverse asset by UID"""

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT uid, viewer_url as url, tags, license, asset_category,
                          geometric_complexity, has_textures, file_format
                   FROM objaverse_assets 
                   WHERE uid = %s""",
                (uid,),
            )
            result = cur.fetchone()

        return result

    def count_assets(self, conn) -> int:
        """Count total Objaverse assets"""

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM objaverse_assets")
            count = cur.fetchone()[0]

        return count

    def get_assets_by_category(
        self, conn, category: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get Objaverse assets by category"""

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT uid, viewer_url as url, tags, license, asset_category,
                          geometric_complexity, has_textures, file_format
                   FROM objaverse_assets 
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
                   FROM objaverse_assets 
                   WHERE asset_category IS NOT NULL
                   GROUP BY asset_category 
                   ORDER BY count DESC"""
            )
            results = cur.fetchall()

        return results


# Singleton instance
objaverse_crud = ObjaverseCRUD()
