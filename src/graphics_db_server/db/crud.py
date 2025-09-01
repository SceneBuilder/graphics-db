"""
Create, Read, Update, Delete operations in the database.
"""

import json
from typing import List

import numpy as np
from psycopg.rows import dict_row

from graphics_db_server.core.config import TABLE_NAME
from graphics_db_server.schemas.asset import Asset
from graphics_db_server.logging import logger


def search_assets(
    conn, query_embedding_clip: np.ndarray, query_embedding_sbert: np.ndarray, top_k: int
) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT
                uid,
                url,
                tags,
                source,
                license,
                asset_type,
                (1 - (clip_embedding <=> %(query_vector_clip)s)) + (1 - (sbert_embedding <=> %(query_vector_sbert)s)) as similarity_score
            FROM {TABLE_NAME}
            ORDER BY (clip_embedding <=> %(query_vector_clip)s) + (sbert_embedding <=> %(query_vector_sbert)s)
            LIMIT %(limit)s;
            """,
            {
                "query_vector_clip": query_embedding_clip,
                "query_vector_sbert": query_embedding_sbert,
                "limit": top_k,
            },
        )
        results = cur.fetchall()
    if not results:
        logger.warning("No results found. The database might be empty.")
    return results


def insert_assets(conn, assets: List[Asset]):
    """
    Inserts assets into the database.
    """
    data = []

    for asset in assets:
        data.append(
            (
                asset.uid,
                asset.url,
                asset.tags,
                asset.source,
                asset.license,
                asset.asset_type,
                asset.clip_embedding,
                asset.sbert_embedding,
            )
        )

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO assets (uid, url, tags, source, license, asset_type, clip_embedding, sbert_embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            data,
        )
        conn.commit()

    logger.success(f"Inserted {len(assets)} assets.")
