"""
Create, Read, Update, Delete operations in the database.
"""

import json
from typing import List

import numpy as np
from psycopg.rows import dict_row

from graphics_db_server.schemas.asset import Asset
from graphics_db_server.logging import logger


def search_assets(
    conn,
    query_embedding_clip: np.ndarray,
    query_embedding_sbert: np.ndarray,
    top_k: int,
) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            (SELECT
                uid,
                url,
                tags,
                source,
                license,
                asset_type,
                (1 - (clip_embedding <=> %(query_vector_clip)s)) + (1 - (sbert_embedding <=> %(query_vector_sbert)s)) as similarity_score
            FROM objaverse_assets
            ORDER BY (clip_embedding <=> %(query_vector_clip)s) + (sbert_embedding <=> %(query_vector_sbert)s)
            LIMIT %(limit)s)
            UNION ALL
            (SELECT
                uid,
                url,
                tags,
                source,
                license,
                asset_type,
                (1 - (clip_embedding <=> %(query_vector_clip)s)) + (1 - (sbert_embedding <=> %(query_vector_sbert)s)) as similarity_score
            FROM polyhaven_assets
            ORDER BY (clip_embedding <=> %(query_vector_clip)s) + (sbert_embedding <=> %(query_vector_sbert)s)
            LIMIT %(limit)s)
            ORDER BY similarity_score DESC
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


def search_materials(
    conn,
    query_embedding_clip: np.ndarray,
    query_embedding_sbert: np.ndarray,
    top_k: int,
) -> list[dict]:
    """Search only polyhaven_assets table for materials."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT
                uid,
                url,
                tags,
                source,
                license,
                asset_type,
                (1 - (clip_embedding <=> %(query_vector_clip)s)) + (1 - (sbert_embedding <=> %(query_vector_sbert)s)) as similarity_score
            FROM polyhaven_assets
            ORDER BY (clip_embedding <=> %(query_vector_clip)s) + (sbert_embedding <=> %(query_vector_sbert)s)
            LIMIT %(limit)s
            """,
            {
                "query_vector_clip": query_embedding_clip,
                "query_vector_sbert": query_embedding_sbert,
                "limit": top_k,
            },
        )
        results = cur.fetchall()
    return results


def insert_objaverse_assets(conn, assets):
    data = [
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
        for asset in assets
    ]

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO objaverse_assets (uid, url, tags, source, license, asset_type, clip_embedding, sbert_embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            data,
        )
        conn.commit()
    logger.success(f"Inserted {len(assets)} Objaverse assets.")


def insert_polyhaven_assets(conn, assets):
    data = [
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
        for asset in assets
    ]

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO polyhaven_assets (uid, url, tags, source, license, asset_type, clip_embedding, sbert_embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            data,
        )
        conn.commit()
    logger.success(f"Inserted {len(assets)} Poly Haven assets.")


def get_asset_by_uid(conn, uid: str) -> dict:
    """
    Get a single asset by its UID from both tables.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        # Try objaverse_assets first
        cur.execute(
            """
            SELECT
                uid,
                url,
                tags,
                source,
                license,
                asset_type
            FROM objaverse_assets
            WHERE uid = %(uid)s
            """,
            {"uid": uid},
        )
        result = cur.fetchone()
        if result:
            return result

        # Try polyhaven_assets if not found
        cur.execute(
            """
            SELECT
                uid,
                url,
                tags,
                source,
                license,
                asset_type
            FROM polyhaven_assets
            WHERE uid = %(uid)s
            """,
            {"uid": uid},
        )
        result = cur.fetchone()
        return result
