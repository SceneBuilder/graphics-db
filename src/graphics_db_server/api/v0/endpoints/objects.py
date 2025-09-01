"""
Objects API - Objaverse furniture, decor, and 3D models for floorplan placement.
"""

import base64
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from graphics_db_server.db.session import get_db_connection
from graphics_db_server.db.objaverse_crud import objaverse_crud
from graphics_db_server.embeddings.clip import get_clip_embeddings
from graphics_db_server.embeddings.sbert import get_sbert_embeddings
from graphics_db_server.logging import logger
from graphics_db_server.sources.from_objaverse import (
    download_assets,
    get_thumbnails,
)

router = APIRouter(prefix="/objects", tags=["objects"])


@router.get("/search")
def search_objects(
    query: str,
    top_k: int = Query(10, ge=1, le=100, description="Number of results to return"),
    category: Optional[str] = Query(
        None, description="Filter by category (furniture, vehicle, character, building)"
    ),
    has_textures: Optional[bool] = Query(
        None, description="Filter by whether object has built-in textures"
    ),
    complexity: Optional[str] = Query(
        None,
        regex="^(simple|moderate|complex)$",
        description="Filter by geometric complexity",
    ),
):
    """
    Search 3D objects for floorplan placement.

    - **query**: Search text (e.g., "dining table", "office chair", "sofa")
    - **top_k**: Maximum results to return (1-100)
    - **category**: Filter by object category
    - **has_textures**: Filter objects with/without built-in materials
    - **complexity**: Filter by model complexity level

    Returns complete 3D models ready for placement in scenes.
    """
    try:
        # Generate embeddings for search
        query_embedding_clip = get_clip_embeddings(query)
        query_embedding_sbert = get_sbert_embeddings(query)

        with get_db_connection() as conn:
            # Search only Objaverse assets
            results = objaverse_crud.search_assets(
                conn=conn,
                query_embedding_clip=query_embedding_clip,
                query_embedding_sbert=query_embedding_sbert,
                top_k=top_k,
                category_filter=category,
            )

            # Apply additional filters
            filtered_results = results
            if has_textures is not None:
                filtered_results = [
                    r for r in filtered_results if r.get("has_textures") == has_textures
                ]

            if complexity:
                filtered_results = [
                    r
                    for r in filtered_results
                    if r.get("geometric_complexity") == complexity
                ]

            # Limit after filtering
            filtered_results = filtered_results[:top_k]

            # Add context for objects
            for result in filtered_results:
                result["source"] = "objaverse"
                result["asset_type"] = "3d_model"
                result["usage"] = "place_in_scene"
                result["file_format"] = result.get("file_format", "glb")

        response = {
            "results": filtered_results,
            "query": query,
            "total_results": len(filtered_results),
            "filters_applied": {
                "category": category,
                "has_textures": has_textures,
                "complexity": complexity,
            },
        }

        if not filtered_results:
            logger.debug(f"No objects found for query: {query}")
        else:
            logger.info(f"Found {len(filtered_results)} objects for query: '{query}'")

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error searching objects: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during object search"
        )


@router.get("/categories")
def get_object_categories():
    """Get available object categories with counts."""

    try:
        with get_db_connection() as conn:
            categories = objaverse_crud.get_categories(conn)

        response = {"categories": categories, "total_categories": len(categories)}

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error getting object categories: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error getting categories"
        )


@router.get("/category/{category_name}")
def get_objects_by_category(
    category_name: str,
    limit: int = Query(50, ge=1, le=200, description="Number of objects to return"),
):
    """Get objects by specific category."""

    try:
        with get_db_connection() as conn:
            objects = objaverse_crud.get_assets_by_category(conn, category_name, limit)

        if not objects:
            raise HTTPException(
                status_code=404, detail=f"No objects found in category: {category_name}"
            )

        response = {
            "category": category_name,
            "objects": objects,
            "count": len(objects),
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting objects by category: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error getting objects by category"
        )


class ObjectThumbnailRequest(BaseModel):
    object_uids: List[str]


@router.post("/thumbnails")
def generate_object_thumbnails(request: ObjectThumbnailRequest):
    """
    Generate thumbnail images for 3D objects.

    Downloads the .glb files and creates preview images using PyVista.
    Returns base64-encoded thumbnail images.
    """
    try:
        if not request.object_uids:
            raise HTTPException(status_code=400, detail="No object UIDs provided")

        if len(request.object_uids) > 20:
            raise HTTPException(
                status_code=400, detail="Too many objects requested (max 20)"
            )

        # Download 3D assets
        asset_paths = download_assets(request.object_uids)

        if not asset_paths:
            raise HTTPException(
                status_code=404, detail="No objects could be downloaded"
            )

        # Generate thumbnails
        asset_thumbnails = get_thumbnails(asset_paths)

        # Convert to base64
        response_data = {}
        for uid, image_path in asset_thumbnails.items():
            try:
                with open(image_path, "rb") as f:
                    image_data = f.read()
                response_data[uid] = base64.b64encode(image_data).decode("utf-8")
            except Exception as e:
                logger.warning(f"Failed to read thumbnail for {uid}: {e}")
                response_data[uid] = None

        logger.info(f"Generated {len(response_data)} object thumbnails")

        return JSONResponse(
            content={
                "thumbnails": response_data,
                "generated_count": len(
                    [t for t in response_data.values() if t is not None]
                ),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating object thumbnails: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error generating thumbnails"
        )


@router.get("/{object_uid}")
def get_object_details(object_uid: str):
    """Get detailed information about a specific object."""

    try:
        with get_db_connection() as conn:
            obj = objaverse_crud.get_asset_by_uid(conn, object_uid)

        if not obj:
            raise HTTPException(
                status_code=404, detail=f"Object not found: {object_uid}"
            )

        # Add additional context
        obj["source"] = "objaverse"
        obj["asset_type"] = "3d_model"
        obj["usage"] = "place_in_scene"

        return JSONResponse(content=obj)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting object details: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error getting object details"
        )
