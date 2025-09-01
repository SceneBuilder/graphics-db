"""
Materials API - Poly Haven textures and HDRIs for architectural surfaces.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from graphics_db_server.db.session import get_db_connection
from graphics_db_server.db.polyhaven_crud import polyhaven_crud
from graphics_db_server.embeddings.clip import get_clip_embeddings
from graphics_db_server.embeddings.sbert import get_sbert_embeddings
from graphics_db_server.logging import logger
from graphics_db_server.sources.from_polyhaven import download_diffuse_maps

router = APIRouter(prefix="/materials", tags=["materials"])


@router.get("/search")
def search_materials(
    query: str,
    top_k: int = Query(10, ge=1, le=100, description="Number of results to return"),
    category: Optional[str] = Query(
        None, description="Filter by material category (wood, metal, fabric, stone)"
    ),
    surface_type: Optional[str] = Query(
        None, regex="^(rough|smooth|glossy)$", description="Filter by surface texture"
    ),
    asset_type: Optional[str] = Query(
        "texture", regex="^(texture|hdri)$", description="Material type"
    ),
    resolution: Optional[str] = Query(
        None, regex="^(1k|2k|4k|8k)$", description="Minimum resolution available"
    ),
):
    """
    Search surface materials for floorplans and architecture.

    - **query**: Search text (e.g., "oak flooring", "brick wall", "marble countertop")
    - **top_k**: Maximum results to return (1-100)
    - **category**: Filter by material type
    - **surface_type**: Filter by surface texture characteristics
    - **asset_type**: "texture" for surfaces or "hdri" for environment lighting
    - **resolution**: Minimum resolution required

    Returns PBR materials for applying to architectural surfaces.
    """
    try:
        # Generate embeddings for search
        query_embedding_clip = get_clip_embeddings(query)
        query_embedding_sbert = get_sbert_embeddings(query)

        with get_db_connection() as conn:
            # Search only Poly Haven assets
            results = polyhaven_crud.search_assets(
                conn=conn,
                query_embedding_clip=query_embedding_clip,
                query_embedding_sbert=query_embedding_sbert,
                top_k=top_k,
                category_filter=category,
            )

            # Apply additional filters
            filtered_results = results

            if surface_type:
                filtered_results = [
                    r for r in filtered_results if r.get("surface_type") == surface_type
                ]

            if asset_type != "texture":  # Default is texture
                filtered_results = [
                    r for r in filtered_results if r.get("asset_type") == asset_type
                ]

            if resolution:
                resolution_order = {"1k": 1, "2k": 2, "4k": 4, "8k": 8}
                min_res_level = resolution_order.get(resolution, 1)

                filtered_results = [
                    r
                    for r in filtered_results
                    if any(
                        resolution_order.get(res, 0) >= min_res_level
                        for res in r.get("resolution_available", [])
                    )
                ]

            # Limit after filtering
            filtered_results = filtered_results[:top_k]

            # Add context for materials
            for result in filtered_results:
                result["source"] = "polyhaven"
                result["usage"] = "apply_to_surface"
                result["license"] = "CC0"

                # Add usage hints based on category
                category_name = result.get("asset_category", "material")
                if category_name in ["wood", "tile", "stone"]:
                    result["suggested_surfaces"] = ["floor", "wall"]
                elif category_name in ["fabric", "leather"]:
                    result["suggested_surfaces"] = ["furniture", "upholstery"]
                elif category_name == "metal":
                    result["suggested_surfaces"] = ["fixtures", "appliances"]
                else:
                    result["suggested_surfaces"] = ["any_surface"]

        response = {
            "results": filtered_results,
            "query": query,
            "total_results": len(filtered_results),
            "filters_applied": {
                "category": category,
                "surface_type": surface_type,
                "asset_type": asset_type,
                "resolution": resolution,
            },
        }

        if not filtered_results:
            logger.debug(f"No materials found for query: {query}")
        else:
            logger.info(f"Found {len(filtered_results)} materials for query: '{query}'")

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error searching materials: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during material search"
        )


@router.get("/categories")
def get_material_categories():
    """Get available material categories with counts."""

    try:
        with get_db_connection() as conn:
            categories = polyhaven_crud.get_categories(conn)

        response = {"categories": categories, "total_categories": len(categories)}

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error getting material categories: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error getting categories"
        )


@router.get("/surface-types")
def get_surface_types():
    """Get available surface types for filtering."""

    try:
        with get_db_connection() as conn:
            # Get surface types from Poly Haven assets
            surface_types = polyhaven_crud.get_materials_by_surface_type(
                conn, "rough", 1
            )  # Just to get structure

        # For now, return predefined surface types - could be made dynamic
        surface_types_info = [
            {"surface_type": "rough", "description": "Textured, bumpy surfaces"},
            {"surface_type": "smooth", "description": "Even, flat surfaces"},
            {"surface_type": "glossy", "description": "Shiny, reflective surfaces"},
        ]

        response = {"surface_types": surface_types_info}

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error getting surface types: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error getting surface types"
        )


@router.get("/category/{category_name}")
def get_materials_by_category(
    category_name: str,
    limit: int = Query(50, ge=1, le=200, description="Number of materials to return"),
):
    """Get materials by specific category."""

    try:
        with get_db_connection() as conn:
            materials = polyhaven_crud.get_assets_by_category(
                conn, category_name, limit
            )

        if not materials:
            raise HTTPException(
                status_code=404,
                detail=f"No materials found in category: {category_name}",
            )

        # Add usage context
        for material in materials:
            material["source"] = "polyhaven"
            material["usage"] = "apply_to_surface"
            material["license"] = "CC0"

        response = {
            "category": category_name,
            "materials": materials,
            "count": len(materials),
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting materials by category: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error getting materials by category",
        )


class MaterialPreviewRequest(BaseModel):
    material_uids: List[str]


@router.post("/previews")
def download_material_previews(request: MaterialPreviewRequest):
    """
    Download preview images (diffuse maps) for materials.

    Downloads 1K diffuse maps from Poly Haven for material preview.
    Returns local file paths or download status.
    """
    try:
        if not request.material_uids:
            raise HTTPException(status_code=400, detail="No material UIDs provided")

        if len(request.material_uids) > 10:
            raise HTTPException(
                status_code=400, detail="Too many materials requested (max 10)"
            )

        # Download diffuse maps
        downloaded_materials = download_diffuse_maps(request.material_uids)

        response_data = {
            "downloads": downloaded_materials,
            "successful_downloads": len(downloaded_materials),
            "requested_count": len(request.material_uids),
        }

        logger.info(f"Downloaded {len(downloaded_materials)} material previews")

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading material previews: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error downloading previews"
        )


@router.get("/{material_uid}")
def get_material_details(material_uid: str):
    """Get detailed information about a specific material."""

    try:
        with get_db_connection() as conn:
            material = polyhaven_crud.get_asset_by_uid(conn, material_uid)

        if not material:
            raise HTTPException(
                status_code=404, detail=f"Material not found: {material_uid}"
            )

        # Add additional context
        material["source"] = "polyhaven"
        material["usage"] = "apply_to_surface"
        material["license"] = "CC0"

        # Add download links (could be expanded to include all texture maps)
        material["available_maps"] = ["diffuse", "normal", "roughness", "displacement"]

        return JSONResponse(content=material)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting material details: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error getting material details"
        )


@router.get("/surface/{surface_type}")
def get_materials_by_surface_type(
    surface_type: str,
    limit: int = Query(50, ge=1, le=200, description="Number of materials to return"),
):
    """Get materials by surface texture type (Poly Haven specific feature)."""

    if surface_type not in ["rough", "smooth", "glossy"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid surface type. Use: rough, smooth, or glossy",
        )

    try:
        with get_db_connection() as conn:
            materials = polyhaven_crud.get_materials_by_surface_type(
                conn, surface_type, limit
            )

        if not materials:
            raise HTTPException(
                status_code=404,
                detail=f"No materials found with surface type: {surface_type}",
            )

        # Add usage context
        for material in materials:
            material["source"] = "polyhaven"
            material["usage"] = "apply_to_surface"
            material["license"] = "CC0"

        response = {
            "surface_type": surface_type,
            "materials": materials,
            "count": len(materials),
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting materials by surface type: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error getting materials by surface type",
        )
