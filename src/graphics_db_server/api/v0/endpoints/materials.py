import base64
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel


from graphics_db_server.db.session import get_db_connection
from graphics_db_server.db import crud
from graphics_db_server.embeddings.clip import get_clip_embeddings
from graphics_db_server.embeddings.sbert import get_sbert_embeddings
from graphics_db_server.logging import logger
from graphics_db_server.schemas.asset import Asset
from graphics_db_server.sources.from_polyhaven import download_diffuse_maps


router = APIRouter()


@router.get("/materials/search", response_model=list[Asset])
def search_materials(query: str, top_k: int = 5):
    """
    Finds the top_k most similar materials for a given query.
    """
    try:
        query_embedding_clip = get_clip_embeddings(query)
        query_embedding_sbert = get_sbert_embeddings(query)
        with get_db_connection() as conn:
            results: list[dict] = crud.search_materials(
                conn=conn,
                query_embedding_clip=query_embedding_clip,
                query_embedding_sbert=query_embedding_sbert,
                top_k=top_k,
            )

        if not results:
            logger.debug(f"No materials found for query: {query}")
            return []

        logger.info(f"Found {len(results)} materials for query: '{query}'")
        return results

    except Exception as e:
        logger.error(f"Error searching materials: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during material search"
        )


class MaterialThumbnailsRequest(BaseModel):
    material_uids: list[str]


@router.post("/materials/thumbnails")
def get_material_thumbnails(request: MaterialThumbnailsRequest):
    """
    Gets material thumbnails for a list of material UIDs.
    """
    try:
        if not request.material_uids:
            raise HTTPException(status_code=400, detail="No material UIDs provided")

        if len(request.material_uids) > 10:
            raise HTTPException(status_code=400, detail="Too many materials (max 10)")

        material_paths = download_diffuse_maps(request.material_uids)

        response_data = {}
        for uid, image_path in material_paths.items():
            try:
                with open(image_path, "rb") as f:
                    image_data = f.read()
                response_data[uid] = base64.b64encode(image_data).decode("utf-8")
            except Exception as e:
                logger.warning(f"Failed to read thumbnail for {uid}: {e}")

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting material thumbnails: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error getting thumbnails"
        )


@router.get("/materials/download/{material_uid}/diffuse")
def download_diffuse_map(material_uid: str):
    """
    Downloads the diffuse texture map for a given Poly Haven material UID.
    Returns the diffuse/albedo texture as a JPEG image file.
    """
    try:
        material_paths = download_diffuse_maps([material_uid])

        if material_uid not in material_paths:
            raise HTTPException(status_code=404, detail="Material not found")

        material_path = material_paths[material_uid]

        return FileResponse(
            path=material_path,
            media_type="image/jpeg",
            filename=f"{material_uid}_diffuse.jpg",
        )
    except Exception as e:
        logger.error(f"Error serving diffuse map for material {material_uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve diffuse map")


@router.get("/materials/{material_uid}/metadata")
def get_material_metadata(material_uid: str):
    """
    Gets metadata for a given material UID.
    """
    try:
        with get_db_connection() as conn:
            material = crud.get_asset_by_uid(conn, material_uid)

        if not material:
            raise HTTPException(status_code=404, detail="Material not found")

        metadata = {
            "uid": material_uid,
            "source": "polyhaven",
            "license": "CC0",
            **material,
        }

        return JSONResponse(content=metadata)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metadata for material {material_uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get material metadata")
