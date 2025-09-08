import base64
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from graphics_db_server.core.config import SCALE_MAX_LENGTH_THRESHOLD
from graphics_db_server.db.session import get_db_connection
from graphics_db_server.db import crud
from graphics_db_server.embeddings.clip import get_clip_embeddings
from graphics_db_server.embeddings.sbert import get_sbert_embeddings
from graphics_db_server.logging import logger
from graphics_db_server.schemas.asset import Asset
from graphics_db_server.sources.from_objaverse import (
    download_assets,
    get_thumbnails,
    locate_assets,
)
from graphics_db_server.utils.scale_validation import validate_asset_scales
from graphics_db_server.utils.geometry import get_glb_dimensions

router = APIRouter()


@router.get("/assets/search", response_model=list[Asset])
def search_assets(query: str, top_k: int = 5, validate_scale: bool = False):
    """
    Finds the top_k most similar assets for a given query.
    """
    query_embedding_clip = get_clip_embeddings(query)
    query_embedding_sbert = get_sbert_embeddings(query)
    with get_db_connection() as conn:
        results: list[dict] = crud.search_assets(
            conn=conn,
            query_embedding_clip=query_embedding_clip,
            query_embedding_sbert=query_embedding_sbert,
            top_k=top_k,
        )

    if not results:
        logger.debug(f"No results found for query: {query}")
        return []
    elif validate_scale:
        asset_uids = [asset["uid"] for asset in results]
        asset_paths = download_assets(asset_uids)
        validation_results = validate_asset_scales(
            asset_paths, SCALE_MAX_LENGTH_THRESHOLD
        )
        return [asset for asset in results if validation_results.get(asset["uid"], False)]
    return results


class AssetThumbnailsRequest(BaseModel):
    asset_uids: list[str]


@router.post("/assets/thumbnails")
def get_asset_thumbnails(request: AssetThumbnailsRequest):
    """
    Gets asset thumbnails for a list of asset UIDs.
    """
    asset_paths = download_assets(request.asset_uids)
    asset_thumbnails = get_thumbnails(asset_paths)

    response_data = {}
    for uid, image_path in asset_thumbnails.items():
        with open(image_path, "rb") as f:
            image_data = f.read()
        response_data[uid] = base64.b64encode(image_data).decode("utf-8")

    return JSONResponse(content=response_data)


@router.get("/assets/download/{asset_uid}/glb")
def download_glb_file(asset_uid: str):
    """
    Downloads the .glb file for a given asset UID.
    """
    try:
        asset_paths = download_assets([asset_uid])

        if asset_uid not in asset_paths:
            raise HTTPException(status_code=404, detail="Asset not found")

        glb_path = asset_paths[asset_uid]

        return FileResponse(
            path=glb_path, media_type="model/gltf-binary", filename=f"{asset_uid}.glb"
        )
    except Exception as e:
        logger.error(f"Error serving .glb file for asset {asset_uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve .glb file")


@router.get("/assets/locate/{asset_uid}/glb")
def locate_glb_file(asset_uid: str):
    """
    Retrieves the path of .glb file in the local filesystem (where graphics-db is running).
    This is useful if the client application is running on the same PC or a shared filesystem.
    """
    try:
        # TODO: check what dataset uid belongs to, and then do downstream
        #       file search / download / access accordingly.
        asset_paths = locate_assets([asset_uid])

        if asset_uid not in asset_paths:
            raise HTTPException(status_code=404, detail="Asset not found")

        glb_path = asset_paths[asset_uid]

        response_data = {}
        response_data["path"] = glb_path
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Error serving .glb file path for asset {asset_uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve .glb file path")


@router.get("/assets/{asset_uid}/metadata")
def get_asset_metadata(asset_uid: str):
    """
    Gets metadata for a given asset UID, including dimensions.
    """
    try:
        asset_paths = download_assets([asset_uid])

        if asset_uid not in asset_paths:
            raise HTTPException(status_code=404, detail="Asset not found")

        glb_path = asset_paths[asset_uid]
        success, dimensions, error = get_glb_dimensions(glb_path)

        if not success:
            logger.error(f"Error getting dimensions for asset {asset_uid}: {error}")
            raise HTTPException(status_code=500, detail="Failed to get asset dimensions")

        x_size, y_size, z_size = dimensions
        metadata = {
            "uid": asset_uid,
            "dimensions": {
                "x": x_size,
                "y": y_size,
                "z": z_size
            }
        }

        return JSONResponse(content=metadata)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metadata for asset {asset_uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get asset metadata")
