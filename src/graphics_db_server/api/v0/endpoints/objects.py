import base64
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from graphics_db_server.core.config import GRAPHICS_DB_BASE_URL, SCALE_MAX_LENGTH_THRESHOLD
from graphics_db_server.db.session import get_db_connection
from graphics_db_server.db import crud
from graphics_db_server.embeddings.clip import get_clip_embeddings
from graphics_db_server.embeddings.sbert import get_sbert_embeddings
from graphics_db_server.logging import logger
from graphics_db_server.schemas.asset import Asset
from graphics_db_server.sources.from_objaverse import (
    download_objects,
    get_thumbnails,
    locate_objects,
)
from graphics_db_server.utils.scale_validation import validate_object_scales
from graphics_db_server.utils.geometry import get_glb_dimensions
from graphics_db_server.utils.rounding import safe_round_dict

router = APIRouter()


@router.get("/objects/search", response_model=list[Asset])
def search_objects(query: str, top_k: int = 5, validate_scale: bool = False):
    """
    Finds the top_k most similar 3D objects for a given query.
    """
    query_embedding_clip = get_clip_embeddings(query)
    query_embedding_sbert = get_sbert_embeddings(query)
    with get_db_connection() as conn:
        results: list[dict] = crud.search_objects(
            conn=conn,
            query_embedding_clip=query_embedding_clip,
            query_embedding_sbert=query_embedding_sbert,
            top_k=top_k,
        )

    if not results:
        logger.debug(f"No results found for query: {query}")
        return []
    elif validate_scale:
        uids = [object["uid"] for object in results]

        # Attempt to locate in FS first, then attempt download
        object_paths = locate_objects(uids)
        found_uids = []
        for path in object_paths:
            uid = Path(path).stem.replace("_rescaled", "")
            found_uids.append(uid)
        missing_uids = list(set(uids) - set(found_uids))
        if missing_uids:  # is not empty
            object_paths += download_objects(missing_uids)
            # TODO: recoup paths for missing uids *while preserving order*

        # TODO: if uid has entry in extra_index db, use its information instead of
        #       relying on heuristic-based `validate_object_scales()`.
        # NOTE: The best way to achieve the necessary changes in behavior
        #       is to make `validate_object_scales()` take in uids instead of paths
        #       so that it only performs object location/download only when necessary,
        #       and prefer to lookup extra_index_db.
        #       That way, the complicated locate-or-download logic can be simplified/hidden too
        validation_results = validate_object_scales(
            object_paths, SCALE_MAX_LENGTH_THRESHOLD
        )
        return [
            object for object in results if validation_results.get(object["uid"], False)
        ]
    return results


class ObjectThumbnailsRequest(BaseModel):
    uids: list[str]
    format: str = "urls"  # "urls" or "base64" or "path"
    # TODO: change urls → url, after confirming it is better


@router.get("/objects/{object_uid}/thumbnail")
def get_object_thumbnail(object_uid: str):
    """
    Gets the thumbnail image for a given 3D object UID.
    Returns the image file directly for use in markdown or web pages.
    """
    try:
        object_paths = locate_objects([object_uid]) or download_objects(
            [object_uid]
        )  # singleton
        if object_uid not in object_paths:
            raise HTTPException(status_code=404, detail="Object not found")

        object_thumbnails = get_thumbnails(object_paths)
        if object_uid not in object_thumbnails:
            raise HTTPException(status_code=404, detail="Thumbnail not found")

        return FileResponse(
            path=object_thumbnails[object_uid],
            media_type="image/png",
            filename=f"{object_uid}_thumbnail.png",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving thumbnail for object {object_uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve thumbnail")


@router.post("/objects/thumbnails")
def get_object_thumbnails(request: ObjectThumbnailsRequest):
    """
    Gets object thumbnails for a list of 3D object UIDs.
    Returns either base64-encoded data or URLs based on format parameter.
    """
    object_paths = download_objects(request.uids)

    response_data = {}

    if request.format == "base64":
        # Return base64-encoded image data
        object_thumbnails = get_thumbnails(object_paths)
        for uid, image_path in object_thumbnails.items():
            with open(image_path, "rb") as f:
                image_data = f.read()
            response_data[uid] = base64.b64encode(image_data).decode("utf-8")
    elif request.format == "path":
        for uid in request.uids:
            if uid in object_paths:
                response_data[uid] = f"/api/v0/objects/{uid}/thumbnail"
    else:  # default to urls
        # Return URLs pointing to the thumbnail endpoint
        for uid in request.uids:
            if uid in object_paths:
                response_data[uid] = f"/api/v0/objects/{uid}/thumbnail"

    return JSONResponse(content=response_data)


@router.get("/objects/{object_uid}/metadata")
def get_object_metadata(object_uid: str):
    """
    Gets metadata for a given 3D object UID, including dimensions.
    """
    try:
        # TODO: improve efficiency by utilizing extra_index db if uid exists in it
        object_paths = locate_objects([object_uid]) or download_objects(
            [object_uid]
        )  # singleton

        if object_uid not in object_paths:
            raise HTTPException(status_code=404, detail="Object not found")

        glb_path = object_paths[object_uid]
        success, dimensions, error = get_glb_dimensions(glb_path)

        if not success:
            logger.error(f"Error getting dimensions for object {object_uid}: {error}")
            raise HTTPException(
                status_code=500, detail="Failed to get object dimensions"
            )

        x_size, y_size, z_size = dimensions
        metadata = {
            "uid": object_uid,
            "dimensions": safe_round_dict({"x": x_size, "y": y_size, "z": z_size}, 2),
        }

        # return JSONResponse(content=metadata)
        return metadata  # HACK?

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metadata for object {object_uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get object metadata")


@router.get("/objects/download/{object_uid}/glb")
def download_object_glb(object_uid: str):
    """
    Downloads the .glb file for a given 3D object UID.
    """
    try:
        # TODO: check what dataset uid belongs to, and then do downstream
        #       file search / download / access accordingly.
        object_paths = locate_objects([object_uid]) or download_objects([object_uid])

        if object_uid not in object_paths:
            raise HTTPException(status_code=404, detail="Object not found")

        glb_path = object_paths[object_uid]

        return FileResponse(
            path=glb_path, media_type="model/gltf-binary", filename=f"{object_uid}.glb"
        )
    except Exception as e:
        logger.error(f"Error serving .glb file for object {object_uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve .glb file")


@router.get("/objects/locate/{object_uid}/glb")
def locate_object_glb(object_uid: str):
    """
    Retrieves the path of .glb file in the local filesystem (where graphics-db is running).
    This is useful if the client application is running on the same PC or a shared filesystem.
    """
    try:
        # TODO: check *what dataset* object_uid belongs to, and then do downstream
        #       file search / download / access accordingly.
        object_paths = locate_objects([object_uid])
        # print(f"{object_paths=}")  # DEBUG

        if object_uid not in object_paths:
            raise HTTPException(status_code=404, detail="Object not found")

        glb_path = object_paths[object_uid]

        response_data = {}
        response_data["path"] = glb_path
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Error serving .glb file path for object {object_uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve .glb file path")


# NOTE: not sure whether what endpoint name is best.
# other candidates: vlm, investigate, ai, ai_search, report, explain
@router.get("/objects/report", response_model=str)
def generate_object_search_report(
    uids: list[str] = Query(),
    report_format: str = "markdown",
    image_format: str = "url",
):
    """
    Returns an LLM/VLM-consumable report with thumbnails and metadata for 3D objects.
    """
    if report_format != "markdown":
        raise NotImplementedError("Only markdown is supported for now")
    doc = ""
    doc += "\n"
    for uid in uids:
        doc += f"\n### {uid}"
        doc += "\n"
        doc += "\n**Thumbnails**:"
        doc += "\n"
        doc += "\nIsometric:"
        doc += "\n"
        # doc += f"\n![thumbnail]({get_object_thumbnail(uid).path})"
        # doc += f"\n![thumbnail]{GRAPHICS_DB_BASE_URL}/api/v0/objects/{uid}/thumbnail)"
        match image_format:
            case "url":
                doc += f"\n![thumbnail_for_{uid}]({GRAPHICS_DB_BASE_URL}/api/v0/objects/{uid}/thumbnail)"
            case "path":
                object_paths = locate_objects([uid])  # or download_objects([uid])
                doc += f"\n![thumbnail_for_{uid}]({get_thumbnails(object_paths)[uid]})"
        doc += "\n"
        doc += "\n**Metadata**:"
        doc += "\n"
        doc += f"\n{get_object_metadata(uid)}"
        doc += "\n"

    return doc
