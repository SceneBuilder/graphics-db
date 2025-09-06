"""
Helper functions for extracting scale analysis data from ObjaTHOR JSON files
and integrating with the graphics DB extra index setup.

NOTE: It seems plausible that `scale` attribute, especially if annotated by gpt 3.5,
      contains significant errors. However, the `size` attribute seems better, and
      we can utilize `calc_optimal_scaling_factor()` to redo SF calculation.
      → Upon consulting objathor codebase, `scale` actually seems to refer to the
      longest edge in meters—not a 0.0-1.0 float that represents scaling factor.
NOTE: The `size` attribute seems to assume z-up orientation, while bounding box
      suggests the model is in y-up orientation—so when feeding into `calc_optimal_scaling_factor`,
      it may be required to either assume the most common transformation, or heuristically mix-match
      the closest values, and report its occurrence in logs/console for potential analysis.
NOTE: `thor_metadata.boundingBox` attribute seems the most reliable for obtaining scale information.
NOTE: `z_axis_scale` attribute may contain reliable information for how to match the y/z size values.
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from graphics_db_server.core.config import OBJATHOR_ANNO_JSON_PATH
from graphics_db_server.logging import logger
from graphics_db_server.utils.rounding import safe_round
from graphics_db_server.utils.geometry import (
    calc_optimal_scaling_factor,
    get_glb_dimensions,
)
from graphics_db_server.utils.scale_validation import scale_glb_model

objathor_annotation = None


def load_objathor_annotation(filepath: str | Path) -> Dict[str, Any]:
    """
    Load ObjaTHOR metadata from JSON file.

    Args:
        path: Path to the JSON file containing ObjaTHOR metadata

    Returns:
        Dictionary with asset UUIDs as keys and metadata as values
    """
    global objathor_annotation
    if objathor_annotation is not None:
        logger.warning("ObjaTHOR metadata is already loaded! Skipping re-load.")
        return
    else:
        logger.info(f"Loading ObjaTHOR metadata from {filepath}")
        with open(Path(filepath).expanduser(), "r") as f:
            objathor_annotation = json.load(f)
        logger.info(f"Loaded ObjaTHOR metadata for {len(objathor_annotation)} assets")


load_objathor_annotation(OBJATHOR_ANNO_JSON_PATH)  # TEMP?


def extract_scale_analysis_from_objathor(
    uuid: str, objathor_data: Dict[str, Any], original_dims: Tuple[float, float, float]
) -> Optional[Dict[str, Any]]:
    """
    Extract scale analysis data from ObjaTHOR metadata for a given asset.

    Args:
        uuid: Asset UUID to look up
        objathor_data: Loaded ObjaTHOR metadata dictionary
        original_dims: Original dimensions of the asset [x, y, z] in meters

    Returns:
        Dictionary containing scale analysis results compatible with the existing
        metadata format, or None if UUID not found in ObjaTHOR data
    """
    if uuid not in objathor_data:
        return None

    asset_data = objathor_data[uuid]

    # Extract relevant information
    category = asset_data.get("category", "unknown")
    description = asset_data.get("description") or asset_data.get(
        "description_auto", ""
    )
    # scale_factor = asset_data.get("scale", 1.0)  # NOTE: this is incorrect; see front matter
    # if not asset_data["z_axis_scale"]:
    logger.debug(f"{asset_data['z_axis_scale']=}")
    # TODO: see if y-z dims mismatch happens in this case
    bbox = (
        asset_data.get("thor_metadata", None)
        .get("assetMetadata", None)
        .get("boundingBox", None)
    )  # NOTE: in meters already
    dmin, dmax = list(bbox["min"].values()), list(bbox["max"].values())
    desired_dims = [max - min for (min, max) in zip(dmin, dmax)]
    scale_factor = calc_optimal_scaling_factor(original_dims, desired_dims)
    annotated_size = asset_data.get("size", [])  # in cm

    # # Fallback: use original dimensions
    # desired_dims = list(original_dims)

    # Calculate if the object is misscaled
    # If scale factor is significantly different from 1.0, it's misscaled
    is_misscaled = not math.isclose(scale_factor, 1.0, abs_tol=0.1)

    # Determine misscaling type based on scale factor
    if is_misscaled:
        if math.isclose(scale_factor, 0.01, rel_tol=0.5):
            misscaling_type = "cm"
        elif math.isclose(scale_factor, 0.001, rel_tol=0.5):
            misscaling_type = "mm"
        else:
            misscaling_type = "arbitrary"
    else:
        misscaling_type = "N/A"

    # Calculate rescaled dimensions
    rescaled_dims = (
        [dim * scale_factor for dim in original_dims] if is_misscaled else [-1, -1, -1]
    )

    return {
        "object_description": f"{category}: {description}",
        "ideal_dimensions": f"[{desired_dims[0]:.2f}, {desired_dims[1]:.2f}, {desired_dims[2]:.2f}] m",
        "reasoning": f"Based on ObjaTHOR annotations: category={category}, THOR bbox size={desired_dims}",
        "misscaled": is_misscaled,
        "misscaling_type": misscaling_type,
        "correction_factor": scale_factor if is_misscaled else None,
        "category": category,
        "annotated_size_cm": annotated_size,
    }


def create_metadata_from_objathor(
    uuid: str,
    file_path: Path,
    original_dims: Tuple[float, float, float],
    objathor_data: Dict[str, Any],
    round_digits: int = 3,
) -> Dict[str, Any]:
    """
    Create metadata dictionary compatible with the existing database schema
    using ObjaTHOR data.

    Args:
        uuid: Asset UUID
        file_path: Path to the asset file
        original_dims: Original dimensions [x, y, z] in meters
        objathor_data: Loaded ObjaTHOR metadata
        round_digits: Number of decimal places for rounding

    Returns:
        Metadata dictionary compatible with database schema
    """
    analysis = extract_scale_analysis_from_objathor(uuid, objathor_data, original_dims)

    if analysis is None:
        # Fallback: assume well-scaled if not in ObjaTHOR data
        return {
            "misscaled": 0,
            "misscaling_type": "N/A",
            "dims_x": safe_round(original_dims[0], round_digits),
            "dims_y": safe_round(original_dims[1], round_digits),
            "dims_z": safe_round(original_dims[2], round_digits),
            "dims_xr": -1,
            "dims_yr": -1,
            "dims_zr": -1,
            "scaling_factor": -1,
            "fs_path": str(file_path),
            "fs_path_rescaled": None,
            "rescaled_by": "objathor-fallback",
        }

    sf = analysis["correction_factor"]
    scaled_model_path = None

    if analysis["misscaled"] and sf is not None:
        scaled_model_path = file_path.with_stem(f"{uuid}_scaled")

    return {
        "misscaled": 1 if analysis["misscaled"] else 0,
        "misscaling_type": analysis["misscaling_type"],
        "dims_x": safe_round(original_dims[0], round_digits),
        "dims_y": safe_round(original_dims[1], round_digits),
        "dims_z": safe_round(original_dims[2], round_digits),
        "dims_xr": safe_round(original_dims[0] * sf, round_digits)
        if analysis["misscaled"]
        else -1,
        "dims_yr": safe_round(original_dims[1] * sf, round_digits)
        if analysis["misscaled"]
        else -1,
        "dims_zr": safe_round(original_dims[2] * sf, round_digits)
        if analysis["misscaled"]
        else -1,
        "scaling_factor": safe_round(sf, round_digits) if analysis["misscaled"] else -1,
        "fs_path": str(file_path),
        "fs_path_rescaled": str(scaled_model_path) if analysis["misscaled"] else None,
        "rescaled_by": "objathor",
    }


def objathor_annotation_available(uuid: str) -> bool:
    """
    Check if we should use ObjaTHOR data for this asset instead of VLM analysis.

    Args:
        uuid: Asset UUID to check

    Returns:
        True if ObjaTHOR data should be used, False otherwise
    """
    return uuid in objathor_annotation


async def calc_metadata_objathor(file_path: Path, round_digits: int = 3) -> dict | str:
    """
    Calculate metadata using ObjaTHOR data if available, otherwise return failure.

    Args:
        file_path: Path to the asset file
        round_digits: Number of decimal places for rounding

    Returns:
        Metadata dictionary compatible with database schema, or "failure" if not available
    """
    uuid = file_path.stem
    _, dims, _ = get_glb_dimensions(file_path)

    # if objathor_annotation is None:  # TEMPDEAC?
    #     load_objathor_annotation(OBJATHOR_ANNO_JSON_PATH)
    if not objathor_annotation_available(uuid):
        return "failure"

    logger.debug(f"Using ObjaTHOR data for asset {uuid}")
    metadata = create_metadata_from_objathor(
        uuid, file_path, dims, objathor_annotation, round_digits
    )

    # Handle scaling if needed
    if metadata["misscaled"] and metadata["scaling_factor"] != -1:
        sf = metadata["scaling_factor"]
        scaled_model_path = file_path.with_stem(f"{uuid}_scaled")
        success = scale_glb_model(file_path, scaled_model_path, sf, backend="blender")
        if not success:
            return "failure"
        metadata["fs_path_rescaled"] = str(scaled_model_path)

        logger.debug(f"Asset [{uuid}] ObjaTHOR Correction Factor: {sf}")
        logger.debug(
            f"Asset [{uuid}] ObjaTHOR Output Dimensions: {[round(e * sf, 2) for e in dims]}"
        )

    return metadata
