"""
Origin type classification module for 3D assets.

This module provides functionality to classify the origin point placement
of 3D mesh assets into categories: off-center, median, ground, contained.

Designed to work with the Dask-based parallel processing pipeline for large-scale
asset datasets (millions of assets).
"""

from pathlib import Path
from typing import Literal

import numpy as np
import trimesh

from graphics_db_server.logging import logger


TOLERANCE_RATIO = 0.1

OriginType = Literal["off-center", "median", "ground", "contained", "invalid", "error"]


def classify_origin_type(file_path: Path | str) -> OriginType:
    """
    Analyzes a single mesh file to classify its origin type.

    This function is designed to be CPU-bound and serializable by Pickle for distributed computing.

    Args:
        file_path: The absolute path to the mesh file (.glb, .gltf, etc.)

    Returns:
        OriginType: One of:
            - 'off-center': Origin is outside the bounding box
            - 'median': Origin is at the geometric center (centroid)
            - 'ground': Origin is centered on XZ plane, grounded on Y (GLTF uses Y-up)
            - 'contained': Origin inside bbox but not median/ground
            - 'invalid': No geometry or invalid mesh
            - 'error': Processing error occurred

    Classification Logic:
        - Check if origin is outside bounding box → 'off-center'
        - Check if origin is at centroid (within tolerance) → 'median'
        - Check if origin is XZ-centered and Y-grounded → 'ground'
        - Otherwise → 'contained'

    Note:
        - Assumes Y-up coordinate system (GLTF/GLB standard)
        - Uses trimesh.Scene for proper GLB hierarchy handling
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    try:
        # Load as scene - critical for GLB files with hierarchies
        # trimesh.load with force='scene' handles transformations properly
        scene = trimesh.load(str(file_path), force="scene")

        # Validate that scene has geometry
        if not scene.geometry:
            return "invalid"

        # Get scene-level bounding properties
        # scene.bounds: (2, 3) array with [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        bounds = scene.bounds
        extents = scene.extents  # [x_size, y_size, z_size]
        box_center = scene.centroid  # Geometric center

        # Define tolerance for floating point comparisons
        max_extent = max(extents)
        tolerance = max_extent * TOLERANCE_RATIO

        origin = np.array([0.0, 0.0, 0.0])

        # Classification logic (order matters!)

        # OFF-CENTER: Origin is outside the bounding box
        if not scene.bounding_box.contains([origin]):
            return "off-center"
        # At this point, we know origin is inside the bounding box

        # MEDIAN: Origin is at the geometric center of the bbox
        if np.allclose(origin, box_center, atol=tolerance):
            return "median"

        # GROUND: Origin is centered on XZ plane and sits on bottom Y plane
        # This is the typical setup for "placeable" objects in 3D engines
        # GLTF uses Y-up coordinate system (right-handed: +Y up, +Z forward, -X right)
        is_xz_centered = np.allclose([origin[0], origin[2]], [box_center[0], box_center[2]], atol=tolerance)
        is_y_grounded = abs(bounds[0][1] - origin[1]) < tolerance

        if is_xz_centered and is_y_grounded:
            return "ground"

        # CONTAINED: Origin is inside bbox but not median/ground
        # This often indicates non-standard origin placement
        return "contained"

    except Exception as e:
        # Log the error for debugging but return error type
        logger.error(f"Error classifying origin for {file_path}: {str(e)}")
        return "error"


def classify_origin_type_with_uuid(uuid: str, file_path: str) -> tuple[str, OriginType]:
    """
    Wrapper function that includes UUID in return value for easier database mapping.

    This is the function that should be used with Dask for parallel processing,
    as it returns both the UUID and the classification result.

    Args:
        uuid: Asset UUID (typically the stem of the filename)
        file_path: Absolute path to the mesh file

    Returns:
        tuple: (uuid, origin_type)

    Example:
        >>> classify_origin_type_with_uuid("abc123", "/path/to/abc123.glb")
        ("abc123", "ground")
    """
    origin_type = classify_origin_type(file_path)
    return uuid, origin_type


# NOTE: integration testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python setup_extra_index_origin.py <path_to_glb_file>")
        sys.exit(1)

    test_path = Path(sys.argv[1])
    if not test_path.exists():
        print(f"Error: File not found at {test_path}")
        sys.exit(1)

    result = classify_origin_type(test_path)
    print(f"Origin type for {test_path.name}: {result}")
