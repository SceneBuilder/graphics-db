"""
Origin validation and correction module for 3D assets.

This module provides functionality to:
1. Classify the origin point placement of 3D mesh assets into categories:
   off-center, median, ground, contained, invalid, error.
2. Recentering GLB files by translating the origin to desired positions
   (ground or median) using the asset's AABB.

Designed for integration with the graphics-db pipeline, supporting Dask-based
parallel processing for large-scale datasets and Blender-based modifications.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import subprocess
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

        # OFF-CENTER: Origin is outside the bounding box (with tolerance)
        min_bounds = bounds[0] - tolerance
        max_bounds = bounds[1] + tolerance
        if not (origin[0] >= min_bounds[0] and origin[0] <= max_bounds[0] and
                origin[1] >= min_bounds[1] and origin[1] <= max_bounds[1] and
                origin[2] >= min_bounds[2] and origin[2] <= max_bounds[2]):
            return "off-center"
        # At this point, we know origin is inside the bounding box (with tolerance)

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


def recenter_glb_model(
    input_path: Path,
    output_path: Path,
    origin_type: Literal["ground", "median"],
    backend: str = "blender",
) -> bool:
    """
    Recenters a GLB file by translating all meshes so the origin aligns with
    the desired origin_type based on the asset's AABB.

    Args:
        input_path: Path to the input GLB file.
        output_path: Path to save the recentered GLB file.
        origin_type: "ground" (center XZ, Y at min/bottom) or "median" (full centroid).
        backend: Currently only "blender" is supported.

    Returns:
        bool: True if successful, False otherwise.

    Note:
        - Uses Blender Python API via subprocess for isolation.
        - Assumes Blender is installed and accessible in PATH.
        - Clears scene and imports/exports in an isolated manner.
    """
    if backend != "blender":
        logger.error(f"Unsupported backend: {backend}")
        return False

    # Blender script to execute
    blender_script = f"""
import bpy
import bmesh
from mathutils import Vector

# Clear existing scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import the GLB
bpy.ops.import_scene.gltf(filepath=r"{input_path}")

# Select all mesh objects (ignore empties, cameras, etc.)
meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if not meshes:
    print("No mesh objects found")
    sys.exit(1)

bpy.ops.object.select_all(action='DESELECT')
for mesh in meshes:
    mesh.select_set(True)
bpy.context.view_layer.objects.active = meshes[0] if meshes else None

# Update scene
bpy.context.view_layer.update()

# Compute AABB in world space
min_coords = Vector((float('inf'), float('inf'), float('inf')))
max_coords = Vector((float('-inf'), float('-inf'), float('-inf')))

for obj in meshes:
    obj_matrix = obj.matrix_world
    for corner in obj.bound_box:
        world_corner = obj_matrix @ Vector(corner)
        min_coords.x = min(min_coords.x, world_corner.x)
        min_coords.y = min(min_coords.y, world_corner.y)
        min_coords.z = min(min_coords.z, world_corner.z)
        max_coords.x = max(max_coords.x, world_corner.x)
        max_coords.y = max(max_coords.y, world_corner.y)
        max_coords.z = max(max_coords.z, world_corner.z)

center_x = (min_coords.x + max_coords.x) / 2
center_y = (min_coords.y + max_coords.y) / 2
center_z = (min_coords.z + max_coords.z) / 2

# Determine translation
if origin_type == "median":
    trans_x, trans_y, trans_z = -center_x, -center_y, -center_z
elif origin_type == "ground":
    if on_floor:
        trans_y = -min_coords.y  # Ground to Y=0
    else:
        trans_y = -center_y
    trans_x, trans_z = -center_x, -center_z
else:
    print(f"Invalid origin_type: {origin_type}")
    sys.exit(1)

# Apply translation to all selected objects
bpy.ops.transform.translate(value=(trans_x, trans_y, trans_z))

# Export as GLB
bpy.ops.export_scene.gltf(
    filepath=r"{output_path}",
    export_format='GLB',
    export_apply=True,
    export_selected=True
)

print("Recentering successful")
"""

    try:
        # Run Blender in background with the script
        cmd = [
            "blender",
            "--background",
            "--python-expr",
            blender_script,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=input_path.parent,
        )
        if result.returncode == 0:
            logger.debug(f"Recentered {input_path} to {output_path} using {origin_type}")
            return True
        else:
            logger.error(f"Blender error recentering {input_path}: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Blender subprocess failed for {input_path}: {e}")
        return False
    except FileNotFoundError:
        logger.error("Blender not found in PATH")
        return False
