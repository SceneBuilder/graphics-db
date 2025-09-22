import subprocess
from pathlib import Path
from typing import Literal, Optional

import pyvista as pv

from graphics_db_server.logging import logger
from graphics_db_server.utils.geometry import get_max_dimension


def check_asset_scale(
    glb_path: str, max_edge_length: float = 100.0
) -> tuple[bool, Optional[str]]:
    """
    Checks that a GLB asset has a reasonable scale by checking if the largest bounding box edge
    exceeds the specified threshold. This helps identify assets that may be in units other than meters.

    Args:
        glb_path: Path to the GLB file to validate
        max_edge_length: Maximum allowed edge length in meters (default: 100.0)

    Returns:
        tuple: (is_valid, reason) where is_valid is True if the asset passes validation,
               and reason is a string explaining why validation failed (if applicable)
    """
    success, max_edge, error = get_max_dimension(glb_path)

    if not success:
        return False, f"Error validating asset: {error}"

    if max_edge > max_edge_length:
        return (
            False,
            f"Asset too large: max edge is {max_edge:.2f}m (limit: {max_edge_length}m)",
        )

    return True, None


def validate_asset_scales(
    asset_paths: dict[str, str], max_edge_length: float = 100.0
) -> dict[str, bool]:
    """
    Validates the scale of downloaded GLB assets to reject those that are too large
    (likely in centimeters instead of meters).

    Args:
        asset_paths: A dictionary mapping asset UIDs to their .glb file paths.
        max_edge_length: Maximum allowed edge length in meters (default: 100.0)

    Returns:
        A dictionary mapping asset UIDs to validation results (True if valid, False if rejected).
    """
    validation_results = {}
    for uid, glb_path_str in asset_paths.items():
        glb_path = Path(glb_path_str).resolve()

        if not glb_path.exists():
            validation_results[uid] = False
            logger.warning(f"GLB file not found for asset {uid}: {glb_path}")
            continue

        is_valid, reason = check_asset_scale(str(glb_path), max_edge_length)
        validation_results[uid] = is_valid

        if not is_valid:
            logger.info(f"Rejecting asset {uid}: {reason}")
        else:
            logger.info(f"Asset {uid} passed scale validation")

    return validation_results


def scale_glb_model_gltf_transform(
    input_path: str, output_path: str, scale_factor: float
) -> bool:
    """
    Scales a GLB model using the gltf-transform CLI tool.

    Args:
        input_path: The file path for the input GLB model.
        output_path: The file path for the scaled output GLB model.
        scale_factor: Uniform scaling factor for all axes.

    Returns:
        True if successful, False otherwise.
    """
    if not Path(input_path).exists():
        logger.error(f"Input file not found at '{input_path}'")
        return False

    if isinstance(input_path, Path):
        input_path = str(input_path)
    if isinstance(output_path, Path):
        output_path = str(output_path)

    command = [
        "gltf-transform",
        "rescale",
        "--factor",
        str(scale_factor),
        input_path,
        output_path,
    ]

    try:
        logger.info(f"Running command: {' '.join(command)}")

        result = subprocess.run(command, check=True, capture_output=True, text=True)

        logger.info("GLB scaling complete!")
        if result.stdout:
            logger.debug(f"gltf-transform output: {result.stdout}")

        return True

    except FileNotFoundError:
        logger.error(
            "'gltf-transform' command not found. Please install it with 'npm install -g @gltf-transform/cli'"
        )
        return False

    except subprocess.CalledProcessError as e:
        logger.error(f"gltf-transform failed with exit code {e.returncode}")
        logger.error(f"Error details: {e.stderr}")
        return False


def scale_glb_model_pyvista(
    input_path: str, output_path: str, scale_factor: float
) -> bool:
    """
    Loads a GLB model, scales it uniformly, and saves it to a new file using PyVista.

    Args:
        input_path: The file path for the input GLB model.
        output_path: The file path for the scaled output GLB model.
        scale_factor: Uniform scaling factor for all axes.

    Returns:
        True if successful, False otherwise.
    """
    if not Path(input_path).exists():
        logger.error(f"Input file not found at '{input_path}'")
        return False

    logger.info(f"Loading model from: {input_path}")

    try:
        mesh = pv.read(input_path)
    except Exception as e:
        logger.error(f"Failed to read the model file: {e}")
        return False

    logger.info("Model loaded successfully.")
    logger.info(f"Original Bounds: {mesh.bounds}")

    if isinstance(mesh, pv.MultiBlock):
        for i, block in enumerate(mesh):
            if block is not None:
                mesh[i] = block.scale(scale_factor, inplace=False)
    else:
        mesh.scale(scale_factor, inplace=True)

    logger.info(f"Applied uniform scaling factor: {scale_factor}")
    logger.info(f"New Scaled Bounds: {mesh.bounds}")

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh)

    logger.info(f"Exporting scaled model to: {output_path}")
    try:
        plotter.export_gltf(output_path)
        logger.info("Export complete!")
        return True
    except Exception as e:
        logger.error(f"Failed to export the model file: {e}")
        return False


def cleanup_blender_memory():  # NOTE: doesn't seem to work
    """Periodically clear Blender memory for batch operations."""
    try:
        import bpy
        import gc
        
        # Clear scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        # Purge all orphaned data
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        
        # Force Python garbage collection
        gc.collect()
        
    except ImportError:
        pass


def scale_glb_model_blender(
    input_path: str, output_path: str, scale_factor: float
) -> bool:
    """
    Scales a GLB model using the 'bpy' Python module. This is a robust method
    that correctly applies transformations. Requires the 'bpy' module to be installed.

    Args:
        input_path: The file path for the input GLB model.
        output_path: The file path for the scaled output GLB model.
        scale_factor: Uniform scaling factor for all axes.

    Returns:
        True if successful, False otherwise.
    """
    try:
        import bpy
    except ImportError:
        logger.error(
            "The 'bpy' module is not installed. Please install it with 'pip install bpy'"
        )
        return False

    if not Path(input_path).exists():
        logger.error(f"Input file not found at '{input_path}'")
        return False

    try:
        logger.debug("Using 'bpy' module for scaling.")

        # Ensure we're in a clean state by clearing the scene
        if bpy.context.object and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

        # Import the GLB file
        logger.debug(f"Importing GLB: {input_path}")
        bpy.ops.import_scene.gltf(filepath=str(input_path))

        # Select all imported MESH objects to scale everything
        bpy.ops.object.select_all(action="DESELECT")
        imported_objects = [
            obj for obj in bpy.context.scene.objects if obj.type == "MESH"
        ]

        if not imported_objects:
            logger.error("No mesh objects found in the imported file.")
            return False

        for obj in imported_objects:
            obj.select_set(True)

        # Set an active object for context-sensitive operations
        bpy.context.view_layer.objects.active = imported_objects[0]

        # Set the 3D Cursor's location to the world center (0, 0, 0)
        bpy.context.scene.cursor.location = (0, 0, 0)
        # Set the pivot point to '3D Cursor'
        bpy.context.scene.tool_settings.transform_pivot_point = 'CURSOR'

        # Find the 3D Viewport area for context override
        area = next((a for a in bpy.context.screen.areas if a.type == 'VIEW_3D'), None)
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        
        if not area:
            logger.error("No 3D Viewport area found in current context")
            return False
        
        # logger.debug(f"Area: {area}")  # DEBUG

        logger.debug(f"Scaling selected objects by a factor of {scale_factor}.")
        
        # Use the modern Blender 4.0+ context override approach
        with bpy.context.temp_override(area=area, region=region):
            # # TEMP
            # bpy.context.scene.cursor.location = (0, 0, 0)
            # # Set the pivot point to '3D Cursor'
            # bpy.context.scene.tool_settings.transform_pivot_point = 'CURSOR'
            
            bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))

        # Apply the scale transformation to bake it into the mesh data
        logger.debug("Applying scale transformation.")
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # Export to GLB, ensuring only the modified objects are exported
        logger.debug(f"Exporting scaled GLB to: {output_path}")
        bpy.ops.export_scene.gltf(
            filepath=str(output_path), export_format="GLB", use_selection=True
        )
        
        # Clean up memory to prevent leaks
        cleanup_blender_memory()
        
        return True

    except Exception as e:
        import traceback

        logger.error(f"An error occurred during the Blender scaling process: {e}")
        logger.error(traceback.format_exc())
        return False


def scale_glb_model(
    input_path: str,
    output_path: str,
    scale_factor: float,
    backend: Literal["pyvista", "gltf_transform", "blender"] = "blender",
) -> bool:
    """
    Scales a GLB model using the specified backend.

    Args:
        input_path: The file path for the input GLB model.
        output_path: The file path for the scaled output GLB model.
        scale_factor: Uniform scaling factor for all axes.
        backend: The backend to use for scaling ("pyvista", "gltf_transform", or "blender").
                 "blender" is recommended for best results.

    Returns:
        True if successful, False otherwise.
    """
    if backend == "pyvista":
        return scale_glb_model_pyvista(input_path, output_path, scale_factor)
    elif backend == "gltf_transform":
        return scale_glb_model_gltf_transform(input_path, output_path, scale_factor)
    elif backend == "blender":
        return scale_glb_model_blender(input_path, output_path, scale_factor)
    else:
        logger.error(
            f"Unknown backend: {backend}. Supported backends: 'pyvista', 'gltf_transform', 'blender'"
        )
        return False
