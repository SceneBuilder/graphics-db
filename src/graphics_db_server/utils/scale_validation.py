import subprocess
import sys
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


def scale_glb_model_gltf_transform(input_path: str, output_path: str, scale_factor: float) -> bool:
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
        "--factor", str(scale_factor),
        input_path,
        output_path
    ]

    try:
        logger.info(f"Running command: {' '.join(command)}")
        
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )

        logger.info("GLB scaling complete!")
        if result.stdout:
            logger.debug(f"gltf-transform output: {result.stdout}")
        
        return True

    except FileNotFoundError:
        logger.error("'gltf-transform' command not found. Please install it with 'npm install -g @gltf-transform/cli'")
        return False

    except subprocess.CalledProcessError as e:
        logger.error(f"gltf-transform failed with exit code {e.returncode}")
        # if e.stderr:
        #     logger.error(f"Error details: {e.stderr}")
        logger.error(f"Error details: {e.stderr}")  # TEMP
        return False


def scale_glb_model_pyvista(input_path: str, output_path: str, scale_factor: float) -> bool:
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


def scale_glb_model(
    input_path: str, 
    output_path: str, 
    scale_factor: float, 
    backend: Literal["pyvista", "gltf_transform"] = "pyvista"
) -> bool:
    """
    Scales a GLB model using the specified backend.

    Args:
        input_path: The file path for the input GLB model.
        output_path: The file path for the scaled output GLB model.
        scale_factor: Uniform scaling factor for all axes.
        backend: The backend to use for scaling ("pyvista" or "gltf_transform").
    
    Returns:
        True if successful, False otherwise.
    """
    if backend == "pyvista":
        return scale_glb_model_pyvista(input_path, output_path, scale_factor)
    elif backend == "gltf_transform":
        return scale_glb_model_gltf_transform(input_path, output_path, scale_factor)
    else:
        logger.error(f"Unknown backend: {backend}. Supported backends: 'pyvista', 'gltf_transform'")
        return False
