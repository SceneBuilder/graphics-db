import math
from typing import Optional

import pyvista as pv

from graphics_db_server.logging import logger
from graphics_db_server.utils.rounding import safe_round


def calc_optimal_scaling_factor(
    original_dims: list[float, float, float], desired_dims: list[float, float, float]
) -> float:
    """
    Calculates the optimal scaling factor given original and desired dimensions.

    Args:
        original_dims: Original dimensions as [x, y, z] list
        desired_dims: Desired dimensions as [x, y, z] list

    Returns:
        float: The optimal scaling factor to apply
    """
    # Calculate scaling factors for each dimension
    scale_factors = [
        desired_dims[i] / original_dims[i] for i in range(3) if original_dims[i] != 0
    ]

    if not scale_factors:
        return 1.0

    # Use uniform scaling - take the geometric mean for balanced scaling
    geometric_mean = math.pow(math.prod(scale_factors), 1.0 / len(scale_factors))

    logger.debug(
        f"[tool] calc_optimal_scaling_factor(): {original_dims} → {desired_dims}; SF={safe_round(geometric_mean, 3)}"
    )

    return safe_round(geometric_mean, 3)


def get_glb_bounding_box(glb_path: str) -> tuple[bool, Optional[tuple], Optional[str]]:
    """
    Extract bounding box from a GLB file using PyVista.

    Args:
        glb_path: Path to the GLB file

    Returns:
        tuple: (success, bounds, error_message)
        bounds format: (xmin, xmax, ymin, ymax, zmin, zmax) or None if failed
    """
    try:
        plotter = pv.Plotter(off_screen=True)
        plotter.import_gltf(glb_path)

        bounds = plotter.renderer.ComputeVisiblePropBounds()
        plotter.close()

        if len(bounds) != 6:
            return False, None, "Could not compute bounding box"

        return True, bounds, None

    except Exception as e:
        return False, None, f"Error loading GLB file: {str(e)}"


def calc_dimensions_from_bounds(bounds: tuple) -> tuple[float, float, float]:
    """
    Calculate x, y, z dimensions from bounds tuple.

    Args:
        bounds: Bounding box as (xmin, xmax, ymin, ymax, zmin, zmax)

    Returns:
        tuple: (x_size, y_size, z_size)
    """
    x_size = abs(bounds[1] - bounds[0])  # xmax - xmin
    y_size = abs(bounds[3] - bounds[2])  # ymax - ymin
    z_size = abs(bounds[5] - bounds[4])  # zmax - zmin

    return x_size, y_size, z_size


def get_glb_dimensions(glb_path: str) -> tuple[bool, Optional[tuple], Optional[str]]:
    """
    Get GLB dimensions directly.

    Args:
        glb_path: Path to the GLB file

    Returns:
        tuple: (success, dimensions, error_message)
        dimensions format: (x_size, y_size, z_size) or None if failed
    """
    success, bounds, error = get_glb_bounding_box(glb_path)

    if not success:
        return False, None, error

    dimensions = calc_dimensions_from_bounds(bounds)
    return True, dimensions, None


def get_max_dimension(glb_path: str) -> tuple[bool, Optional[float], Optional[str]]:
    """
    Get the maximum dimension (edge length) of a GLB file.

    Args:
        glb_path: Path to the GLB file

    Returns:
        tuple: (success, max_dimension, error_message)
    """
    success, dimensions, error = get_glb_dimensions(glb_path)

    if not success:
        return False, None, error

    max_dimension = max(dimensions)
    return True, max_dimension, None
