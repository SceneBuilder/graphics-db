# IDEA: Just have VLM output ideal size in structured text, and then do downstream calculation
#       (classification between cm, mm, 0.1x, arbitrary) as well as optimal scaling calculation
#       with a traditional pipeline. This may allow much faster and much more efficient execution.
#       Plus, concatenating thumbnail images can be a good idea — depending on accuracy & cost.

"""
This script sets up an additional SQLite-based DB for metadata interaction,
such as computing 3D bounding box, analyzing re-scaling parameters, thumbnails,
and filesystem locations for fast disk lookups.

This is meant as a stable staging area for all relevant asset (for now, object) annotations,
for offline ingestion from external data sources or VLM-based analysis in-house.

NOTE: Currently, this script is specifically written for Objaverse dataset.
TODO: Modularize to suit different asset data sources (different asset datasets).
TODO: add generic name/description to DB schema for easier debugging / other potential uses
"""

import argparse
import asyncio
import datetime
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Literal

import dask
import logfire
from dask.distributed import Client, LocalCluster, as_completed
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import RunContext
from tqdm import tqdm

from graphics_db_server.core.config import (
    EXTRA_INDEX_DB_FILE,
    LOCAL_FS_PATHS,
    THUMBNAIL_RESOLUTION,
    VLM_MODEL_NAME,
    VLM_PROVIDER_BASE_URL,
)
from graphics_db_server.core.config import LOGFIRE_SERVICE_NAME
from graphics_db_server.logging import logger
from graphics_db_server.tools.read_file import read_media_file
from graphics_db_server.utils.geometry import (
    calc_optimal_scaling_factor,
    get_glb_dimensions,
)
from graphics_db_server.utils.pai import transform_paths_to_binary
from graphics_db_server.utils.rounding import safe_round
from graphics_db_server.utils.scale_validation import scale_glb_model
from graphics_db_server.utils.thumbnail import generate_thumbnail_from_glb
from graphics_db_server.scripts.setup_extra_index_objathor import (
    calc_metadata_objathor,
    extract_scale_analysis_from_objathor,
    load_objathor_annotation,
    objathor_annotation_available,
)
from graphics_db_server.scripts.setup_extra_index_origin import (
    classify_origin_type_with_uuid,
)

# Configuration
THUMBNAIL_DIR = Path("/mnt/4TBStorageA/ycho358/graphics_db_thumbnails")
METADATA_VERSION = 1  # NOTE: increment with logic changes
BATCH_SIZE = 100  # for periodic DB commits
MAX_CONCURRENT = 100  # for VLM calls
ROUND_DIGITS = 3

DEBUG = True
# DEBUG = False

logfire.configure(service_name=LOGFIRE_SERVICE_NAME)
logfire.instrument_pydantic_ai()


def setup_database():
    """
    Sets up SQLite database. Adds columns for metadata if not exists.
    """
    conn = sqlite3.connect(EXTRA_INDEX_DB_FILE)
    cursor = conn.cursor()

    logger.info("Setting up database schema...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assets (
            uuid TEXT PRIMARY KEY,
            file_path TEXT NOT NULL
        )
    """)

    # Add columns for metadata
    columns = {
        "misscaled": "INTEGER DEFAULT 0",
        "misscaling_type": "TEXT",
        "dims_x": "REAL",
        "dims_y": "REAL",
        "dims_z": "REAL",
        "dims_xr": "REAL",
        "dims_yr": "REAL",
        "dims_zr": "REAL",
        "scaling_factor": "REAL",
        "metadata_version": "INTEGER",
        "last_updated": "TEXT",
        "thumbnail_paths": "TEXT",
        "fs_path": "TEXT",
        "fs_path_rescaled": "TEXT",
        "rescaled_by": "TEXT",
        "origin_type": "TEXT",  # off-center, median, ground, contained
    }

    cursor.execute("PRAGMA table_info(assets)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    for col_name, col_type in columns.items():
        if col_name not in existing_columns:
            logger.debug(f"Adding column: {col_name}")
            cursor.execute(f"ALTER TABLE assets ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()
    logger.info(f"Database {EXTRA_INDEX_DB_FILE} is set up and ready.")


def setup_index(data_dir: Path):
    """
    Scans the target directory for .glb files and populates the SQLite database.

    Args:
        data_directory (Path): The root directory of the 3D asset dataset.
    """
    if not data_dir.is_dir():
        logger.error(f"Directory not found at {data_dir}")
        sys.exit(1)

    logger.info(f"Starting to index .glb files in {data_dir}...")

    conn = sqlite3.connect(EXTRA_INDEX_DB_FILE)
    cursor = conn.cursor()

    logger.debug("Discovering all .glb files first to show progress...")
    asset_files = [p for p in data_dir.rglob("*.glb") if not p.stem.endswith("_scaled")]

    if not asset_files:
        logger.warning("No .glb files found to index.")
        conn.close()
        return

    logger.info(f"Found {len(asset_files)} files. Indexing...")
    asset_data_generator = ((p.stem, str(p.resolve())) for p in asset_files)

    with tqdm(total=len(asset_files), desc="Indexing assets") as pbar:
        while True:
            batch = [
                item for _, item in zip(range(BATCH_SIZE * 10), asset_data_generator)
            ]
            if not batch:
                break
            cursor.executemany(
                "INSERT OR IGNORE INTO ASSETS (uuid, file_path) VALUES (?, ?)", batch
            )
            conn.commit()
            pbar.update(len(batch))
    conn.close()
    logger.info("File indexing complete.")


def generate_thumbnails(uuid: str, path: str | Path, thumb_dir: Path) -> list[str]:
    """
    Creates thumbnail images on disk and returns their paths for a given asset.

    Args:
        uuid (str): The asset's unique identifier.
        thumb_dir (Path): The directory where thumbnails should be saved.

    Returns:
        list[str]: A list of absolute paths to the generated thumbnails
    """
    if isinstance(path, str):
        path = Path(path)

    directions = ["top", "front", "right", "isometric"]
    thumbnail_paths = []

    for direction in directions:
        fn = THUMBNAIL_DIR / f"{uuid}_thumbnail_{direction}.png"
        generate_thumbnail_from_glb(
            path,
            fn,
            THUMBNAIL_RESOLUTION,
            view_direction=direction,
        )
        thumbnail_paths.append(fn)

    return thumbnail_paths


class ScaleAnalysisInput(BaseModel):
    thumbnail_paths: list[str]
    question: str | None = None
    prompt: str | None = None


class ScaleAnalysisOutput(BaseModel):
    object_description: str
    ideal_dimensions: str
    reasoning: str
    misscaled: bool
    misscaling_type: Literal["mm", "cm", "arbitrary", "N/A"]
    correction_factor: float | None = None


# model = GoogleModel("gemini-2.5-flash")
model = OpenAIChatModel("gpt-5-mini")
# model = OpenAIChatModel("gpt-5-nano")  # NOTE: not good enough.
# client = AsyncOpenAI(base_url=VLM_PROVIDER_BASE_URL, api_key="empty")
# model = OpenAIChatModel(
#     VLM_MODEL_NAME,
#     provider=OpenAIProvider(base_url=VLM_PROVIDER_BASE_URL, api_key="EMPTY"),
# )
system_prompt = (
    "You are an AI model that is in charge of the quality control of "
    "3d assets that sometimes have an incorrect scale.\n"
    "Your job is to check whether or not a given 3D asset has realistic dimensions that "
    "are consistent with the typical sizes of respective objects in the real world.\n"
    "For example, it is unreasonable for a desk lamp to be [4.0, 3.0, 4.0] m, since in the real world, "
    "a desk lamp would normally not be larger than ~[0.3, 0.5, 0.3] m."
    "\nYou will be given paths to thumbnail images of the object, as well as the 3D dimensions. "
    "(If given image paths, please ALWAYS see all of the images by using `read_media_file(filepath: str)`.) "
    "First, please output a description of what object you are looking at.\n"
    "Then, please output what you think as the ideal dimensions of the object.\n"
    "Note: A good rule to follow is to think of what the most typical example of a real-world object that belongs to the, "
    "same category as the asset, such as a sofa, and then to think of what the physicall plausible dimensions of a sofa would be.\n"
    "Now, please use the `calc_optimal_scaling_factor(original_dims=[float, float, float], desired_dims=[float, float, float]) -> float` tool.\n"
    "Then, please output your reasoning and analysis in comparing the original dimensions of the object to the ideal dimensions.\n"
    "Then, please classify whether the object is misscaled and the type of misccaling (mm, cm, 10x, or arbitrary)."
    "Finally, if the object is misscaled, please output a 'correction factor'—a scaling factor that must be applied to restore the asset to a correct scale.\n"
    "A common cause of error is the use of non-meter length unit such as mm or cm, "
    "despite GLTF specifications that mandate meter units. "
    "In this case, it is easy to deduce the correct scale—by simply dividing the "
    "dimensions by 100 or 1,000 respectively. "
    "There are also assets that are arbitrarily misscaled. "
    # "Be very careful to assure that the final dimensions have physically plausible sizes: not too small, and not too large.\n"
    "Please assure that the final dimensions have physically plausible sizes: not too small, and not too large.\n"
    "Note: the correction factor is to be *multiplied* to the asset, so if you want to size down, provide a number smaller than 1, and vice versa. "
)
# NOTE can include: output response example
# NOTE can run through claude's prompt optimizer
scale_analysis_agent = Agent(
    model,
    system_prompt=system_prompt,
    output_type=ScaleAnalysisOutput,
    tools=[calc_optimal_scaling_factor],
)


@scale_analysis_agent.system_prompt
async def add_inputs(ctx: RunContext[ScaleAnalysisInput]) -> str:
    if not hasattr(ctx.deps, "thumbnail_paths"):
        # return None  # ? Problematic? → Yes.
        return ""
    return f"Paths to thumbnails:\n {ctx.deps.thumbnail_paths}"


def calc_metadata(file_path: Path, thumbnail_paths: list[Path] | None = None) -> dict:
    """ """
    uuid = file_path.stem
    _, dims, _ = get_glb_dimensions(file_path)

    user_prompt = (
        f"**Asset {uuid}**:",
        f"Dimensions: {[round(e, 2) for e in dims]} m",
        "Please analyze this 3D asset.",
    )
    user_prompt = "\n".join(user_prompt)
    extra_info = (
        "\nExtra information:",
        # f"- Larger than than 100 m (e.g., originally in cm): {max(dims) > 100}",
        f"- Larger than than 100 m (e.g., potentially in cm): {max(dims) > 100}",
        f"- Dimensions if scaled by 0.01: {[round(e / 100, 2) for e in dims]} m",
        # f"- Larger than 1000 m (e.g., originally in mm): {max(dims) > 1000}",
        f"- Larger than 1000 m (e.g., potentially in mm): {max(dims) > 1000}",
        f"- Dimensions if scaled by 0.001: {[round(e / 1000, 2) for e in dims]} m",
    )
    extra_info = "\n".join(extra_info)
    # question = "Are you able to see the thumbnail images?"  # DEBUG
    thumbnail_contents: list[BinaryContent] = transform_paths_to_binary(thumbnail_paths)
    inputs = thumbnail_paths  # TEMP
    if DEBUG:
        logger.debug(f"Asset [{uuid}] Inputs: {inputs}")
        logger.debug(f"Asset [{uuid}] \nUser Prompt: {user_prompt + extra_info}")

    response = scale_analysis_agent.run_sync(
        thumbnail_contents + [user_prompt + extra_info]
    )
    output: ScaleAnalysisOutput = response.output

    if DEBUG:
        logger.debug(f"Asset [{uuid}] Object Description: {output.object_description}")
        logger.debug(f"Asset [{uuid}] Ideal Dimensions: {output.reasoning}")
        logger.debug(f"Asset [{uuid}] Reasoning: {output.reasoning}")
        logger.debug(f"Asset [{uuid}] Misscaled: {output.misscaled}")
        logger.debug(f"Asset [{uuid}] Misscaling Type: {output.misscaling_type}")

    sf = output.correction_factor  # scaling factor
    if sf is not None and not math.isclose(sf, 1, abs_tol=0.1):
        scaled_model_path = file_path.with_stem(f"{uuid}_scaled")
        success = scale_glb_model(file_path, scaled_model_path, sf, backend="blender")
        if not success:
            return "failure"

        if DEBUG:
            logger.debug(f"Asset [{uuid}] Output Correction Factor: {sf}")
            logger.debug(
                f"Asset [{uuid}] Output Dimensions: {[round(e * sf, 2) for e in dims]}"
            )
    elif sf is not None and math.isclose(sf, 1, abs_tol=0.1):
        # NOTE: Even though the model thinks the object is mis-scaled, the correction factor
        #       is not significant enough, so we ignore it and mark the object well-scaled.
        output.misscaled = False
        output.misscaling_type = "N/A"

    if sf is None and output.misscaled:
        return "failure"
        # NOTE: This might be cases where dims are zeros, e.g., due to being non-surface photogrammetry

    return {
        "misscaled": 1 if output.misscaled else 0,
        "misscaling_type": output.misscaling_type,
        "dims_x": safe_round(dims[0], ROUND_DIGITS),
        "dims_y": safe_round(dims[1], ROUND_DIGITS),
        "dims_z": safe_round(dims[2], ROUND_DIGITS),
        "dims_xr": safe_round(dims[0] * sf, ROUND_DIGITS) if output.misscaled else -1,
        "dims_yr": safe_round(dims[1] * sf, ROUND_DIGITS) if output.misscaled else -1,
        "dims_zr": safe_round(dims[2] * sf, ROUND_DIGITS) if output.misscaled else -1,
        "scaling_factor": safe_round(sf, ROUND_DIGITS) if output.misscaled else -1,
        "fs_path": str(file_path),
        "fs_path_rescaled": str(scaled_model_path) if output.misscaled else None,
        "rescaled_by": "graphics-db",
    }


async def calc_metadata_async(
    file_path: Path,
    thumbnail_paths: list[Path] | None = None,
    strategy: str = "vlm_only",
) -> dict:
    """Async version of calc_metadata function."""
    uuid = file_path.stem

    # Handle different annotation strategies
    if strategy == "external_only":
        if objathor_annotation_available(uuid):
            metadata = await calc_metadata_objathor(file_path, ROUND_DIGITS)
            if metadata != "failure":
                return metadata
        return "failure"  # No external data available for external_only strategy
    elif strategy == "prefer_external":
        if objathor_annotation_available(uuid):
            metadata = await calc_metadata_objathor(file_path, ROUND_DIGITS)
            if metadata != "failure":
                return metadata
        # Fall back to VLM analysis if external data not available
    elif strategy == "vlm_only":
        pass  # Skip external annotation, proceed directly to VLM analysis

    _, dims, _ = get_glb_dimensions(file_path)

    user_prompt = (
        f"**Asset {uuid}**:",
        f"Dimensions: {[round(e, 2) for e in dims]} m",
        "Please analyze this 3D asset.",
    )
    user_prompt = "\n".join(user_prompt)
    extra_info = (
        "\nExtra information:",
        # f"- Larger than than 100 m (e.g., originally in cm): {max(dims) > 100}",
        f"- Larger than than 100 m (e.g., potentially in cm): {max(dims) > 100}",
        f"- Dimensions if scaled by 0.01: {[round(e / 100, 2) for e in dims]} m",
        # f"- Larger than 1000 m (e.g., originally in mm): {max(dims) > 1000}",
        f"- Larger than 1000 m (e.g., potentially in mm): {max(dims) > 1000}",
        f"- Dimensions if scaled by 0.001: {[round(e / 1000, 2) for e in dims]} m",
    )
    extra_info = "\n".join(extra_info)
    # question = "Are you able to see the thumbnail images?"  # DEBUG
    thumbnail_contents: list[BinaryContent] = transform_paths_to_binary(thumbnail_paths)
    inputs = thumbnail_paths  # TEMP
    if DEBUG:
        logger.debug(f"Asset [{uuid}] Inputs: {inputs}")
        logger.debug(f"Asset [{uuid}] \nUser Prompt: {user_prompt + extra_info}")

    response = await scale_analysis_agent.run(
        thumbnail_contents + [user_prompt + extra_info]
    )
    output: ScaleAnalysisOutput = response.output

    if DEBUG:
        logger.debug(f"Asset [{uuid}] Object Description: {output.object_description}")
        logger.debug(f"Asset [{uuid}] Ideal Dimensions: {output.reasoning}")
        logger.debug(f"Asset [{uuid}] Reasoning: {output.reasoning}")
        logger.debug(f"Asset [{uuid}] Misscaled: {output.misscaled}")
        logger.debug(f"Asset [{uuid}] Misscaling Type: {output.misscaling_type}")

    sf = output.correction_factor  # scaling factor
    scaled_model_path = None
    if sf is not None and not math.isclose(sf, 1, abs_tol=0.1):
        scaled_model_path = file_path.with_stem(f"{uuid}_scaled")
        success = scale_glb_model(file_path, scaled_model_path, sf, backend="blender")
        if not success:
            return "failure"

        if DEBUG:
            logger.debug(f"Asset [{uuid}] Output Correction Factor: {sf}")
            logger.debug(
                f"Asset [{uuid}] Output Dimensions: {[round(e * sf, 2) for e in dims]}"
            )
    elif sf is not None and math.isclose(sf, 1, abs_tol=0.1):
        # NOTE: Even though the model thinks the object is mis-scaled, the correction factor
        #       is not significant enough, so we ignore it and mark the object well-scaled.
        output.misscaled = False
        output.misscaling_type = "N/A"

    if sf is None and output.misscaled:
        return "failure"
        # NOTE: This might be cases where dims are zeros, e.g., due to being non-surface photogrammetry

    return {
        "misscaled": 1 if output.misscaled else 0,
        "misscaling_type": output.misscaling_type,
        "dims_x": safe_round(dims[0], ROUND_DIGITS),
        "dims_y": safe_round(dims[1], ROUND_DIGITS),
        "dims_z": safe_round(dims[2], ROUND_DIGITS),
        "dims_xr": safe_round(dims[0] * sf, ROUND_DIGITS) if output.misscaled else -1,
        "dims_yr": safe_round(dims[1] * sf, ROUND_DIGITS) if output.misscaled else -1,
        "dims_zr": safe_round(dims[2] * sf, ROUND_DIGITS) if output.misscaled else -1,
        "scaling_factor": safe_round(sf, ROUND_DIGITS) if output.misscaled else -1,
        "fs_path": str(file_path),
        "fs_path_rescaled": str(scaled_model_path) if output.misscaled else None,
        "rescaled_by": "graphics-db",
    }


def reset_metadata():
    """
    Clear all metadata.
    """
    conn = sqlite3.connect(EXTRA_INDEX_DB_FILE)
    cursor = conn.cursor()

    logger.info("Resetting all existing metadata")
    cursor.execute("""
        UPDATE assets SET
            misscaled = NULL,
            misscaling_type = NULL,
            dims_x = NULL,
            dims_y = NULL,
            dims_z = NULL,
            dims_xr = NULL,
            dims_yr = NULL,
            dims_zr = NULL,
            scaling_factor = NULL,
            metadata_version = NULL,
            last_updated = NULL,
            thumbnail_paths = NULL,
            fs_path = NULL,
            fs_path_rescaled = NULL,
            rescaled_by = NULL,
            origin_type = NULL
    """)
    conn.commit()
    conn.close()
    logger.info("Metadata has been reset.")


def compute_metadata(
    version: int, max_concurrent: int = MAX_CONCURRENT, strategy: str = "vlm_only"
):
    """
    Computes and updates metadata for assets that are out of date using async processing.

    Args:
        version (int): The version of the metadata logic to apply.
        max_concurrent (int): Maximum number of concurrent LLM API calls.
        strategy (str): Annotation strategy - 'vlm_only', 'prefer_external', or 'external_only'.
    """
    load_objathor_annotation()
    asyncio.run(_compute_metadata_async(version, max_concurrent, strategy))


async def _compute_metadata_async(
    version: int, max_concurrent: int = MAX_CONCURRENT, strategy: str = "vlm_only"
):
    """
    Internal async function that performs the actual metadata computation.

    Args:
        version (int): The version of the metadata logic to apply.
        max_concurrent (int): Maximum number of concurrent LLM API calls.
        strategy (str): Annotation strategy - 'vlm_only', 'prefer_external', or 'external_only'.
    """
    conn = sqlite3.connect(EXTRA_INDEX_DB_FILE)
    cursor = conn.cursor()

    query = "SELECT uuid, file_path FROM assets WHERE metadata_version IS NULL OR metadata_version < ?"
    params = (version,)
    if LIMIT:
        query += " LIMIT ?"
        params += (LIMIT,)
    target_assets = cursor.execute(query, params).fetchall()

    if OBJATHOR_ONLY:
        target_assets = [
            (uuid, path_str)
            for uuid, path_str in target_assets
            if objathor_annotation_available(uuid)
        ]
        logger.info(f"Filtered to {len(target_assets)} ObjaTHOR assets for processing.")

    if not target_assets:
        logger.warning(
            f"All assets are already up to date with metadata version {version}."
        )
        conn.close()
        return
    else:
        logger.info(f"Found {len(target_assets)} assets requiring metadata update.")

    # Ensure that thumbnail directory exists
    THUMBNAIL_DIR.mkdir(exist_ok=True)
    conn.close()

    # Create semaphore to limit concurrent LLM calls
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_asset(uuid: str, path_str: str) -> tuple[str, dict | str]:
        """Process a single asset with concurrency control."""
        async with semaphore:
            file_path = Path(path_str)
            if not file_path.exists():
                logger.warning(f"SKIPPING missing file: {uuid}")
                return uuid, "missing_file"

            try:
                # Only generate thumbnails if strategy requires VLM analysis
                thumbnail_paths = None
                if strategy in ["vlm_only", "prefer_external"]:
                    thumbnail_paths = generate_thumbnails(uuid, path_str, THUMBNAIL_DIR)
                    thumbnail_paths = [str(path) for path in thumbnail_paths]  # TEMP

                metadata = await calc_metadata_async(
                    file_path,
                    thumbnail_paths=thumbnail_paths,
                    strategy=strategy,
                )

                if metadata == "failure":
                    logger.warning(f"{uuid}: metadata generation failure.")
                    return uuid, "failure"

                # Add additional metadata fields
                metadata["thumbnail_paths"] = json.dumps(thumbnail_paths)
                metadata["metadata_version"] = version
                metadata["last_updated"] = datetime.datetime.now().isoformat()
                metadata["rescaled_by"] = "graphics-db"

                return uuid, metadata
            except Exception as e:
                logger.error(f"Error processing asset {uuid}: {e}")
                return uuid, "error"

    # Create tasks for all assets
    tasks = [process_single_asset(uuid, path_str) for uuid, path_str in target_assets]

    # Handle database updates with progress bar
    conn = sqlite3.connect(EXTRA_INDEX_DB_FILE)
    cursor = conn.cursor()

    successful_updates = 0
    failed_updates = 0

    def update_database(batch_results):
        nonlocal successful_updates, failed_updates

        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                failed_updates += 1
                continue

            uuid, metadata = result

            if metadata in ["failure", "missing_file", "error"]:
                failed_updates += 1
                continue

            # Update database
            try:
                update_query = f"UPDATE assets SET {', '.join(f'{k} = ?' for k in metadata)} WHERE uuid = ?"
                values = list(metadata.values()) + [uuid]
                cursor.execute(update_query, values)
                successful_updates += 1

            except Exception as e:
                logger.error(f"Database update failed for {uuid}: {e}")
                failed_updates += 1

        conn.commit()
        logger.debug(f"Committed batch of {BATCH_SIZE} updates")

    # Process tasks concurrently w/ progress bar and database updates
    with tqdm(total=len(tasks), desc=f"Computing metadata (v{version})") as pbar:
        for i in range(0, len(tasks), BATCH_SIZE):
            batch_tasks = tasks[i : i + BATCH_SIZE]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            update_database(batch_results)
            pbar.update(len(batch_tasks))

    # Commit any remaining updates & close
    conn.commit()
    conn.close()

    logger.info(
        f"Metadata computation complete. Success: {successful_updates}, Failed: {failed_updates}"
    )


def compute_origin_types_dask(version: int):
    """
    Compute origin types for all assets using Dask for parallel processing.

    This is optimized for large-scale processing (millions of assets) using
    distributed computing with Dask.

    Args:
        version: Metadata version number (same as used for scale analysis).
                 Assets with origin_type=NULL or metadata_version < version will be processed.
    """
    logger.info("Starting origin type analysis with Dask...")

    # Set up Dask cluster
    cluster = LocalCluster(
        threads_per_worker=1,  # for CPU-bound tasks
    )
    client = Client(cluster)
    logger.info(f"Dask dashboard available at: {client.dashboard_link}")

    # Query assets that need origin analysis
    conn = sqlite3.connect(EXTRA_INDEX_DB_FILE)
    cursor = conn.cursor()

    query = """
        SELECT uuid, file_path FROM assets
        WHERE origin_type IS NULL OR metadata_version IS NULL OR metadata_version < ?
    """
    params = (version,)
    if LIMIT:
        query += " LIMIT ?"
        params += (LIMIT,)
    target_assets = cursor.execute(query, params).fetchall()
    conn.close()

    if OBJATHOR_ONLY:
        target_assets = [(uuid, path_str) for uuid, path_str in target_assets if objathor_annotation_available(uuid)]
        logger.info(f"Filtered to {len(target_assets)} ObjaTHOR assets for origin analysis.")

    if not target_assets:
        logger.info(f"All assets already have origin analysis v{version}")
        client.close()
        cluster.close()
        return

    logger.info(f"Found {len(target_assets)} assets requiring origin analysis")

    # Submit tasks using client.map for memory efficiency
    logger.info("Submitting origin type analysis tasks to Dask...")
    futures = client.map(
        classify_origin_type_with_uuid,
        [uuid for uuid, _ in target_assets],
        [fp for _, fp in target_assets],
    )

    # Process results as they complete to avoid holding them all in memory
    logger.info("Processing results...")
    conn = sqlite3.connect(EXTRA_INDEX_DB_FILE)
    cursor = conn.cursor()

    successful = 0
    errors = 0
    update_data = []
    timestamp = datetime.datetime.now().isoformat()

    with tqdm(
        as_completed(futures, with_results=True),
        total=len(target_assets),
        desc="Analyzing origins",
    ) as pbar:
        for future, result in pbar:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                errors += 1
                continue

            uuid, origin_type = result
            if origin_type == "error":
                errors += 1
                continue

            # Append data for batch update
            update_data.append((origin_type, version, timestamp, uuid))
            successful += 1

            # Commit to database in batches
            if len(update_data) >= BATCH_SIZE:
                cursor.executemany(
                    "UPDATE assets SET origin_type = ?, metadata_version = ?, last_updated = ? WHERE uuid = ?",
                    update_data,
                )
                conn.commit()
                update_data = []

    # Commit any remaining final batch
    if update_data:
        cursor.executemany(
            "UPDATE assets SET origin_type = ?, metadata_version = ?, last_updated = ? WHERE uuid = ?",
            update_data,
        )
        conn.commit()

    conn.close()
    client.close()
    cluster.close()

    logger.info(f"Origin analysis complete. Success: {successful}, Errors: {errors}")


def rescale_asset_worker(uuid: str, file_path_str: str, scaling_factor: float) -> tuple[str, str, float]:
    """
    A simple, synchronous worker function that rescales a single asset.
    This function will be distributed by Dask.
    Returns: (uuid, path_to_scaled_file, actual_scaling_factor)
    """
    try:
        file_path = Path(file_path_str)
        scaled_model_path = file_path.with_stem(f"{uuid}_scaled")
        success = scale_glb_model(file_path, scaled_model_path, scaling_factor, backend="blender")
        
        if success:
            return uuid, str(scaled_model_path), scaling_factor
        else:
            logger.warning(f"Scaling failed for {uuid}")
            return uuid, "failure", -1.0
    except Exception as e:
        logger.error(f"Exception during scaling of {uuid}: {e}")
        return uuid, "failure", -1.0


def perform_rescaling_dask(version: int):
    """
    Computes rescaling for assets using offline ObjaTHOR annotations and Dask for parallel processing.
    """
    logger.info("Starting offline asset rescaling with Dask...")

    # Load the offline annotations first
    objathor_annotations = load_objathor_annotation()
    if not objathor_annotations:
        logger.warning("ObjaTHOR annotations not found. Skipping rescaling.")
        return

    # Query assets that have not been rescaled yet
    conn = sqlite3.connect(EXTRA_INDEX_DB_FILE)
    cursor = conn.cursor()
    query = "SELECT uuid, file_path FROM assets WHERE fs_path_rescaled IS NULL"
    params = ()
    if LIMIT:
        query += " LIMIT ?"
        params += (LIMIT,)
    target_assets = cursor.execute(query, params).fetchall()
    conn.close()

    # Identify rescaling jobs from offline data
    jobs = []
    for uuid, file_path_str in target_assets:
        if uuid in objathor_annotations:
            file_path = Path(file_path_str)
            _, original_dims, _ = get_glb_dimensions(file_path)
            analysis = extract_scale_analysis_from_objathor(uuid, objathor_annotations, original_dims)
            if analysis and analysis["misscaled"]:
                sf = analysis["correction_factor"]
                if sf is not None and not math.isclose(sf, 1.0, abs_tol=0.01):
                    jobs.append({'uuid': uuid, 'file_path': file_path_str, 'scaling_factor': sf})

    if not jobs:
        logger.info("No assets found requiring offline rescaling.")
        return
    
    logger.info(f"Found {len(jobs)} assets for offline rescaling.")

    # Set up Dask and run the jobs
    cluster = LocalCluster(threads_per_worker=1)
    client = Client(cluster)
    logger.info(f"Dask dashboard available at: {client.dashboard_link}")

    futures = client.map(
        rescale_asset_worker,
        [j['uuid'] for j in jobs],
        [j['file_path'] for j in jobs],
        [j['scaling_factor'] for j in jobs],
    )

    # Process results and prepare for DB update
    conn = sqlite3.connect(EXTRA_INDEX_DB_FILE)
    cursor = conn.cursor()
    update_data = []
    timestamp = datetime.datetime.now().isoformat()
    successful, errors = 0, 0

    with tqdm(as_completed(futures, with_results=True), total=len(jobs), desc="Rescaling assets") as pbar:
        for future, result in pbar:
            if isinstance(result, Exception):
                errors += 1
                continue
            
            uuid, scaled_path, sf = result
            if scaled_path == "failure":
                errors += 1
                continue

            # Append data for batch update
            update_data.append((scaled_path, sf, "objaverse-thor", version, timestamp, uuid))
            successful += 1

            if len(update_data) >= BATCH_SIZE:
                cursor.executemany(
                    """UPDATE assets SET
                       fs_path_rescaled = ?,
                       scaling_factor = ?,
                       rescaled_by = ?,
                       metadata_version = ?,
                       last_updated = ?
                       WHERE uuid = ?""",
                    update_data,
                )
                conn.commit()
                update_data = []

    # Commit any remaining updates
    if update_data:
        cursor.executemany(
            """UPDATE assets SET
               fs_path_rescaled = ?,
               scaling_factor = ?,
               rescaled_by = ?,
               metadata_version = ?,
               last_updated = ?
               WHERE uuid = ?""",
            update_data,
        )
        conn.commit()
    conn.close()

    client.close()
    cluster.close()
    logger.info(f"Offline rescaling complete. Success: {successful}, Errors: {errors}")


def main():
    global LIMIT
    global OBJATHOR_ONLY
    parser = argparse.ArgumentParser(
        description="Setup extra index database with metadata for 3D assets"
    )
    # NOTE: data_dir is sourced from core/config.py
    parser.add_argument("--reset", action="store_true", help="Clear all existing data.")
    parser.add_argument(
        "--strategy",
        choices=["vlm_only", "prefer_external", "external_only"],
        # default="prefer_external",
        default="external_only",
        help="Annotation strategy: vlm_only (VLM analysis only), prefer_external (try external first, fallback to VLM), external_only (external annotations only)",
    )
    parser.add_argument(
        "--compute-origins",
        action="store_true",
        help="Compute origin types for all assets using Dask (CPU-intensive, parallelized)",
    )
    parser.add_argument(
        "--perform-rescaling",
        action="store_true",
        help="Perform rescaling for assets using offline data and Dask (CPU-intensive, parallelized)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of assets to process for testing.",
    )
    parser.add_argument(
        "--objathor-only",
        action="store_true",
        default=False,
        help="Only target assets contained in ObjaTHOR for metadata computation.",
    )
    args = parser.parse_args()
    LIMIT = args.limit
    OBJATHOR_ONLY = args.objathor_only
    if OBJATHOR_ONLY:
        load_objathor_annotation()

    setup_database()
    if args.reset:
        reset_metadata()

    # Run origin analysis if requested
    if args.compute_origins:
        compute_origin_types_dask(METADATA_VERSION)

    # Run rescaling if requested
    if args.perform_rescaling:
        perform_rescaling_dask(METADATA_VERSION)

    for _source_name, local_dir in LOCAL_FS_PATHS.items():
        setup_index(Path(local_dir).expanduser())
        compute_metadata(METADATA_VERSION, strategy=args.strategy)


if __name__ == "__main__":
    main()
