"""
This script sets up an additional SQLite-based DB for metadata interaction,
such as computing 3D bounding box, analyzing re-scaling parameters, thumbnails,
and filesystem locations for fast disk lookups.

NOTE: Currently, this script is specifically written for Objaverse dataset.
TODO: Modularize to suit different asset data sources.
"""

import argparse
import datetime
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Literal

# import tqdm
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
    LOCAL_FS_PATHS,
    THUMBNAIL_RESOLUTION,
    VLM_MODEL_NAME,
    VLM_PROVIDER_BASE_URL,
)
from graphics_db_server.logging import logger
from graphics_db_server.tools.read_file import read_media_file
from graphics_db_server.utils.geometry import get_glb_dimensions
from graphics_db_server.utils.pai import transform_paths_to_binary
from graphics_db_server.utils.scale_validation import scale_glb_model
from graphics_db_server.utils.thumbnail import generate_thumbnail_from_glb

# Configuration
DB_FILE = "graphics_db_extra_index.db"
THUMBNAIL_DIR = Path("/media/ycho358/YunhoStrgExt/graphics_db_thumbnails")
METADATA_VERSION = 1  # NOTE: increment with logic changes
BATCH_SIZE = 1000

DEBUG = True
# DEBUG = False


def setup_database():
    """
    Sets up SQLite database. Adds columns for metadata if not exists.
    """
    conn = sqlite3.connect(DB_FILE)
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
        "metadata_version": "INTEGER",
        "last_updated": "TEXT",
        "thumbnail_paths": "TEXT",
        "fs_path": "TEXT",
        "fs_path_scaled": "TEXT",
    }

    cursor.execute("PRAGMA table_info(assets)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    for col_name, col_type in columns.items():
        if col_name not in existing_columns:
            logger.info(f"Adding column: {col_name}")
            cursor.execute(f"ALTER TABLE assets ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()
    logger.info(f"Database {DB_FILE} is set up and ready.")


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

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    logger.debug("Discovering all .glb files first to show progres...")
    asset_files = [p for p in data_dir.rglob("*.glb") if not p.stem.endswith("_scaled")]

    if not asset_files:
        logger.warning("No .glb files found to index.")
        conn.close()
        return

    logger.info(f"Found {len(asset_files)} files. Indexing...")
    asset_data_generator = ((p.stem, str(p.resolve())) for p in asset_files)

    with tqdm(total=len(asset_files), desc="Indexing assets") as pbar:
        while True:
            batch = [item for _, item in zip(range(BATCH_SIZE), asset_data_generator)]
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
    misscaled: bool
    misscaling_type: Literal["mm", "cm", "10x", "arbitrary"]
    correction_factor: float | None = None
    object_description: str | None = None
    rationale: str | None = None


# model = GoogleModel("gemini-2.5-flash")
client = AsyncOpenAI(base_url=VLM_PROVIDER_BASE_URL, api_key="empty")
model = OpenAIChatModel(
    VLM_MODEL_NAME,
    provider=OpenAIProvider(base_url=VLM_PROVIDER_BASE_URL, api_key="EMPTY"),
)
system_prompt = (
    "You are an AI model that is in charge of the quality control of "
    "3d assets that sometimes have an incorrect scale. "
    "Your job is to check if the given 3D assets have realistic sizes that are consistent with "
    "the typical sizes of respective objects in the real world. "
    "\nYou will be given paths to thumbnail images of the object, as well as the 3D dimensions. "
    "(If given image paths, please ALWAYS see all of the images first by using `read_media_file(filepath: str)`.) "
    "A common cause of error is the use of non-meter length unit such as mm or cm, "
    "despite GLTF specifications that mandate meter units. "
    "In this case, it is easy to deduce the correct scale—by simply dividing the "
    "dimensions by 100 or 1,000 respectively. "
    "There are also assets that are arbitrarily misscaled. "
    "\nPlease classify whether the object is misscaled, the type of misccaling (mm, cm, 10x, or arbitrary), "
    "and a 'correction factor' that must be applied to restore the asset to the correct scale. "
    "The correction factor is will be multiplied to the asset, so if you want to size it down, provide a number smaller than 1. "
    "Additionally, please output a description of what object you are looking at.\n"
    "Finally, please output a rationale explaining why you chose to classify it as misscaled or not, and your reasoning behind the correction factor."
)
# NOTE can include: output response example
# NOTE can run through claude's prompt optimizer
scale_analysis_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=ScaleAnalysisInput,
    output_type=ScaleAnalysisOutput,
    tools=[read_media_file],
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
        f"Dimensions: {[round(e, 2) for e in dims]}",
        "Please analyze this 3D asset.",
    )
    user_prompt = "\n".join(user_prompt)
    extra_info = (
        "\nExtra information:",
        f"- Larger than than 100 m: {max(dims) > 100}",
        f"- Dimensions if scaled down by 100: {[round(e / 100, 2) for e in dims]}",
        f"- Larger than 1,000 m: {max(dims) > 1000}",
        f"- Dimensions if scaled down by 1,000: {[round(e / 1000, 2) for e in dims]}",
    )
    extra_info = "\n".join(extra_info)
    # question = "Are you able to see the thumbnail images?"  # DEBUG
    # inputs = ScaleAnalysisInput(
    #     thumbnail_paths=thumbnail_paths, question=question if DEBUG else None
    # )  # ORIG
    thumbnail_contents: list[BinaryContent] = transform_paths_to_binary(thumbnail_paths)  # ALT
    inputs = thumbnail_contents  # TEMP
    if DEBUG:
        logger.debug(f"Asset [{uuid}] Inputs: {inputs}")
        logger.debug(f"Asset [{uuid}] User Prompt: {user_prompt + extra_info}")

    # response = scale_analysis_agent.run_sync(user_prompt + extra_info, deps=inputs)  # ORIG
    response = scale_analysis_agent.run_sync(
        thumbnail_contents + [user_prompt + extra_info]
    )  # ALT
    output: ScaleAnalysisOutput = response.output

    if DEBUG:
        logger.debug(f"Asset [{uuid}] Object Description: {output.object_description}")
        logger.debug(f"Asset [{uuid}] Rationale: {output.rationale}")

    sf = output.correction_factor  # scaling factor
    if sf is not None and not math.isclose(sf, 1, abs_tol=0.1):
        scaled_model_path = file_path.with_stem(f"{uuid}_scaled")
        success = scale_glb_model(file_path, scaled_model_path, sf, backend="blender")
        if not success:
            return "failure"

    if DEBUG:
        logger.debug(f"Asset [{uuid}] Output Scale: {sf}")
        logger.debug(f"Asset [{uuid}] Output Dimensions: {[round(e * sf, 2) for e in dims]}")

    if sf is None and output.misscaled:
        return "failure"
        # NOTE: This might be cases where dims are zeros, e.g., due to being non-surface photogrammetry

    return {
        "misscaled": 1 if output.misscaled else 0,
        "misscaling_type": output.misscaling_type,
        "dims_x": dims[0],
        "dims_y": dims[1],
        "dims_z": dims[2],
        "fs_path": str(file_path),
        "fs_path_scaled": str(scaled_model_path) if output.misscaled else None,
    }


def reset_metadata():
    """
    Clear all metadata.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    print("Resetting all existing metadata")
    cursor.execute("""
        UPDATE assets SET
            misscaled = NULL, 
            misscaling_type = NULL,
            dims_x = NULL,
            dims_y = NULL,
            dims_z = NULL,
            correction_factor = NULL,
            metadata_version = NULL,
            last_updated = NULL,
            thumbnail_paths = NULL
    """)
    conn.commit()
    logger.info("Metadata has been reset.")


def compute_metadata(version: int):
    """
    Computes and updates metadata for assets that are out of date.

    Args:
        version (int): The version of the metadata logic to apply.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    query = "SELECT uuid, file_path FROM assets WHERE metadata_version IS NULL OR metadata_version < ?"
    target_assets = cursor.execute(query, (version,)).fetchall()

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

    with tqdm(
        total=len(target_assets), desc=f"Computing metadata (v{version})"
    ) as pbar:
        for uuid, path_str in target_assets:
            file_path = Path(path_str)
            if file_path.exists():
                thumbnail_paths = generate_thumbnails(uuid, path_str, THUMBNAIL_DIR)
                thumbnail_paths = [str(path) for path in thumbnail_paths]  # TEMP
                metadata = calc_metadata(file_path, thumbnail_paths=thumbnail_paths)
                if metadata == "failure":
                    logger.warning(f"{uuid}: metadata generation failure.")
                    continue
                    # TODO: decide if row should be deleted from DB as well
                metadata["thumbnail_paths"] = json.dumps(thumbnail_paths)
                metadata["metadata_version"] = version
                metadata["last_updated"] = datetime.datetime.now().isoformat()

                update_query = f"UPDATE assets SET {', '.join(f'{k} = ?' for k in metadata)} WHERE uuid = ?"
                values = list(metadata.values()) + [uuid]

                cursor.execute(update_query, values)
            else:
                # Case in which file might have been deleted since indexing
                pbar.set_postfix_str(f"SKIPPING missing file: {uuid}", refresh=True)

            pbar.update(1)
            if pbar.n % BATCH_SIZE == 0:  # save periodically
                conn.commit()

    conn.commit()
    conn.close()
    logger.info("Metadata computation complete.")


def get_asset_details(uuid_to_check: str) -> dict | None:
    """
    Retrieves all stored data for a given UUID.

    Args:
        uuid_to_check (str): The asset UUID to find.

    Returns:
        dict | None: A dictionary of the asset's data if found, otherwise None.
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM assets WHERE uuid = ?", (uuid_to_check,))
    result = cursor.fetchone()

    conn.close()
    return dict(result) if result else None


def main():
    parser = argparse.ArgumentParser(description="")
    # NOTE: data_dir is sourced from core/config.py
    parser.add_argument("--reset", action="store_true", help="Clear all existing data.")
    args = parser.parse_args()

    setup_database()
    if args.reset:
        reset_metadata()
    for source_name, local_dir in LOCAL_FS_PATHS.items():
        setup_index(Path(local_dir))
        compute_metadata(METADATA_VERSION)


if __name__ == "__main__":
    main()
