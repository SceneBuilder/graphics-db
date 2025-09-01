import os
import requests
from typing import Any, Optional
from pathlib import Path

from graphics_db_server.schemas.asset import AssetCreate
from graphics_db_server.core.config import (
    POLYHAVEN_API_BASE_URL,
    POLYHAVEN_USER_AGENT,
    POLYHAVEN_CACHE_DIR,
)
from graphics_db_server.embeddings.clip import get_clip_embeddings
from graphics_db_server.embeddings.sbert import get_sbert_embeddings


def _make_api_request(endpoint: str) -> dict[str, Any]:
    """
    Makes a request to the Poly Haven API with proper headers.
    """
    headers = {"User-Agent": POLYHAVEN_USER_AGENT}
    url = f"{POLYHAVEN_API_BASE_URL}/{endpoint}"

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def _get_asset_list(asset_type: str = "textures") -> dict[str, Any]:
    """
    Retrieves the list of assets from Poly Haven API.

    Args:
        asset_type: Type of assets to fetch ("textures", "hdris", "models")

    Returns:
        Dictionary mapping asset IDs to their metadata
    """
    return _make_api_request(f"assets?type={asset_type}")


def _get_asset_info(asset_id: str) -> dict[str, Any]:
    """
    Retrieves detailed information about a specific asset.
    """
    return _make_api_request(f"info/{asset_id}")


def _get_asset_files(asset_id: str) -> dict[str, Any]:
    """
    Retrieves file information (download URLs) for a specific asset.
    """
    return _make_api_request(f"files/{asset_id}")


def _get_diffuse_url(asset_id: str) -> Optional[str]:
    """
    Gets the download URL for the 1k diffuse map of a texture asset.

    Args:
        asset_id: The asset identifier

    Returns:
        Download URL for the 1k diffuse map, or None if not found
    """
    files_data = _get_asset_files(asset_id)

    # Check if 1k resolution and jpg files exist
    if "1k" in files_data and "jpg" in files_data["1k"]:
        jpg_files = files_data["1k"]["jpg"]

        # Find the diffuse map (usually has "diff" in the filename)
        for filename, file_info in jpg_files.items():
            if "diff" in filename.lower():
                return file_info["url"]

    print(f"No 1k diffuse map found for asset {asset_id}")
    return None


def _create_text_description(asset_id: str, asset_data: dict[str, Any]) -> str:
    """
    Creates a text description for embedding generation from asset metadata.

    Args:
        asset_id: The asset identifier
        asset_data: Asset metadata from the API

    Returns:
        Combined text description for embedding generation
    """
    parts = [asset_id.replace("_", " ").replace("-", " ")]

    # Add categories if available
    if "categories" in asset_data and asset_data["categories"]:
        parts.extend(asset_data["categories"])

    # Add tags if available
    if "tags" in asset_data and asset_data["tags"]:
        parts.extend(asset_data["tags"])

    return " ".join(parts)


def load_polyhaven_assets(
    limit: Optional[int] = None, asset_type: str = "textures"
) -> list[AssetCreate]:
    """
    Loads asset metadata from the Poly Haven API and generates embeddings.

    Args:
        limit: Maximum number of assets to load (None for all)
        asset_type: Type of assets to fetch ("textures", "hdris", "models")

    Returns:
        List of AssetCreate objects with generated embeddings
    """
    # Get list of assets
    asset_list = _get_asset_list(asset_type)

    assets: list[AssetCreate] = []

    for asset_id, asset_metadata in asset_list.items():
        if limit is not None and len(assets) >= limit:
            break

        try:
            # Get detailed asset information
            asset_info = _get_asset_info(asset_id)

            # Create text description for embeddings
            text_description = _create_text_description(asset_id, asset_info)

            # Generate embeddings
            clip_embedding = get_clip_embeddings(text_description)
            sbert_embedding = get_sbert_embeddings(text_description)

            # Create asset URL (link to Poly Haven page)
            asset_url = f"https://polyhaven.com/a/{asset_id}"

            # Extract tags and categories
            tags = []
            if "categories" in asset_info and asset_info["categories"]:
                tags.extend(asset_info["categories"])
            if "tags" in asset_info and asset_info["tags"]:
                tags.extend(asset_info["tags"])

            asset = AssetCreate(
                uid=asset_id,
                url=asset_url,
                tags=tags,
                source="polyhaven",
                license="CC0",  # All Poly Haven assets are CC0
                asset_type=asset_type.rstrip("s"),  # "textures" -> "texture"
                clip_embedding=clip_embedding,
                sbert_embedding=sbert_embedding,
            )
            assets.append(asset)

        except Exception as e:
            print(f"Error processing asset {asset_id}: {e}")
            continue

    return assets


def ensure_cache_dir():
    """
    Ensures the Poly Haven cache directory exists.
    """
    cache_dir = Path(POLYHAVEN_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_diffuse_maps(asset_ids: list[str]) -> dict[str, str]:
    """
    Downloads 1k diffuse maps from Poly Haven for the given asset IDs.

    Args:
        asset_ids: List of asset IDs to download

    Returns:
        Dictionary mapping asset IDs to local file paths of downloaded diffuse maps
    """
    cache_dir = ensure_cache_dir()
    downloaded_assets = {}

    for asset_id in asset_ids:
        diffuse_url = _get_diffuse_url(asset_id)

        if diffuse_url:
            # Create local filename
            file_extension = diffuse_url.split(".")[
                -1
            ]  # Get extension (jpg, png, etc.)
            local_filename = f"{asset_id}_diff_1k.{file_extension}"
            local_path = cache_dir / local_filename

            # Download if not already cached
            if not local_path.exists():
                print(f"Downloading diffuse map for {asset_id}...")
                response = requests.get(
                    diffuse_url, headers={"User-Agent": POLYHAVEN_USER_AGENT}
                )

                if response.status_code == 200:
                    with open(local_path, "wb") as f:
                        f.write(response.content)
                    print(f"Downloaded: {local_path}")
                    downloaded_assets[asset_id] = str(local_path)
                else:
                    print(f"Failed to download {asset_id}: HTTP {response.status_code}")
            else:
                print(f"Using cached: {local_path}")
                downloaded_assets[asset_id] = str(local_path)
        else:
            print(f"No diffuse URL found for {asset_id}")

    return downloaded_assets


if __name__ == "__main__":
    # Test with a small sample
    assets = load_polyhaven_assets(limit=3)
    print(f"Loaded {len(assets)} assets from Poly Haven")
    for asset in assets:
        print(f"- {asset.uid}: {asset.url} (tags: {asset.tags[:3]})")

    # Test download functionality
    if assets:
        asset_ids = [asset.uid for asset in assets[:2]]  # Test with first 2 assets
        print(f"\nTesting download for: {asset_ids}")
        downloaded = download_diffuse_maps(asset_ids)
        print(f"Downloaded {len(downloaded)} diffuse maps")
