from graphics_db_server.db.session import get_db_connection
from graphics_db_server.db.objaverse_crud import objaverse_crud
from graphics_db_server.db.polyhaven_crud import polyhaven_crud
from graphics_db_server.sources.from_objaverse import load_objaverse_assets
from graphics_db_server.sources.from_polyhaven import load_polyhaven_assets
from graphics_db_server.schemas import ObjaverseAssetCreate, PolyhavenAssetCreate
from graphics_db_server.logging import logger
from graphics_db_server.core.config import VALIDATE_SCALE, SCALE_RESOLUTION_STRATEGY


def convert_to_objaverse_assets(legacy_assets):
    """Convert legacy AssetCreate objects to ObjaverseAssetCreate with categorization"""
    objaverse_assets = []

    for asset in legacy_assets:
        if asset.source != "objaverse":
            continue

        # Create Objaverse asset with auto-categorization
        objaverse_asset = ObjaverseAssetCreate(
            uid=asset.uid,
            viewer_url=asset.url,  # Map legacy url to viewer_url
            url=asset.url,  # Keep for BaseAsset compatibility
            tags=asset.tags,
            clip_embedding=asset.clip_embedding,
            sbert_embedding=asset.sbert_embedding,
            # Auto-categorization happens in __init__
        )
        objaverse_assets.append(objaverse_asset)

    return objaverse_assets


def convert_to_polyhaven_assets(legacy_assets):
    """Convert legacy AssetCreate objects to PolyhavenAssetCreate with categorization"""
    polyhaven_assets = []

    for asset in legacy_assets:
        if asset.source != "polyhaven":
            continue

        # Create Poly Haven asset with auto-categorization
        polyhaven_asset = PolyhavenAssetCreate(
            uid=asset.uid,
            polyhaven_url=asset.url,  # Map legacy url to polyhaven_url
            url=asset.url,  # Keep for BaseAsset compatibility
            tags=asset.tags,
            categories=getattr(asset, "categories", []),  # If available from source
            asset_type=asset.asset_type or "texture",
            clip_embedding=asset.clip_embedding,
            sbert_embedding=asset.sbert_embedding,
            # Auto-categorization happens in __init__
        )
        polyhaven_assets.append(polyhaven_asset)

    return polyhaven_assets


if __name__ == "__main__":
    logger.info("Ingesting data with pre-categorization...")

    with get_db_connection() as conn:
        # Load Objaverse assets with validation options from main branch
        logger.info("Loading Objaverse assets...")
        options = {"validate_scale": VALIDATE_SCALE,
                   "scale_resolution_strategy": SCALE_RESOLUTION_STRATEGY}
        legacy_objaverse_assets = load_objaverse_assets(**options)
        logger.info(f"Loaded {len(legacy_objaverse_assets)} raw Objaverse assets")

        # Convert to categorized Objaverse assets
        logger.info("Converting to categorized Objaverse assets...")
        objaverse_assets = convert_to_objaverse_assets(legacy_objaverse_assets)
        logger.info(f"Categorized {len(objaverse_assets)} Objaverse assets")

        # Insert Objaverse assets into separate table
        if objaverse_assets:
            logger.info("Inserting Objaverse assets...")
            objaverse_crud.insert_assets(conn, objaverse_assets)

        # Load Poly Haven assets
        logger.info("Loading Poly Haven assets...")
        legacy_polyhaven_assets = load_polyhaven_assets(limit=50, asset_type="textures")
        logger.info(f"Loaded {len(legacy_polyhaven_assets)} raw Poly Haven assets")

        # Convert to categorized Poly Haven assets
        logger.info("Converting to categorized Poly Haven assets...")
        polyhaven_assets = convert_to_polyhaven_assets(legacy_polyhaven_assets)
        logger.info(f"Categorized {len(polyhaven_assets)} Poly Haven assets")

        # Insert Poly Haven assets into separate table
        if polyhaven_assets:
            logger.info("Inserting Poly Haven assets...")
            polyhaven_crud.insert_assets(conn, polyhaven_assets)

        total_assets = len(objaverse_assets) + len(polyhaven_assets)
        logger.success(
            f"Successfully ingested {total_assets} assets with pre-categorization"
        )

        # Log categorization summary
        if objaverse_assets:
            categories = {}
            for asset in objaverse_assets:
                cat = asset.asset_category or "unknown"
                categories[cat] = categories.get(cat, 0) + 1
            logger.info(f"Objaverse categories: {categories}")

        if polyhaven_assets:
            categories = {}
            for asset in polyhaven_assets:
                cat = asset.asset_category or "unknown"
                categories[cat] = categories.get(cat, 0) + 1
            logger.info(f"Poly Haven categories: {categories}")

    logger.success("Data ingestion complete!")
