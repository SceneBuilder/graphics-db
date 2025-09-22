from graphics_db_server.db.crud import insert_objaverse_assets, insert_polyhaven_assets
from graphics_db_server.db.session import get_db_connection
from graphics_db_server.sources.from_objaverse import load_objaverse_assets
from graphics_db_server.sources.from_polyhaven import load_polyhaven_assets
from graphics_db_server.logging import logger
from graphics_db_server.core.config import VALIDATE_SCALE, SCALE_RESOLUTION_STRATEGY


if __name__ == "__main__":
    logger.info("Ingesting data...")
    with get_db_connection() as conn:
        # Load Objaverse assets with validation options
        logger.info("Loading Objaverse assets...")
        options = {
            "validate_scale": VALIDATE_SCALE,
            "scale_resolution_strategy": SCALE_RESOLUTION_STRATEGY,
        }
        objaverse_assets = load_objaverse_assets(**options)
        logger.info(f"Loaded {len(objaverse_assets)} Objaverse assets")

        # Load Poly Haven assets
        logger.info("Loading Poly Haven assets...")
        polyhaven_assets = load_polyhaven_assets(asset_type="textures")
        logger.info(f"Loaded {len(polyhaven_assets)} Poly Haven assets")

        # Insert assets into their respective tables
        if objaverse_assets:
            logger.info(f"Inserting {len(objaverse_assets)} Objaverse assets...")
            insert_objaverse_assets(conn, objaverse_assets)

        if polyhaven_assets:
            logger.info(f"Inserting {len(polyhaven_assets)} Poly Haven assets...")
            insert_polyhaven_assets(conn, polyhaven_assets)

    logger.info("Ingesting data complete")
