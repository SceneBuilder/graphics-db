import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from graphics_db_server.api.v0.endpoints import objects, materials
from graphics_db_server.db.session import get_db_connection
from graphics_db_server.logging import configure_logging, logger

configure_logging(level="DEBUG")

app = FastAPI(
    title="Graphics DB API",
    description="Semantic search API for 3D objects and surface materials",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Purpose-specific endpoints
app.include_router(objects.router, prefix="/api/v0", tags=["3D Objects"])
app.include_router(materials.router, prefix="/api/v0", tags=["Surface Materials"])


@app.get("/healthcheck")
def healthcheck():
    """
    Check database connection and asset counts in separated tables.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.execute("SELECT COUNT(*) FROM objaverse_assets;")
                objaverse_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM polyhaven_assets;")
                polyhaven_count = cur.fetchone()[0]

        return {
            "status": "ok",
            "db": "ok",
            "objaverse_assets": objaverse_count,
            "polyhaven_assets": polyhaven_count,
            "total_assets": objaverse_count + polyhaven_count,
        }
    except psycopg.Error as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection error")
