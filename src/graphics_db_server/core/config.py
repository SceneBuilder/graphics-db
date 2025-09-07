import os

from dotenv import load_dotenv
from pydantic import BaseModel

# Consts
# Database
TABLE_NAME = "assets"
EMBEDDING_DIMS = 768
INDEX_NAME = "assets_vec_idx"
INDEX_TYPE = "diskann"
SIMILARITY_OPS = "vector_cosine_ops"

# Data sources
EMBEDDING_PATHS = {
    "Objaverse": {
        "clip": "data/objaverse/clip_features.pkl",
        "sbert": "data/objaverse/sbert_features.pkl",
    }
}
LOCAL_FS_PATHS = {
    # "Objaverse": "./data/mini_objaverse"  # debug
    "Objaverse": "~/.objaverse_full"  # prod
}
OBJATHOR_ANNO_JSON_PATH = "~/.objathor-assets/2023_09_23/annotations.json"  # set this to your ObjaTHOR JSON file path

# Data validation
# Objects
# VALIDATE_SCALE = True
VALIDATE_SCALE = False
SCALE_RESOLUTION_STRATEGY = "reject"  # options: ["reject", "rescale"]
SCALE_MAX_LENGTH_THRESHOLD = 100.0  # filter out centimeter-based (or just large) assets

# VLM
# VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
VLM_PROVIDER_BASE_URL = "http://localhost:8000/v1"

# App
USE_MEAN_POOL = True
THUMBNAIL_RESOLUTION = 1024

load_dotenv()


class DBSettings(BaseModel):
    pguser: str = os.environ["POSTGRES_USER"]
    pgpass: str = os.environ["POSTGRES_PASSWORD"]
    pgname: str = os.environ["POSTGRES_DB"]
    pghost: str = os.environ.get("POSTGRES_HOST", "db")
    port: str = os.environ.get("POSTGRES_PORT", "5432")
    DATABASE_URL: str = f"postgresql://{pguser}:{pgpass}@{pghost}:{port}/{pgname}"


db_settings = DBSettings()
