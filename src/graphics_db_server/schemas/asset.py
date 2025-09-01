import uuid
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict


class Asset(BaseModel):
    uid: str
    url: str
    tags: Optional[list[str]] = []
    source: str | None = None # TEMP
    license: str | None = None # TEMP
    asset_type: str | None = None  # "model", "texture", "hdri"
    # sourceId: str  # NOTE: let's try not to keep this unless really necessary

    model_config = ConfigDict(from_attributes=True)


class AssetCreate(Asset):
    clip_embedding: list[float] | np.ndarray
    sbert_embedding: list[float] | np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)


# class SearchResult(BaseModel):
#     raise NotImplementedError
#     # NOTE: Let's keep the schema simple for now.
