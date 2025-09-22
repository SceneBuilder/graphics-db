from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict


class Asset(BaseModel):
    uid: str
    url: str
    tags: Optional[list[str]] = []
    source: str | None = None
    license: str | None = None
    asset_type: str | None = None

    model_config = ConfigDict(from_attributes=True)


class AssetCreate(Asset):
    clip_embedding: list[float] | np.ndarray
    sbert_embedding: list[float] | np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)


# class SearchResult(BaseModel):
#     raise NotImplementedError
#     # NOTE: Let's keep the schema simple for now.
