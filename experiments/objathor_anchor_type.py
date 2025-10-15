# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %% [markdown]
# # **ObjaTHOR Anchor Type Analysis**

# %% [markdown]
# ## Setup

# %%
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib_venn import venn3

from graphics_db_server.scripts.setup_extra_index_objathor import load_objathor_annotation
from graphics_db_server.core.config import OBJATHOR_ANNO_JSON_PATH, EXTRA_INDEX_DB_FILE

# %%
# Load the ObjaTHOR annotations
load_objathor_annotation(OBJATHOR_ANNO_JSON_PATH)

# objathor_annotation is now loaded as a global dictionary
from graphics_db_server.scripts.setup_extra_index_objathor import objathor_annotation


# %%
def print_fs_paths(uuids: list[str]):
    """Print the file system paths (prioritizing rescaled) for given asset UUIDs."""
    conn = sqlite3.connect(EXTRA_INDEX_DB_FILE)
    assert Path(EXTRA_INDEX_DB_FILE).exists()
    
    cursor = conn.cursor()
    for uid in uuids:
        cursor.execute(
            "SELECT fs_path_rescaled, fs_path FROM assets WHERE uuid = ?",
            (uid,)
        )
        result = cursor.fetchone()
        if result:
            rescaled, original = result
            path = rescaled if rescaled else original
            print(f"{uid}: {path}")
        else:
            print(f"{uid}: Not found in extra index")
    conn.close()


# %% [markdown]
# ## Experiment

# %%
# Extract sets for each boolean property
set_floor = {uid for uid, data in objathor_annotation.items() if data.get('onFloor', False)}
set_object = {uid for uid, data in objathor_annotation.items() if data.get('onObject', False)}
set_wall = {uid for uid, data in objathor_annotation.items() if data.get('onWall', False)}

all_three = set_floor & set_object & set_wall

# Create the Venn diagram
venn3([set_floor, set_object, set_wall],
      set_labels=('onFloor', 'onObject', 'onWall'))

plt.title('Co-occurrence of ObjaTHOR Anchor Type Properties (onFloor, onObject, onWall)')
plt.show()

# %% [markdown]
# ## Inspect

# %% [markdown]
# ### All Three

# %%
# Print paths for assets in all three categories
print(f"Number of assets in all three categories: {len(all_three)}")
print_fs_paths(list(all_three))

# %% [markdown]
# **Verdict**: Clasify as `onFloor` (as far as re-centering goes).

# %% [markdown]
# ### `onWall`

# %%
print(f"Number of assets in onWall category: {len(set_wall)}")
print_fs_paths(list(set_wall))

# %%
