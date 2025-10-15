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

# %%
from graphics_db_server.scripts.setup_extra_index_objathor import load_objathor_annotation
from graphics_db_server.core.config import OBJATHOR_ANNO_JSON_PATH
from matplotlib_venn import venn3
import matplotlib.pyplot as plt

# Load the ObjaTHOR annotations
load_objathor_annotation(OBJATHOR_ANNO_JSON_PATH)

# Assuming objathor_annotation is now loaded as a global dictionary
from src.graphics_db_server.scripts.setup_extra_index_objathor import objathor_annotation

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

# %%
