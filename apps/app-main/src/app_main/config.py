"""
Application configuration for app-main.
"""

import os

# ROOT DATA FOLDER
DATA_FOLDER = os.environ.get("DATA_FOLDER", "./data")

# LANGGRAPH CHECKPOINT FILE
sqlite_folder = os.path.join(DATA_FOLDER, "sqlite-db")
os.makedirs(sqlite_folder, exist_ok=True)
LANGGRAPH_CHECKPOINT_FILE = os.path.join(sqlite_folder, "checkpoints.sqlite")

# UPLOADS FOLDER
UPLOADS_FOLDER = os.path.join(DATA_FOLDER, "uploads")
os.makedirs(UPLOADS_FOLDER, exist_ok=True)

# TIKTOKEN CACHE FOLDER
TIKTOKEN_CACHE_DIR = os.path.join(DATA_FOLDER, "tiktoken-cache")
os.makedirs(TIKTOKEN_CACHE_DIR, exist_ok=True)
