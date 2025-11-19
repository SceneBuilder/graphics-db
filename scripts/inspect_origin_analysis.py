import sqlite3
import os
import pathlib
from loguru import logger

# --- Configuration ---
DB_PATH = "graphics_db_extra_index.db"
OUTPUT_BASE_DIR = "test_output"
TABLE_NAME = "assets"
PATH_COLUMN = "fs_path"
RESCALED_PATH_COLUMN = "fs_path_rescaled"
RECENTERED_PATH_COLUMN = "fs_path_recentered"
TYPE_COLUMN = "origin_type"

# Setup basic logging

def classify_and_symlink_files():
    """
    Connects to the SQLite database, reads file metadata, and creates
    classified symbolic links in subdirectories.
    """
    # Check if the database file exists
    if not os.path.exists(DB_PATH):
        logger.error(f"Database file not found at '{DB_PATH}'. Please check the path.")
        return

    try:
        # Connect to the database and fetch data
        conn = sqlite3.connect(DB_PATH)
        # Use a dictionary cursor for easier column access by name
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        logger.info(f"Connected to database: {DB_PATH}")
        
        query = f"SELECT {PATH_COLUMN}, {RESCALED_PATH_COLUMN}, {RECENTERED_PATH_COLUMN}, {TYPE_COLUMN} FROM {TABLE_NAME} WHERE {PATH_COLUMN} IS NOT NULL"
        logger.info(f"Executing query: {query}")
        
        cursor.execute(query)
        records = cursor.fetchall()

        if not records:
            logger.warning("No records found in the database. Nothing to do.")
            return

        logger.info(f"Found {len(records)} records to process.")

        # Process each record
        for record in records:
            try:
                origin_type = record[TYPE_COLUMN]
                
                # Determine primary path: prefer rescaled, then original
                primary_file_path_str = None
                if record[RESCALED_PATH_COLUMN] and record[RESCALED_PATH_COLUMN].strip():
                    primary_file_path_str = record[RESCALED_PATH_COLUMN]
                elif record[PATH_COLUMN] and record[PATH_COLUMN].strip():
                    primary_file_path_str = record[PATH_COLUMN]

                if not origin_type or not primary_file_path_str:
                    logger.warning(f"Skipping record with missing data: {dict(record)}")
                    continue

                primary_file_path = pathlib.Path(primary_file_path_str).resolve()

                # Ensure the primary file exists before creating a link
                if not primary_file_path.exists():
                    logger.warning(f"Primary source file does not exist, skipping: {primary_file_path}")
                    continue

                # Create target directory
                target_dir = pathlib.Path(OUTPUT_BASE_DIR) / origin_type
                target_dir.mkdir(parents=True, exist_ok=True)

                # Symlink primary file
                primary_link_path = target_dir / primary_file_path.name
                
                if primary_link_path.exists() or primary_link_path.is_symlink():
                    logger.info(f"Primary link already exists, skipping: {primary_link_path}")
                else:
                    primary_link_path.symlink_to(primary_file_path)
                    logger.info(f"Created primary link: {primary_link_path} -> {primary_file_path}")

                # Handle recentered file if available
                recentered_file_path_str = record[RECENTERED_PATH_COLUMN]
                if recentered_file_path_str and recentered_file_path_str.strip():
                    recentered_file_path = pathlib.Path(recentered_file_path_str).resolve()
                    if not recentered_file_path.exists():
                        logger.warning(f"Recentered source file does not exist, skipping: {recentered_file_path}")
                    else:
                        recentered_link_path = target_dir / recentered_file_path.name
                        
                        if recentered_link_path.exists() or recentered_link_path.is_symlink():
                            logger.info(f"Recentered link already exists, skipping: {recentered_link_path}")
                        else:
                            recentered_link_path.symlink_to(recentered_file_path)
                            logger.info(f"Created recentered link: {recentered_link_path} -> {recentered_file_path}")

            except (sqlite3.Error, KeyError, TypeError) as e:
                logger.error(f"Error processing record {dict(record)}: {e}")

    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    classify_and_symlink_files()
    logger.info("Script finished.")
