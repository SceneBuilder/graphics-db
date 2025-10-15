import sqlite3
import os
import pathlib
import logging

# --- Configuration ---
DB_PATH = "graphics_db_extra_index.db"
OUTPUT_BASE_DIR = "test_output"
TABLE_NAME = "assets"
# PATH_COLUMN = "fs_path"
PATH_COLUMN = "fs_path_rescaled"
TYPE_COLUMN = "origin_type"

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def classify_and_symlink_files():
    """
    Connects to the SQLite database, reads file metadata, and creates
    classified symbolic links in subdirectories.
    """
    # Check if the database file exists
    if not os.path.exists(DB_PATH):
        logging.error(f"Database file not found at '{DB_PATH}'. Please check the path.")
        return

    try:
        # Connect to the database and fetch data
        conn = sqlite3.connect(DB_PATH)
        # Use a dictionary cursor for easier column access by name
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        logging.info(f"Connected to database: {DB_PATH}")
        
        query = f"SELECT fs_path, fs_path_rescaled, {TYPE_COLUMN} FROM {TABLE_NAME}"
        logging.info(f"Executing query: {query}")
        
        cursor.execute(query)
        records = cursor.fetchall()

        if not records:
            logging.warning("No records found in the database. Nothing to do.")
            return

        logging.info(f"Found {len(records)} records to process.")

        # Process each record
        for record in records:
            try:
                origin_type = record[TYPE_COLUMN]
                fs_path = record['fs_path']
                fs_path_rescaled = record['fs_path_rescaled']
                if fs_path_rescaled and fs_path_rescaled.strip():
                    original_file_path_str = fs_path_rescaled
                else:
                    original_file_path_str = fs_path

                if not origin_type or not original_file_path_str:
                    logging.warning(f"Skipping record with missing data: {dict(record)}")
                    continue

                original_file_path = pathlib.Path(original_file_path_str).resolve()

                # Ensure the original file exists before creating a link
                if not original_file_path.exists():
                    logging.warning(f"Source file does not exist, skipping: {original_file_path}")
                    continue

                # Create target directory and symbolic link
                target_dir = pathlib.Path(OUTPUT_BASE_DIR) / origin_type
                target_dir.mkdir(parents=True, exist_ok=True)

                link_path = target_dir / original_file_path.name
                
                # Check if a link/file already exists to avoid errors
                if link_path.exists() or link_path.is_symlink():
                    logging.info(f"Link already exists, skipping: {link_path}")
                else:
                    link_path.symlink_to(original_file_path)
                    logging.info(f"Created link: {link_path} -> {original_file_path}")

            except (sqlite3.Error, KeyError, TypeError) as e:
                logging.error(f"Error processing record {dict(record)}: {e}")

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    classify_and_symlink_files()
    logging.info("Script finished.")
