import sqlite3

from graphics_db_server.core.config import EXTRA_INDEX_DB_FILE


def get_asset_details(uuid_to_check: str) -> dict | None:
    """
    Retrieves all stored data for a given UUID.

    Args:
        uuid_to_check (str): The asset UUID to find.

    Returns:
        dict | None: A dictionary of the asset's data if found, otherwise None.
    """
    conn = sqlite3.connect(EXTRA_INDEX_DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM assets WHERE uuid = ?", (uuid_to_check,))
    result = cursor.fetchone()

    conn.close()
    return dict(result) if result else None
