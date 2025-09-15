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


def test_get_asset_details():
    """
    Tests that get_asset_details returns None for non-existent UUID.
    """
    result = get_asset_details("non-existent-uuid")
    assert result is None

    # Test with a hardcoded UUID that should exist
    result = get_asset_details("03c68480c9c34174826f836b6c95c27e")
    assert result is not None
    assert isinstance(result, dict)
    assert "uuid" in result


if __name__ == "__main__":
    test_get_asset_details()
