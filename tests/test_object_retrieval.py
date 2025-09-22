import requests

from graphics_db_server.core.config import GRAPHICS_DB_BASE_URL
from graphics_db_server.logging import logger


def test_object_search(query_text: str):
    """
    Tests graphics asset semantic search functionality.
    """
    response = requests.get(
        f"{GRAPHICS_DB_BASE_URL}/api/v0/objects/search",
        params={"query": query_text},
    )
    logger.info(f"Query: {query_text}. Response: {response}")
    assert response.status_code == 200
    response_json = response.json()
    assert len(response_json) > 0
    return response_json


if __name__ == "__main__":
    test_object_search("a blue car")
