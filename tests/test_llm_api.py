import requests
from pathlib import Path

from graphics_db_server.core.config import GRAPHICS_DB_BASE_URL
from graphics_db_server.logging import logger


def test_report_generation(query_text: str):
    """
    Tests LLM/VLM-consumable object search report.
    """
    objects_response = requests.get(
        f"{GRAPHICS_DB_BASE_URL}/api/v0/objects/search",
        params={"query": query_text},
    )
    logger.info(f"Query: {query_text}. Response: {objects_response}")
    objects = objects_response.json()
    response = requests.get(
        f"{GRAPHICS_DB_BASE_URL}/api/v0/objects/report",
        params={"uids": [object["uid"] for object in objects]},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert len(response_json) > 0

    # Save the markdown report to a file
    output_file = Path(__file__).parent / "output_search_report.md"
    with open(output_file, "w") as f:
        f.write(response_json)
    logger.info(f"Saved search report to {output_file}")


if __name__ == "__main__":
    test_report_generation("a blue car")
