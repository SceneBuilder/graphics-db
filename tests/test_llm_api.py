import requests
from pathlib import Path

from src.graphics_db_server.logging import logger


def test_report_generation(query_text: str):
    """
    Tests LLM/VLM-consumable asset search report.
    """
    assets_response = requests.get(
        "http://localhost:2692/api/v0/assets/search",
        params={"query": query_text},
    )
    logger.info(f"Query: {query_text}. Response: {assets_response}")
    assets = assets_response.json()
    response = requests.get(
        "http://localhost:2692/api/v0/assets/report",
        params={"asset_uids": [asset["uid"] for asset in assets]},
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
