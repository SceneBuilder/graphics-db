import requests
import base64

import io

from PIL import Image

from graphics_db_server.core.config import GRAPHICS_DB_BASE_URL
from graphics_db_server.logging import logger
from test_object_retrieval import test_object_search


def test_thumbnail_retrieval():
    """
    Tests getting object thumbnails.
    """
    search_results = test_object_search("a blue car")
    uids = [object["uid"] for object in search_results]

    response = requests.post(
        f"{GRAPHICS_DB_BASE_URL}/api/v0/objects/thumbnails",
        json={"uids": uids},
    )
    logger.info(f"object UIDs: {uids}. Response: {response}")
    assert response.status_code == 200
    response_json = response.json()
    assert len(response_json) > 0
    for uid, image_data in response_json.items():
        assert uid in uids
        # Check if the image data is a valid base64 string
        decoded_image = base64.b64decode(image_data)
        assert decoded_image

        # Load the image data into a PIL image
        image = Image.open(io.BytesIO(decoded_image))
        assert image


if __name__ == "__main__":
    test_thumbnail_retrieval()
