import requests
import base64

import io

from PIL import Image

from src.graphics_db_server.logging import logger
from test_object_retrieval import test_object_search


def test_thumbnail_retrieval():
    """
    Tests getting object thumbnails.
    """
    search_results = test_object_search("a blue car")
    object_uids = [object["uid"] for object in search_results]

    response = requests.post(
        "http://localhost:2692/api/v0/objects/thumbnails",
        json={"object_uids": object_uids},
    )
    logger.info(f"object UIDs: {object_uids}. Response: {response}")
    assert response.status_code == 200
    response_json = response.json()
    assert len(response_json) > 0
    for uid, image_data in response_json.items():
        assert uid in object_uids
        # Check if the image data is a valid base64 string
        decoded_image = base64.b64decode(image_data)
        assert decoded_image

        # Load the image data into a PIL image
        image = Image.open(io.BytesIO(decoded_image))
        assert image


if __name__ == "__main__":
    test_thumbnail_retrieval()
