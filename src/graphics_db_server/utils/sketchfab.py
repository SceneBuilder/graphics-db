import os
from typing import Optional

import requests

from graphics_db_server.logging import logger


def get_sketchfab_model_info(uid: str) -> Optional[dict]:
    """
    Fetches model information from Sketchfab for a given UID.
    """
    api_token = os.getenv("SKETCHFAB_API_TOKEN")
    if not api_token:
        logger.error("SKETCHFAB_API_TOKEN environment variable not set.")
        return None

    url = f"https://api.sketchfab.com/v3/models/{uid}"
    headers = {"Authorization": f"Bearer {api_token}"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Sketchfab model info for {uid}: {e}")
        return None

def get_sketchfab_thumbnail_url(uid: str) -> Optional[str]:
    """
    Gets the URL of the largest thumbnail for a Sketchfab model.
    """
    model_info = get_sketchfab_model_info(uid)
    if not model_info or "thumbnails" not in model_info:
        return None

    thumbnails = model_info["thumbnails"]["images"]
    if not thumbnails:
        return None

    # Find the largest thumbnail
    largest_thumbnail = max(thumbnails, key=lambda x: x["width"] * x["height"])
    return largest_thumbnail["url"]

def get_sketchfab_thumbnails(uids: list[str]) -> dict[str, str]:
    """
    Gets thumbnail URLs for a list of Sketchfab model UIDs.
    """
    # TODO: Implement concurrent requests for better performance
    thumbnail_urls = {}
    for uid in uids:
        url = get_sketchfab_thumbnail_url(uid)
        if url:
            thumbnail_urls[uid] = url
    return thumbnail_urls