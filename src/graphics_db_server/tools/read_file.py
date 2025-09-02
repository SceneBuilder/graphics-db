import mimetypes
import os

from pydantic_ai import BinaryContent

from graphics_db_server.logging import logger

# Controls whether to raise exceptions or return graceful error messages (for the LLM to try again)
# STRICT_MODE = True
STRICT_MODE = False


def read_media_file(file_path: str) -> BinaryContent | str:
    """
    Reads a media file from a given path and returns it as a BinaryContent object.

    This tool is a simplified, direct-access version of `read_file` for when
    the content is known to be binary, such as an image or video. 
    
    Behavior depends on STRICT_MODE:
    - If STRICT_MODE is True: raises exceptions on failure
    - If STRICT_MODE is False: returns graceful error messages as TextContent

    Args:
        file_path: The local path to the media file.

    Returns:
        A BinaryContent object containing the binary data, or a TextContent
        object with an error message if STRICT_MODE is False and file is not found.

    Raises:
        FileNotFoundError: If the file does not exist (only when STRICT_MODE is True).
        IOError: If the file cannot be read (only when STRICT_MODE is True).
    """
    if not os.path.exists(file_path):
        if STRICT_MODE:
            raise FileNotFoundError(f"File not found at path: {file_path}")
        else:
            return f"Error: File not found at path '{file_path}'. Please check the file path and try again."

    logger.debug(f"[tool] Read media file: {file_path}")

    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(file_path, "rb") as f:
            raw_content = f.read()

        return BinaryContent(data=raw_content, media_type=mime_type)

    except (IOError, OSError) as e:
        if STRICT_MODE:
            raise IOError(f"Could not read file: {e}") from e
        else:
            return f"Error: Could not read file '{file_path}': {e}. Please check file permissions and try again."