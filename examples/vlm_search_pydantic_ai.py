"""
TODO: explain the purpose of this example
"""

import mimetypes
import re
import requests
from pathlib import Path
from urllib.parse import urlparse

# Params
API_BASE_URL = "http://localhost:2692"
OUTPUT_DIR = Path(__file__).parent / "vlm_search"


def download_image(url, output_dir="images", name=None):
    filename = name or Path(urlparse(url).path).name or "image"
    response = requests.get(url)
    mime_type = response.headers.get("content-type")
    if mime_type:
        mime_type = mime_type.split(";")[0].strip()
        ext = mimetypes.guess_extension(mime_type)
        if ext:
            filename = Path(filename).with_suffix(ext).name
    local_path = Path(output_dir) / filename
    if not local_path.exists():
        local_path.parent.mkdir(exist_ok=True)
        local_path.write_bytes(response.content)
    relative_path = local_path.relative_to(output_dir)
    return relative_path


def flatten_markdown_images(markdown_content, output_dir: Path = OUTPUT_DIR):
    def replacer(match):
        alt_text, url = match.groups()
        local_path = download_image(url, output_dir, name=alt_text)
        return f"![{alt_text}]({local_path})"

    return re.sub(r"!\[([^\]]*)\]\((http[^)]+)\)", replacer, markdown_content)


def generate_report(query_text, output_dir: Path = OUTPUT_DIR):
    # Search for objects
    objects_response = requests.get(
        f"{API_BASE_URL}/api/v0/objects/search",
        params={"query": query_text},
    )
    print(f"Query: {query_text}. Response: {objects_response}")
    objects = objects_response.json()

    # Create report (with thumbnails and metadata to help VLM's decision making)
    report_response = requests.get(
        f"{API_BASE_URL}/api/v0/objects/report",
        params={"object_uids": [object["uid"] for object in objects]},
    )
    report = report_response.json()
    print("Generated report")

    # Transform thumbnail URLs into local paths
    report = flatten_markdown_images(report, output_dir)
    print("Transformed thumbnail URLs into local paths")
    return report


def save_report(report, output_dir: Path = OUTPUT_DIR):
    # Save the markdown report to a file
    output_file = output_dir / "output_search_report.md"
    with open(output_file, "w") as f:
        f.write(report)
    print(f"Saved search report to {output_file}")


def main():
    report = generate_report(query_text="a blue car")
    save_report(report)


if __name__ == "__main__":
    main()
