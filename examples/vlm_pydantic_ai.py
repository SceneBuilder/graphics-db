import re
import requests
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

# Params
API_BASE_URL = "http://localhost:2692"


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


def process_markdown_images(markdown_content, output_dir="images"):
    def replacer(match):
        alt_text, url = match.groups()
        local_path = download_image(url, output_dir, name=alt_text)
        return f"![{alt_text}]({local_path})"

    return re.sub(r"!\[([^\]]*)\]\((http[^)]+)\)", replacer, markdown_content)


def main(query_text, output_dir: Path = Path(__file__).parent):
    assets_response = requests.get(
        f"{API_BASE_URL}/api/v0/assets/search",
        params={"query": query_text},
    )
    print(f"Query: {query_text}. Response: {assets_response}")
    assets = assets_response.json()

    report_response = requests.get(
        f"{API_BASE_URL}/api/v0/assets/report",
        params={"asset_uids": [asset["uid"] for asset in assets]},
    )
    report = report_response.json()
    print("Generated report")

    # Transform thumbnail URLs into local paths
    report = process_markdown_images(report, output_dir)
    print("Transformed thumbnail URLs into local paths")

    # Save the markdown report to a file
    output_file = output_dir / "output_search_report.md"
    with open(output_file, "w") as f:
        f.write(report)
    print(f"Saved search report to {output_file}")


if __name__ == "__main__":
    main("a blue car")
