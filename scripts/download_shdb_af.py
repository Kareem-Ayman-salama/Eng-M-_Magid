"""Download SHDB-AF files from PhysioNet with resume support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class DownloadConfig:
    """SHDB-AF download configuration."""

    base_url: str
    output_dir: Path
    chunk_size: int = 1024 * 1024


def discover_files(base_url: str) -> list[tuple[str, str]]:
    """Discover downloadable files from PhysioNet project page.

    Args:
        base_url: PhysioNet content page URL.

    Returns:
        Pairs of filename and file URL.
    """

    response = requests.get(base_url, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    files: list[tuple[str, str]] = []
    for anchor in soup.find_all("a"):
        href = anchor.get("href")
        if not href or "/files/shdb-af/1.0.1/" not in href or "?download" not in href:
            continue
        url = urljoin(base_url, href)
        filename = Path(href.split("?")[0]).name
        files.append((filename, url))
    unique: dict[str, str] = {}
    for filename, url in files:
        unique[filename] = url
    return sorted(unique.items())


def remote_size(url: str) -> int | None:
    """Return remote file size if available."""

    response = requests.head(url, allow_redirects=True, timeout=60)
    if response.status_code >= 400:
        return None
    value = response.headers.get("content-length")
    return int(value) if value and value.isdigit() else None


def download_file(filename: str, url: str, output_dir: Path, chunk_size: int) -> None:
    """Download one file with resume support."""

    target = output_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = remote_size(url)
    existing_size = target.stat().st_size if target.exists() else 0
    if expected_size is not None and existing_size == expected_size:
        print(f"SKIP {filename} ({existing_size} bytes)")
        return

    headers = {}
    mode = "wb"
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"
    print(f"DOWNLOAD {filename} from {existing_size} / {expected_size or 'unknown'}")
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        if response.status_code == 416:
            print(f"SKIP {filename} already complete")
            return
        response.raise_for_status()
        with target.open(mode + "") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)


def main() -> None:
    """Download all SHDB-AF files."""

    root = Path.cwd()
    config = DownloadConfig(
        base_url="https://physionet.org/content/shdb-af/1.0.1/",
        output_dir=root / "data" / "shdb_af",
    )
    files = discover_files(config.base_url)
    (config.output_dir / "download_manifest.txt").write_text(
        "\n".join(filename for filename, _ in files),
        encoding="utf-8",
    )
    print(f"Discovered {len(files)} files")
    for filename, url in files:
        download_file(filename, url, config.output_dir, config.chunk_size)
    print("Done")


if __name__ == "__main__":
    main()
