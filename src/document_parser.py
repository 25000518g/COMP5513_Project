"""Download and extract text from supported announcement documents."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re


def normalize_text(text: str, max_characters: int = 20_000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_characters]


def extract_html_text(content: str, max_characters: int = 20_000) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("Install beautifulsoup4 to parse HTML announcements.") from exc
    soup = BeautifulSoup(content, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    text = normalize_text(soup.get_text(" ", strip=True), max_characters)
    if not text:
        raise ValueError("The HTML announcement did not contain readable text.")
    return text


def extract_pdf_text(content: bytes, max_characters: int = 20_000) -> str:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("Install pdfplumber to parse PDF announcements.") from exc
    try:
        with pdfplumber.open(BytesIO(content)) as pdf:
            text = " ".join((page.extract_text() or "") for page in pdf.pages)
    except Exception as exc:  # pdfplumber raises several format-specific exceptions.
        raise ValueError(f"The PDF could not be read: {exc}") from exc
    text = normalize_text(text, max_characters)
    if not text:
        raise ValueError("The PDF contains no extractable text; OCR may be required.")
    return text


def extract_file(path: str | Path, max_characters: int = 20_000) -> str:
    """Extract text from a previously saved PDF or HTML announcement."""

    file_path = Path(path)
    content = file_path.read_bytes()
    if file_path.suffix.lower() == ".pdf":
        return extract_pdf_text(content, max_characters)
    return extract_html_text(content.decode("utf-8", errors="replace"), max_characters)


def download_and_extract(url: str, timeout: int = 15, save_path: str | Path | None = None) -> str:
    """Download an announcement, optionally save it, and extract its text."""

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Install requests to download announcements.") from exc
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "COMP5513-StockAnalysis/1.0"})
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Announcement download failed: {exc}") from exc
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return extract_file(path)
    content_type = response.headers.get("content-type", "").lower()
    if "pdf" in content_type or url.lower().split("?", 1)[0].endswith(".pdf"):
        return extract_pdf_text(response.content)
    return extract_html_text(response.text)
