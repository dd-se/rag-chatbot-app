import io
import re
from pathlib import Path

from pypdf import PdfReader

from .logging_helper import get_logger

logger = get_logger(__name__)


def load_pdf_data(content: io.BufferedReader | Path) -> str:
    """Extracts and cleans text from a PDF file."""
    reader = PdfReader(content)
    cleaned_text_parts = []
    for i, page in enumerate(reader.pages, 1):
        try:
            raw_text = page.extract_text()
            # Skip pages with little text
            if not raw_text or len(raw_text.strip()) < 50:
                continue
        except Exception:
            logger.error(f"Error extracting text from page {i}: {Exception}")
            continue
        # Remove non-ASCII characters
        text = raw_text.encode("ascii", "ignore").decode()
        # Remove line break hyphenation (co-\noperation → cooperation)
        text = re.sub(r"-\s+", "", text)
        # Remove whitespace (newlines, multiple spaces)
        text = re.sub(r"\s+", " ", text)
        # Remove spacing around punctuation (word ! → word!)
        text = re.sub(r'\s([?.!"](?:\s|$))', r"\1", text)
        text = text.lower()
        cleaned_text_parts.append(text.strip())
    logger.info("Text extracted from the document successfully.")
    return " ".join(cleaned_text_parts)


def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 200) -> list[str]:
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    logger.debug(f"{len(text) =  } | {len(chunks) = } | {chunk_size = } | {overlap = }")
    logger.info("Text splitted into chunks successfully.")
    return chunks
