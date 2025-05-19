import io

from pypdf import PdfReader

from .logging_helper import get_logger

logger = get_logger(__name__)


def load_pdf_data(content: io.BufferedReader):
    logger.debug(f"{content.name = }")
    reader = PdfReader(content)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    logger.info("Text extracted successfully.")
    return text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    logger.debug(f"{len(text) =  } | {len(chunks) = } | {chunk_size = } | {overlap = }")
    logger.info("Text splitted into chunks successfully.")
    return chunks
