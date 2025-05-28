import io
from pathlib import Path

import regex
from pypdf import PdfReader

from ..logging_helper import get_logger
from .abbreviations import abbreviations, reversed_abbreviations

logger = get_logger(__name__)


def load_and_chunk_pdf_data(content: io.BufferedReader | Path) -> list[str]:
    """Extracts text from a PDF, cleans it, and splits it into sentence-level chunks."""
    reader = PdfReader(content)
    sentences = []
    for i, page in enumerate(reader.pages, 1):
        raw_text = page.extract_text()
        # Skip pages with little text
        if not raw_text or len(raw_text.strip()) < 50:
            logger.debug(f"Skipped page {i} due to insufficient text length.")
            continue
        # Replace smart double quotes
        text = regex.sub(r"[“”]", '"', raw_text)
        # Replace smart single quotes
        text = regex.sub(r"[‘’]", "'", text)
        # Replace en dash (–) with a standard hyphen (-)
        text = regex.sub(r"–", "-", text)
        # Keep latin letters, digits, whitespace, and various chars.
        text = regex.sub(r"[^\p{Latin}\d\s.,!?'\":;/\[\]&\-+()@]", "", text)
        # Remove line break hyphenation (co-\noperation -> cooperation)
        text = regex.sub(r"-\s+", "", text)
        # Remove spacing around punctuation (word ! -> word!)
        text = regex.sub(r'\s([?.!"](?:\s|$))', r"\1", text)
        # Remove numbers followed by a newline
        text = regex.sub(r"\d+\n", "", text)
        # Replace abbreviations with tokens to avoid splitting them
        for abbr, token in abbreviations.items():
            text = text.replace(abbr, token)
        # Splits a text into sentences (ending with ., !, ?...  followed by whitespace)
        for sentence in regex.split(r'(?<=[.!?]["\')\]]?)\s+', text):
            # Replace tokens back to abbreviations
            for token, abbr in reversed_abbreviations.items():
                sentence = sentence.replace(token, abbr)
            # Remove whitespace (newlines, multiple spaces)
            sentence = regex.sub(r"\s+", " ", sentence).lower().strip()
            if not sentence or sentence.isdigit():
                continue
            # If sentence is short, append to previous sentence
            if len(sentence) < 40 and sentences:
                sentences[-1] = f"{sentences[-1]}{'' if sentences[-1].endswith(('.', '!', '?')) else '.'} {sentence}"
            elif sentence not in sentences:
                sentences.append(sentence)
    logger.info(f"Text extracted, cleaned, and sentence-level chunked({len(sentences)}) successfully.")
    return sentences


def fixed_size_chunker(text: str, chunk_size: int = 1024, overlap: int = 200) -> list[str]:
    """Splits the input text into overlapping fixed-size chunks."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    logger.debug(f"{len(text) =  } | {len(chunks) = } | {chunk_size = } | {overlap = }")
    logger.info("Text splitted into chunks successfully.")
    return chunks
