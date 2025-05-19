#!/usr/bin/env python3
import hashlib
import io
import random
import string
from pathlib import Path
from typing import Any

import chromadb

from .genai import create_embeddings
from .logging_helper import get_logger

logger = get_logger(__name__)
chroma_client = chromadb.PersistentClient(path=str(Path(__file__).parent / "chroma_db"))
collection = chroma_client.get_or_create_collection("documents", metadata={"hnsw:space": "cosine"})
current_docs = {m["source"]: m["hash"] for m in collection.get(include=["metadatas"])["metadatas"]}


def random_letters(n=2):
    return "".join(random.choices(string.ascii_lowercase, k=n))


def get_document_hash(doc: io.BufferedReader | Any):
    """Generates a hash for a file"""
    hasher = hashlib.md5()
    hasher.update(doc.read())
    hash = hasher.hexdigest()
    logger.debug(f"{doc.name = } | {hash = }")
    return hash


def is_in_db(doc_hash: str):
    """Checks if any document chunks with the given hash exist in the collection."""
    in_db = len(collection.get(where={"hash": doc_hash})["ids"]) > 0
    logger.debug(f"{doc_hash = } | {in_db = }")
    return in_db


def get_relevant_context(query_embedding: list[float], doc_hash: str = None, k: int = 7):
    """Retrieves relevant document chunks having a specific hash"""
    logger.debug(f"{len(query_embedding)= } | {k = } | {doc_hash = }")
    where_clause = {"hash": doc_hash} if doc_hash else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_clause,
    )
    logger.info("Context retrieved successfully from document store.")
    return results


def process_and_store_document_chunks(chunks: list[str], filename: str, doc_hash: str):
    """Processes document chunks, generates embeddings, and stores them in the ChromaDB collection."""
    logger.debug(f"{len(chunks) = } | {filename = } | {doc_hash = }")
    embeddings = create_embeddings(chunks)
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings.embeddings[i].values],
            metadatas=[{"source": filename, "chunk_id": i, "hash": doc_hash}],
            ids=[f"{doc_hash}_{i}"],
        )
    logger.info(f"{filename} added to document store.")
    return doc_hash
