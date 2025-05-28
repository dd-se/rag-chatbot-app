#!/usr/bin/env python3
import hashlib
import io
import random
import string
from pathlib import Path
from typing import Any

import chromadb

from ..genai.genai_client import create_embeddings
from ..logging_helper import get_logger

logger = get_logger(__name__)
chroma_client = chromadb.PersistentClient(path=str(Path(__file__).parent / "data"))
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


def get_doc_name_by_hash(doc_hash: str):
    return next(k for k, v in current_docs.items() if v == doc_hash)


def delete_document(doc_hash: str, doc_name: str = None):
    """Deletes all document chunks with the given hash from the collection."""
    collection.delete(where={"hash": doc_hash})
    if not doc_name:
        doc_name = get_doc_name_by_hash(doc_hash)
    del current_docs[doc_name]
    logger.info(f"Document {doc_name} with hash {doc_hash} deleted from store.")


def get_relevant_context(query_embedding: list[float], doc_hash: str = None, k: int = 5, sort_by_id: bool = False):
    """Retrieves relevant document chunks having a specific hash"""
    logger.debug(f"{len(query_embedding)= } | {k = } | {doc_hash = }")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"hash": doc_hash} if doc_hash else None,
        include=["documents", "metadatas"],
    )
    logger.info("Context retrieved successfully.")
    docs: list[str] = results["documents"][0]
    if sort_by_id:
        meta = results["metadatas"][0]
        reranked = sorted(zip(docs, meta), key=lambda x: x[1]["chunk_id"])
        docs = [doc for doc, _ in reranked]
        logger.info("Sorted context successfully by chunk ID.")
    return docs


def process_and_store_document_chunks(chunks: list[str], filename: str, doc_hash: str):
    """Processes document chunks, generates embeddings, and stores them in the ChromaDB collection."""
    logger.debug(f"{len(chunks) = } | {filename = } | {doc_hash = }")
    embeddings = create_embeddings(chunks)
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i].values],
            metadatas=[{"source": filename, "chunk_id": i, "hash": doc_hash}],
            ids=[f"{doc_hash}_{i}"],
        )
    logger.info(f"{filename} added to the vector store.")
    return doc_hash
