import argparse
import csv
from datetime import datetime
from pathlib import Path
from time import sleep

import pandas as pd

from .genai.genai_client import (
    client,
    context_aware_response,
    context_aware_response_stream,
    create_embeddings,
    generate_eval_response,
    refined_question_response,
)
from .genai.models import EvalResponse, QAItem, qa_list_adapter
from .logging_helper import get_logger
from .pdf_loader.chunker import fixed_size_chunker, load_and_chunk_pdf_data
from .vector_store.db_client import (
    collection,
    current_docs,
    delete_document,
    get_doc_name_by_hash,
    get_document_hash,
    get_relevant_context,
    is_in_db,
    process_and_store_document_chunks,
    random_letters,
)

bg_img_url = "https://i.imgur.com/6yLAgLv.jpeg"
css = f"""
    <style>
    .stChatMessage {{
        background: #262730ee;
        border-radius: 15px;
        padding: 16px;
    }}
   .stApp {{
        background-image: url("{bg_img_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;

    }}
    body {{
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
    }}
    </style>
    """
