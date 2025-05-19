import argparse
import csv
from datetime import datetime
from pathlib import Path
from time import sleep

import pandas as pd

from .db import (
    collection,
    current_docs,
    get_document_hash,
    get_relevant_context,
    is_in_db,
    process_and_store_document_chunks,
    random_letters,
)
from .genai import (
    client,
    context_aware_response,
    context_aware_response_stream,
    create_embeddings,
    generate_eval_response,
    refined_question_response,
)
from .logging_helper import get_logger
from .models import EvalResponse, QAItem, qa_list_adapter
from .parse_pdf import chunk_text, load_pdf_data

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
