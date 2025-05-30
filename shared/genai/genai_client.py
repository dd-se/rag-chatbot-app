#!/usr/bin/env python3
import os
from typing import Iterator, Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types

from ..logging_helper import get_logger
from .models import EvalResponse
from .prompts import *

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key is None:
    raise RuntimeError("Missing required environment variable: GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
logger = get_logger(__name__)
cut = slice(0, 50)


def create_embeddings(
    chunks: list[str],
    task_type: Literal["SEMANTIC_SIMILARITY", "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"] = "SEMANTIC_SIMILARITY",
    batch_size: int = 100,
    model: str = "text-embedding-004",
):
    # Split into chunks of 100 as Google only allows 100 maximum per request
    logger.debug(f"{len(chunks) = } | {model = }")
    batch_number = 1
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = client.models.embed_content(
            model=model,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        ).embeddings
        all_embeddings.extend(embeddings)
        logger.info(f"Batch {batch_number} processed ({len(embeddings)})")
        batch_number += 1
    return all_embeddings


def refined_question_response(
    question: str,
    chat_history: list[dict[str, str]],
    model="gemini-2.0-flash",
) -> types.GenerateContentResponse:
    logger.debug(f"{question[cut] = } | {len(chat_history) = } | {model = }")
    chat_history = "\n\n".join(f"{m['role']}: {m['content']}" for m in chat_history)
    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(system_instruction=REFINED_QUESTION_SYSTEM_PROMPT),
        contents=REFINED_QUESTION_PROMPT_TEMPLATE.format(chat_history=chat_history, question=question),
    )
    logger.debug(f"Refined question response: {response.text.strip('\n')}")
    logger.info("Response generated successfully.")
    return response


def context_aware_response(
    question: str,
    context: list[str],
    temperature: float = 0.7,
    max_output_tokens: int = 1024,
    model="gemini-2.0-flash",
) -> types.GenerateContentResponse:
    logger.debug(f"{question[cut] = } | {len(context) = } | {model = }")
    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=CONTEXT_SYSTEM_PROMPT,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
        contents=CONTEXT_PROMPT_TEMPLATE.format(context="\n".join(context), question=question),
    )
    logger.info("Response generated successfully.")
    return response


def context_aware_response_stream(
    question: str,
    context: list[str],
    temperature: float = 0.7,
    max_output_tokens: int = 1024,
    model="gemini-2.0-flash",
) -> Iterator[types.GenerateContentResponse]:
    logger.debug(f"{question[cut] = } | {len(context) = } | {model = }")
    stream = client.models.generate_content_stream(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=CONTEXT_SYSTEM_PROMPT,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
        contents=CONTEXT_PROMPT_TEMPLATE.format(context="\n".join(context), question=question),
    )
    logger.info("Response generated successfully.")
    return stream


def generate_eval_response(
    question: str,
    ai_answer: str,
    ideal_answer: str,
    temperature: float = 0.7,
    max_output_tokens: int = 1024,
    model="gemini-2.0-flash-lite",
) -> types.GenerateContentResponse:
    logger.debug(f"{question[cut] = } | {ai_answer[cut] = } | {ideal_answer[cut] = } | {model = }")
    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=EVAL_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=EvalResponse,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
        contents=f"Question: {question}\nAI assistant's answer: {ai_answer}\nIdeal answer: {ideal_answer}",
    )
    logger.debug(f"{response.parsed.evaluation[cut]}... | Score: {response.parsed.score} ")
    logger.info("Response generated successfully.")
    return response
