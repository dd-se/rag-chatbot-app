#!/usr/bin/env python3
import os
from typing import Iterator

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .logging_helper import get_logger
from .models import EvalResponse

load_dotenv()
try:
    api_key = os.environ["GEMINI_API_KEY"]
except KeyError:
    raise RuntimeError("Missing required environment variable: GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

logger = get_logger(__name__)
cutoff = slice(0, 50)

EVAL_SYSTEM_PROMPT = """
You are an impartial evaluation system. Your task is to assess the AI assistant's answer compared to the ideal answer.

Scoring:
- 1.0: The answer is very close to the ideal answer.
- 0.5: The answer is partially correct or incomplete.
- 0: The answer is incorrect or irrelevant.

Assign only one of these scores (0, 0.5, or 1.0) and briefly justify your decision.
"""

CONTEXT_SYSTEM_PROMPT = """
I will ask you a question, and I want you to answer
based only on the context I provide, and no other information.
If there is not enough information in the context to answer the question,
say "I don't know". Do not try to guess."""
CONTEXT_PROMPT_TEMPLATE = """
Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the question below:
{question}
"""

REFINED_QUESTION_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant."""
REFINED_QUESTION_PROMPT_TEMPLATE = """
Chat History:
---------------------
{chat_history}
---------------------
Follow Up Question: {question}
Given the above conversation and a follow up question, rephrase the follow up question to be a standalone question.
Standalone question:
"""


def create_embeddings(text: list[str], model="text-embedding-004"):
    logger.debug(f"{len(text) = } | {model = }")
    return client.models.embed_content(model=model, contents=text, config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"))


def refined_question_response(question: str, chat_history: list[dict[str, str]], model="gemini-2.0-flash") -> types.GenerateContentResponse:
    logger.debug(f"{question[cutoff] = } | {len(chat_history) = } | {model = }")
    chat_history = "\n".join(f"{message['role']}: {message['content_mod']}" if message["role"] == "user" else f"{message['role']}: {message['content']}" for message in chat_history)

    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=REFINED_QUESTION_SYSTEM_PROMPT,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_output_tokens=512,
        ),
        contents=REFINED_QUESTION_PROMPT_TEMPLATE.format(chat_history=chat_history, question=question),
    )
    logger.info("Response generated successfully.")
    logger.debug(f"Refined question response: {response.text.strip('\n')}")
    return response


def context_aware_response(question: str, context: list[str], model="gemini-2.0-flash") -> types.GenerateContentResponse:
    logger.debug(f"{question[cutoff] = } | {len(context) = } | {model = }")
    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=CONTEXT_SYSTEM_PROMPT,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_output_tokens=512,
        ),
        contents=CONTEXT_PROMPT_TEMPLATE.format(context=" ".join(context), question=question),
    )
    logger.info("Response generated successfully.")
    logger.debug(f"Response: {response.text[cutoff]}")
    return response


def context_aware_response_stream(question: str, context: list[str], model="gemini-2.0-flash") -> Iterator[types.GenerateContentResponse]:
    logger.debug(f"{question[cutoff] = } | {len(context) = } | {model = }")
    stream = client.models.generate_content_stream(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=CONTEXT_SYSTEM_PROMPT,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_output_tokens=512,
        ),
        contents=CONTEXT_PROMPT_TEMPLATE.format(context=" ".join(context), question=question),
    )
    logger.info("Response generated successfully.")
    return stream


def generate_eval_response(question: str, ai_answer: str, ideal_answer: str, model="gemini-2.0-flash") -> types.GenerateContentResponse:
    logger.debug(f"{question[cutoff] = } | {ai_answer[cutoff] = } | {ideal_answer[cutoff] = } | {model = }")
    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=EVAL_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=EvalResponse,
            temperature=0.1,
            max_output_tokens=256,
        ),
        contents=f"Question: {question}\nAI assistant's answer: {ai_answer}\nIdeal answer: {ideal_answer}",
    )
    logger.debug(f"{response.parsed.evaluation[cutoff]}... | Score: {response.parsed.score} ")
    logger.info("Response generated successfully.")
    return response
