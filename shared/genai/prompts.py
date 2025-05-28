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
