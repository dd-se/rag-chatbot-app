# Adapted from https://github.com/meteatamel/genai-beyond-basics/blob/main/samples/evaluation/deepeval/rag_eval/test_rag_triad_cymbal.py
# > deepeval test run tests/test_rag.py
import os

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

from shared import *

logger = get_logger(__name__)
eval_model = GeminiModel(model_name="gemini-2.0-flash-lite", api_key=os.environ["GEMINI_API_KEY"])
answer_relevancy = AnswerRelevancyMetric(model=eval_model)
faithfulness = FaithfulnessMetric(model=eval_model)

with open("example/test.json", encoding="utf-8") as f:
    qa_list = qa_list_adapter.validate_json(f.read())
with open("example/test.pdf", "rb") as f:
    doc_hash = get_document_hash(f)


@pytest.mark.parametrize("qa", qa_list)
def test_rag(qa: QAItem):
    embedding = create_embeddings([qa.question])[0].values
    top_chunks = get_relevant_context(embedding, doc_hash)["documents"][0]
    response = context_aware_response(qa.question, top_chunks).text

    test_case = LLMTestCase(
        input=qa.question,
        actual_output=response,
        expected_output=qa.ideal_answer,
        retrieval_context=top_chunks,
    )
    logger.info(f"Evaluating: {qa.question[:20]}")
    assert_test(test_case, [answer_relevancy, faithfulness], run_async=False)
