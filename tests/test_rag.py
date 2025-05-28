# Add example RAG tests for demonstration purposes
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
qa_file = "example/test.json"
pdf_document = "example/test.pdf"
eval_model = GeminiModel(model_name="gemini-2.0-flash-lite", api_key=os.environ["GEMINI_API_KEY"])
answer_relevancy = AnswerRelevancyMetric(model=eval_model)
faithfulness = FaithfulnessMetric(model=eval_model)

with open(qa_file, encoding="utf-8") as f:
    qa_list = qa_list_adapter.validate_json(f.read())
with open(pdf_document, "rb") as f:
    doc_hash = get_document_hash(f)
    assert is_in_db(doc_hash) is True, f"Please add the document {pdf_document} to the vector store before proceeding with the tests."


@pytest.mark.parametrize("qa", qa_list)
def test_rag(qa: QAItem):
    embedding = create_embeddings([qa.question])[0].values
    top_chunks = get_relevant_context(embedding, doc_hash)
    response = context_aware_response(qa.question, top_chunks).text
    test_case = LLMTestCase(
        input=qa.question,
        actual_output=response,
        expected_output=qa.ideal_answer,
        retrieval_context=top_chunks,
    )
    assert_test(test_case, [answer_relevancy, faithfulness], run_async=False)
