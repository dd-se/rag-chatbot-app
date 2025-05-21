#!/usr/bin/env python3
from shared import *

DOC_PROCESSED = "Document processed."
DOC_ALREADY_PROCESSED = "Document already processed."
ANSWER = "Answer"
DOC_NOT_FOUND = "Document not found in the database. Please add it first using the 'add' command."

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="RAG Chatbot CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add PDF to document store")
    add_parser.add_argument("pdf", type=str, help="PDF filename")

    query_parser = subparsers.add_parser("query", help="Ask the AI-assistant about the document's contents")
    query_parser.add_argument("pdf", type=str, help="PDF filename")
    query_parser.add_argument("question", type=str, help="Question to ask")

    eval_parser = subparsers.add_parser("eval", help="Evaluate using loaded embeddings")
    eval_parser.add_argument("pdf", type=str, help="PDF filename")
    eval_parser.add_argument("validation_data", type=str, help="Path to validation data JSON file")
    eval_parser.add_argument("--output", type=str, default=f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", help="Output CSV filename")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.pdf, "rb") as doc:
        doc_hash = get_document_hash(doc)
        in_db = is_in_db(doc_hash)
        if args.command == "add":
            if not in_db:
                text = load_pdf_data(doc)
                chunks = chunk_text(text)
                fname = f"{Path(args.pdf).name}-{random_letters()}"
                process_and_store_document_chunks(chunks, fname, doc_hash)
                logger.info(DOC_PROCESSED)
            else:
                logger.info(DOC_ALREADY_PROCESSED)

        elif args.command == "query":
            if in_db:
                query_embedding = create_embeddings([args.question])[0].values
                top_chunks = get_relevant_context(query_embedding, doc_hash)["documents"][0]
                response = context_aware_response(args.question, top_chunks).text
                logger.info(f"{ANSWER}: {response}")
            else:
                logger.info(DOC_NOT_FOUND)
        elif args.command == "eval":
            if in_db:
                with open(args.validation_data, encoding="utf-8") as f:
                    validation_data = qa_list_adapter.validate_json(f.read())

                with open(args.output, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=EvalResponse.model_fields.keys())
                    writer.writeheader()
                    for item in validation_data:
                        question = item.question
                        ideal_answer = item.ideal_answer

                        query_embedding = create_embeddings([question])[0].values
                        top_chunks = get_relevant_context(query_embedding, doc_hash)["documents"][0]
                        response = context_aware_response(question, top_chunks).text

                        eval: EvalResponse = generate_eval_response(question, response, ideal_answer).parsed
                        eval.question = question
                        eval.context = next(k for k, v in current_docs.items() if v == doc_hash)
                        eval.hash = doc_hash
                        writer.writerow(eval.model_dump())
            else:
                logger.info(DOC_NOT_FOUND)


if __name__ == "__main__":
    main()
