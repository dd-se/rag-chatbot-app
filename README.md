# RAG Chatbot App and CLI

A Conversational AI Assistant with Document Context (Retrieval-Augmented Generation)

## Getting Started

### Prerequisites
- Python 3.10+
- Install dependencies with pip or uv:
  ```bash
  pip install -r requirements.txt
  ```
  ```bash
  uv sync
  ```
- Follow the official guide from [Google AI](https://ai.google.dev/gemini-api/docs/api-key) on how to setup your own private API key.

### Running the Streamlit App
```bash
streamlit run rag_app.py
```
- Upload a PDF to process it.
- Select the document in the sidebar.
- Ask questions in the chat input.

### Using the CLI
```bash
python rag_cli.py add <pdf_file>
python rag_cli.py query <pdf_file> "Your question here"
python rag_cli.py eval <pdf_file> <qa_file.json> [--output <results.csv>]
```

## Evaluation
- Prepare a JSON file with a list of questions and answers. (see example folder)
- Upload via the web app or use the CLI `eval` command.
- Results are shown in the app or saved as CSV via CLI.

## License
MIT License


