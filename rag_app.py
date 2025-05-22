import streamlit as st

from shared import *

st.set_page_config(layout="wide")
st.sidebar.header("Conversational AI with Document Context")
st.markdown(css, unsafe_allow_html=True)

if "doc_name" not in st.session_state:
    # Name of the currently selected document/context
    st.session_state.doc_name = None
if "doc_hash" not in st.session_state:
    # Hash identifier for the selected document/context
    st.session_state.doc_hash = None
if "qa_list" not in st.session_state:
    # Uploaded QA list for evaluating the AI-Assistant
    st.session_state.qa_list: list[QAItem] = None  # type: ignore
if "messages" not in st.session_state:
    # Stores the chat history for the current session
    st.session_state.messages = []
if "eval_results" not in st.session_state:
    # Stores evaluation results for the current session
    st.session_state.eval_results = []


def display_chat_history(messages: list[dict], noop: bool = False):
    if noop:
        return
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def display_qa_results(data: list[dict]):
    """Present the results in tabular form"""
    st.write("## Results")
    st.dataframe(
        data,
        use_container_width=True,
        column_config={"score": st.column_config.ProgressColumn(format="plain")},
    )
    st.write("### Evaluation Score:", round(sum(r["score"] for r in data) / len(data), 2))


def evaluate_ai(data: list[QAItem]):
    """Evaluates AI-generated responses for a list of question-answer items."""
    p = st.sidebar.progress(0)
    for i, qa_item in enumerate(data, 1):
        p.progress(i / len(data), "Running...")
        question = qa_item.question
        ideal_answer = qa_item.ideal_answer
        # Get relevant chunks using similarity search
        query_embedding = create_embeddings([question])[0].values
        top_chunks = get_relevant_context(
            query_embedding,
            st.session_state.doc_hash,
        )["documents"][0]
        # Get a response from the AI-Assistant
        response = context_aware_response(
            question,
            top_chunks,
            st.session_state.temperature,
            st.session_state.max_tokens,
        ).text
        # Ask the AI-Assistant to rate the response
        eval: EvalResponse = generate_eval_response(
            question,
            response,
            ideal_answer,
        ).parsed
        eval.question = question
        eval.context = st.session_state.doc_name
        eval.hash = st.session_state.doc_hash
        st.session_state.eval_results.append(eval.model_dump())
    p.empty()


def process_pdf_or_json_file():
    if st.session_state.file:
        file = st.session_state.file
        if file.type == "application/pdf":
            doc_hash = get_document_hash(file)
            if not is_in_db(doc_hash):
                with st.spinner("Please wait...", show_time=True):
                    text = load_pdf_data(file)
                    chunks = chunk_text(text)
                    fname = f"{file.name}-{random_letters()}"
                    doc_hash = process_and_store_document_chunks(chunks, fname, doc_hash)
                    if doc_hash:
                        current_docs[fname] = doc_hash
                        st.toast("Document processed.", icon="ℹ️")
            else:
                st.toast("Document already processed.", icon="ℹ️")
        else:
            st.session_state.qa_list = qa_list_adapter.validate_json(file.read())
            st.toast("JSON processed.", icon="ℹ️")


st.sidebar.file_uploader(
    "Select a file to upload...",
    type=["pdf", "json"],
    key="file",
    help="Upload a PDF document and start ask questions about its content.  \
        \nUpload a Questions & Answers file in JSON format to evaluate the AI-Assistant.",
    on_change=process_pdf_or_json_file,
)

st.session_state.doc_name = st.sidebar.radio(
    "Select a document to use as context:",
    current_docs.keys(),
    help="Choose one of the processed documents.  \
        \nYour questions will be answered based on the selected document content.",
    on_change=lambda: (
        st.session_state.messages.clear(),
        st.toast(
            "Context changed. Chat history reset.",
            icon="ℹ️",
        ),
    ),
)
st.session_state.doc_hash = current_docs.get(st.session_state.doc_name)

col_1, col_2 = st.sidebar.columns(2)
col_1.button(
    "Evaluate AI",
    help="Upload a valid QA file to enable this button",
    on_click=evaluate_ai,
    kwargs={"data": st.session_state.qa_list},
    disabled=any((st.session_state.doc_hash is None, st.session_state.qa_list is None)),
    use_container_width=True,
)
col_1.button(
    "Clear Chat History",
    on_click=lambda: (
        st.session_state.messages.clear(),
        st.toast("Chat history reset.", icon="ℹ️"),
    ),
    use_container_width=True,
)
col_2.button(
    "QA Results",
    key="qa_button_pressed",
    help="Click to display evaluation results after running 'Evaluate AI'.",
    on_click=display_qa_results,
    kwargs={"data": st.session_state.eval_results},
    disabled=not st.session_state.eval_results,
    use_container_width=True,
)
col_2.button(
    "Display Chat History",
    use_container_width=True,
)
# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/content-generation-parameters#temperature
st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    help="Lower temperatures lead to more predictable results.",
    key="temperature",
    step=0.1,
)
st.sidebar.slider(
    "Max Output Tokens",
    min_value=512,
    max_value=2048,
    value=1024,
    help="Maximum number of tokens that can be generated in the response.",
    key="max_tokens",
    step=512,
)
# Accept user input when doc_hash is defined
prompt = st.chat_input(disabled=st.session_state.doc_hash is None)
display_chat_history(st.session_state.messages, st.session_state.qa_button_pressed)
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.messages:
        # Refine user question using recent chat history
        refined_prompt = refined_question_response(prompt, st.session_state.messages[-4:]).text
    prompt_potentially_enhanced = locals().get("refined_prompt") or prompt
    query_embedding = create_embeddings([prompt_potentially_enhanced])[0].values
    top_chunks = get_relevant_context(query_embedding, doc_hash=st.session_state.doc_hash)["documents"][0]

    with st.chat_message("assistant"):
        response = st.write_stream(
            chunk.text
            for chunk in context_aware_response_stream(
                prompt_potentially_enhanced,
                top_chunks,
                st.session_state.temperature,
                st.session_state.max_tokens,
            )
        )

    st.session_state.messages.append({"role": "user", "content": prompt, "content_mod": prompt_potentially_enhanced})
    st.session_state.messages.append({"role": "assistant", "content": response})
