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


@st.cache_data
def display_chat_history(messages: list[str], noop: bool = False):
    if noop:
        return
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


@st.cache_data
def display_qa_results(data: list[dict] | pd.DataFrame):
    """Present the results in tabular form"""
    st.header("Results")
    st.dataframe(data, use_container_width=True)


def evaluate_ai(data: list[QAItem]):
    p = st.sidebar.progress(0)
    for i, qa_item in enumerate(data, 1):
        p.progress(i / len(data), "Running...")
        question = qa_item.question
        ideal_answer = qa_item.ideal_answer
        # Get relevant chunks using similarity search
        query_embedding = create_embeddings([question])[0].values
        top_chunks = get_relevant_context(query_embedding, st.session_state.doc_hash)["documents"][0]
        # Get a response from the AI-Assistant
        response = context_aware_response(question, top_chunks).text
        # Ask the AI-Assistant to rate the response
        eval: EvalResponse = generate_eval_response(question, response, ideal_answer).parsed
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
        \n Upload a Questions & Answers file in JSON format to evaluate the AI-Assistant.",
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


evaluate_button, qa_results_button = st.sidebar.columns(2)
evaluate_button.button(
    "Evaluate AI",
    help="Upload a valid QA file to enable this button",
    on_click=evaluate_ai,
    kwargs={"data": st.session_state.qa_list},
    disabled=any((st.session_state.doc_hash is None, st.session_state.qa_list is None)),
    use_container_width=True,
)

qa_results_button.button(
    "QA Results",
    key="qa_button_pressed",
    help="Click to display evaluation results after running 'Evaluate AI'.",
    on_click=display_qa_results,
    kwargs={"data": st.session_state.eval_results},
    disabled=not st.session_state.eval_results,
    use_container_width=True,
)
# Accept user input when doc_hash is defined
prompt = st.chat_input(
    disabled=st.session_state.doc_hash is None,
)
display_chat_history(st.session_state.messages, st.session_state.qa_button_pressed)
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    # Refine user question using chat history as context
    if st.session_state.messages:
        refined_prompt = refined_question_response(prompt, st.session_state.messages).text
    prompt_potentially_enhanced = locals().get("refined_prompt") or prompt
    query_embedding = create_embeddings([prompt_potentially_enhanced])[0].values
    top_chunks = get_relevant_context(query_embedding, doc_hash=st.session_state.doc_hash)["documents"][0]

    with st.chat_message("assistant"):
        response = st.write_stream(chunk.text for chunk in context_aware_response_stream(prompt_potentially_enhanced, top_chunks))

    st.session_state.messages.append({"role": "user", "content": prompt, "content_mod": prompt_potentially_enhanced})
    st.session_state.messages.append({"role": "assistant", "content": response})
