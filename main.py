import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from database import create_vectorstore

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI


# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="CourseMate",
    layout="wide"
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = None


@st.cache_resource
def load_components(persist_dir):
    """
    Load embeddings, vectorstore, retriever and LLM once.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    model = ChatMistralAI(
        model="mistral-small",
        temperature=0.3,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )

    return retriever, model


# Sidebar
st.sidebar.title("📄 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file:

    if st.session_state.persist_dir is None:

        with st.spinner("Processing PDF..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:
                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name

            persist_dir = tempfile.mkdtemp()

            vectorstore, page_count, chunk_count = create_vectorstore(
                pdf_path,
                persist_dir
            )

            st.session_state.persist_dir = persist_dir
            st.session_state.messages = []

            os.unlink(pdf_path)

        st.success(
            f"✅ Document Ready! Pages: {page_count} | Chunks: {chunk_count}"
        )


# Main UI
st.title("🎓 CourseMate – AI Study Partner")

if st.session_state.persist_dir is None:

    st.info("Upload a PDF to start chatting.")

else:

    retriever, model = load_components(
        st.session_state.persist_dir
    )

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask your question...")

    if query:

        st.session_state.messages.append(
            {"role": "user", "content": query}
        )

        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Thinking..."):

            docs = retriever.invoke(query)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            prompt = f"""
You are an intelligent AI assistant.

Rules:
1. If the answer exists in the provided context, answer using the context.
2. If the question is related to the document but information is missing, politely say that the information is not available in the document.
3. If the question is general knowledge and unrelated to the document, answer normally.

Context:
{context}

Question:
{query}

Answer:
"""

            response = model.invoke(prompt)

            answer = response.content

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )