import streamlit as st
import os
import tempfile
from database import create_vectorstore

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI

from dotenv import load_dotenv

load_dotenv()

st.write("✅ App Loaded")


# ─── Page Config ─────────────────────────────────────────
st.set_page_config(page_title="CourseMate", layout="wide")

# ─── Session State ───────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = None


# ✅ Load components (cached → FAST)
@st.cache_resource
def load_components(persist_dir):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    model = ChatMistralAI(
        model="mistral-small",
        temperature=0.3
    )

    return retriever, model


# ─── Sidebar (Upload) ────────────────────────────────────
st.sidebar.title("📄 Upload PDF")

uploaded = st.sidebar.file_uploader("Upload your file", type=["pdf"])

if uploaded:
    if st.session_state.persist_dir is None:

        with st.spinner("Processing PDF..."):

            # temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            persist_dir = tempfile.mkdtemp()

            vs, pages, chunks = create_vectorstore(tmp_path, persist_dir)

            st.session_state.persist_dir = persist_dir
            st.session_state.messages = []

            os.unlink(tmp_path)

        st.success("✅ Document Ready!")


# ─── Main Chat UI ────────────────────────────────────────
st.title("🎓 CourseMate – AI Study Partner")

if st.session_state.persist_dir is None:
    st.info("Upload a PDF to start chatting.")
else:

    # ✅ Load retriever + model
    retriever, model = load_components(st.session_state.persist_dir)

    # chat history display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # input
    query = st.chat_input("Ask your question...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Thinking..."):

            # ✅ Retrieve docs
            docs = retriever.invoke(query)

            context = "\n\n".join([doc.page_content for doc in docs])

            # ✅ Smart prompt
            prompt = f"""
You are an intelligent AI assistant.

Rules:
1. If answer is in context → answer from context
2. If related but missing → say politely not found
3. If general → answer normally

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

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })