import re

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving paragraphs.
    """

    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def create_vectorstore(
    pdf_path: str,
    persist_directory: str = "ChromaDB"
):
    """
    Load PDF -> Clean -> Chunk -> Embed -> Store in Chroma
    """

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    page_count = len(docs)

    cleaned_documents = [
        Document(
            page_content=clean_text(doc.page_content),
            metadata=doc.metadata,
        )
        for doc in docs
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n### ",
            "\n",
            ".",
            " ",
        ],
    )

    chunks = splitter.split_documents(
        cleaned_documents
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    return (
        vectorstore,
        page_count,
        len(chunks),
    )