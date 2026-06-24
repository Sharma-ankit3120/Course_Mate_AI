from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import re
# 🔹 Clean text
def clean_text(text: str) -> str:
    text = re.sub(r"\n\s*\n", "\n\n", text)   # keep paragraph breaks
    text = re.sub(r"\n\s*", " ", text)        # remove random newlines
    text = re.sub(r"\s+", " ", text)          # normalize spaces
    return text.strip()


# 🔹 Create vector store
def create_vectorstore(pdf_path: str, persist_directory: str = "ChromaDB"):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    page_count = len(docs)

    # ✅ Clean per document (preserve metadata)
    cleaned_documents = [
        Document(
            page_content=clean_text(doc.page_content),
            metadata=doc.metadata,
        )
        for doc in docs
    ]

    # ✅ Smart chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n### ", "\n", ".", " "],
    )

    chunks = splitter.split_documents(cleaned_documents)

    # ✅ Fast embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ✅ Store in Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    return vectorstore, page_count, len(chunks)