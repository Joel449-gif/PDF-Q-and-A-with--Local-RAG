import os
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import fitz

import config
from vector_store import add_documents


def extract_text_from_pdf(file_path: str) -> str:
    text_pages = []
    doc = fitz.open(file_path)
    for page in doc:
        text_pages.append(page.get_text())
    doc.close()
    return "\n\n".join(text_pages)


def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def process_uploaded_file(uploaded_file) -> list[Document]:
    name = uploaded_file.name
    ext = os.path.splitext(name)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    suffix_map = {".txt": ".txt", ".md": ".md"}
    suffix = suffix_map.get(ext, ".pdf")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            raw_text = extract_text_from_pdf(tmp_path)
        else:
            raw_text = extract_text_from_txt(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not raw_text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(raw_text)

    documents = []
    for i, chunk in enumerate(chunks):
        documents.append(Document(
            page_content=chunk,
            metadata={"source": name, "chunk": i},
        ))

    return documents


def index_file(uploaded_file) -> int:
    docs = process_uploaded_file(uploaded_file)
    if not docs:
        return 0
    add_documents(docs)
    return len(docs)
