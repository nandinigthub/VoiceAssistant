import fitz
from backend.controller.qdrant import VectorStore
import uuid
from langchain_core.documents import Document

from langchain_text_splitters import (
    TokenTextSplitter,
)


def extract_pdf_content(file_bytes: bytes):
    """Extracts raw text"""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = ""
    for page_num, page in enumerate(doc):
        full_text += f"\nPage {page_num + 1}\n"
        full_text += page.get_text("text")

    return full_text

def parse_pdf(file):
    file_bytes = file.file.read()
    text = extract_pdf_content(file_bytes)
    return text

def GetTextSplitter(chunk_size: int = 1536, chunk_overlap: int = 100):
    return TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )



def embed_chunks(text:str):    
    text_splitter = GetTextSplitter(chunk_size=512,chunk_overlap=100)
    documents = text_splitter.split_documents([
        Document(page_content=text, metadata={"doc_id": str(uuid.uuid4())})
    ])
    return VectorStore().add_documents(documents)
