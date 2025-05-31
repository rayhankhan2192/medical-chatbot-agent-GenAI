from typing import List, Dict
from pathlib import Path

from langchain.docstore.document import Document
from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_loader()-> DirectoryLoader:
    """Get a DirectoryLoader for loading documents. returns the loader ready to be used."""
    return DirectoryLoader(
        Path("docs"),
        glob="**/*",
        loader_cls_mapping={
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".html": UnstructuredHTMLLoader,
            ".md": TextLoader,
            ".json": TextLoader,
        },
        silent_errors=True,
    )

def load_documents(docs_dir: str) -> List[Document]:
    loader = DirectoryLoader(
        path=docs_dir,
        glob="**/*",
        loader_cls_mapping={
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".html": UnstructuredHTMLLoader,
            ".md": TextLoader,
            ".json": TextLoader,
        },
        silent_errors=True,
    )
    return loader.load()

def split_document(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """Split documents into smaller chunks. using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)


def format_chunks(documents: List[Document], min_chunks_length = 50) -> List[Dict]:
    """Format the chunks into a list of dictionaries with 'text' and 'metadata' keys."""
    formatted_chunks = []
    
    # for i, doc in enumerate(documents)
    #     if len(doc.page_content.strip()) >= min_chunks_length:
    for i, doc in enumerate(documents):
        text = doc.page_content.strip()
        if len(text) >= min_chunks_length:
            metadata = doc.metadata if doc.metadata else {}
            formatted_chunks.append({
                "text": text,
                "metadata": metadata,
                "chunks_id": f"chunk_{i}"
            })
    return formatted_chunks


def load_and_parse_doc(doc_dir: str, min_chunks_length: int = 50) -> List[Dict]:
    """Load and parse documents from a directory, returning formatted chunks."""
    documents = load_documents(doc_dir)
    if not documents:
        print(f"No documents found in {doc_dir}.")
        return []
    
    split_docs = split_document(documents)
    chunks = format_chunks(split_docs, min_chunks_length)
    
    return chunks