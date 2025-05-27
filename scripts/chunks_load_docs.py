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