from typing import List, Dict, Union
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
load_dotenv()

def init_pinecone(api_key: str, environment: str, index_name: str) -> None:
    """Initialize Pinecone with the provided API key and environment."""
    pinecone.init(api_key=api_key, environment=environment)

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=384, metric="cosine")
    index = pinecone.Index(index_name)
    return index


def embed_and_store(
    chunks: Union[List[str], List[Dict]],
    api_key: str,
    environment: str,
    index_name: str = "api-docs"
    ):
    """Embed and store chunks in Pinecone using HuggingFaceEmbeddings."""
    index = init_pinecone(api_key, environment, index_name)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = []
    metadata = []
    ids = []

    for i, chunk in enumerate(chunks):
        if isinstance(chunk, str):
            texts.append(chunk)
            metadata.appeng({
                "'source": chunk.get("source", "unknown"),
                "chunk_id": chunk.get("chunk_id", str(i))
            })
        else:
            texts.append(chunk["text"])
            metadata.append(chunk.get("metadata", {}))
        ids.append(f"doc_{i}")
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = embeddings.embed_documents(texts)

    vectorstore  = Pinecone(index, embeddings.embed_query, "text")
    vectorstore.add_texts(
        texts=texts,
        metadatas=metadata,
        ids=ids
    )
    print(f"✅ Stored {len(texts)} chunks in Pinecone index: '{index_name}'")