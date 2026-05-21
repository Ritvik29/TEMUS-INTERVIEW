"""
Ingestion pipeline: chunk documents, embed with multilingual model, store in ChromaDB + BM25.
"""
from __future__ import annotations

import os
from typing import Any
from pathlib import Path

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
import pickle

from src.rag.documents import DOCUMENTS

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 50
COLLECTION_NAME = "aurawealth_docs"
PERSIST_DIR = str(Path(__file__).parent.parent.parent / "data" / "chromadb")
BM25_PATH = str(Path(__file__).parent.parent.parent / "data" / "bm25.pkl")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) >= 20]


def build_index(persist_dir: str = PERSIST_DIR, bm25_path: str = BM25_PATH) -> tuple[Any, BM25Okapi, list[dict]]:
    os.makedirs(persist_dir, exist_ok=True)
    os.makedirs(os.path.dirname(bm25_path), exist_ok=True)

    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=persist_dir)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    all_chunks: list[dict] = []
    for doc in DOCUMENTS:
        for i, chunk in enumerate(chunk_text(doc["content"])):
            all_chunks.append({
                "id": f"{doc['id']}_chunk_{i:04d}",
                "text": chunk,
                "metadata": {
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "category": doc["category"],
                    "language": doc["language"],
                    "date": doc["date"],
                    "chunk_index": i,
                },
            })

    print(f"Total chunks: {len(all_chunks)}")

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True).tolist()

    collection.add(
        ids=[c["id"] for c in all_chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[c["metadata"] for c in all_chunks],
    )

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": all_chunks}, f)

    print(f"Index built: {len(all_chunks)} chunks in ChromaDB + BM25.")
    return collection, bm25, all_chunks


def load_index(persist_dir: str = PERSIST_DIR, bm25_path: str = BM25_PATH):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(COLLECTION_NAME)
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    return collection, data["bm25"], data["chunks"]


if __name__ == "__main__":
    build_index()
