"""
RAG service using ChromaDB (replaces FAISS for easier Mac installation).

ChromaDB is a pure-Python vector database that installs cleanly on all
platforms including Apple Silicon — no swig, no C++ compilation required.
Functionally identical to the FAISS version: chunk, embed, retrieve, answer.
"""

import time
import uuid
import asyncio
from typing import Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import anthropic

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("rag")
settings = get_settings()


class DocumentChunk:
    def __init__(self, text: str, doc_id: str, chunk_index: int, metadata: dict = None):
        self.chunk_id = str(uuid.uuid4())
        self.text = text
        self.doc_id = doc_id
        self.chunk_index = chunk_index
        self.metadata = metadata or {}


class RAGService:
    def __init__(self):
        self.embedder = SentenceTransformer(settings.embedding_model)
        # Persistent ChromaDB — survives restarts
        db_path = str(Path(settings.faiss_index_path).parent / "chromadb")
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma.get_or_create_collection(
            name="fintellidoc",
            metadata={"hnsw:space": "cosine"}
        )
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        logger.info("rag_service_initialized", docs=self.collection.count())

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        step = settings.chunk_size - settings.chunk_overlap
        for i in range(0, len(words), step):
            chunk_words = words[i: i + settings.chunk_size]
            chunks.append(" ".join(chunk_words))
            if i + settings.chunk_size >= len(words):
                break
        return chunks

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    async def index_document(self, text: str, doc_id: str, metadata: dict = None) -> int:
        chunks_text = self._chunk_text(text)
        embeddings = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._embed(chunks_text)
        )
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks_text))]
        metas = [{"doc_id": doc_id, "chunk_index": i, **(metadata or {})} for i in range(len(chunks_text))]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks_text,
            metadatas=metas,
        )
        logger.info("document_indexed", doc_id=doc_id, chunks=len(chunks_text))
        return len(chunks_text)

    async def retrieve(self, question: str, doc_ids: Optional[list[str]] = None) -> list[dict]:
        if self.collection.count() == 0:
            return []

        query_embedding = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._embed([question])
        )

        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(settings.top_k_chunks, self.collection.count()),
            where=where,
        )

        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            chunks.append({"text": doc, "doc_id": meta.get("doc_id", ""), "chunk_index": meta.get("chunk_index", i)})
        return chunks

    async def query(self, question: str, doc_ids: Optional[list[str]] = None) -> dict:
        start = time.time()
        relevant_chunks = await self.retrieve(question, doc_ids)

        if not relevant_chunks:
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "confidence": 0.0,
                "processing_time_ms": int((time.time() - start) * 1000),
            }

        context = "\n\n---\n\n".join(
            f"[Source {i+1} | Doc: {c['doc_id']}]\n{c['text']}"
            for i, c in enumerate(relevant_chunks)
        )

        prompt = (
            f"Answer using ONLY the provided context. If the answer is not in the context, say so.\n\n"
            f"Question: {question}\n\nContext:\n{context}"
        )

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.messages.create(
                model=settings.claude_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
        )

        return {
            "answer": response.content[0].text,
            "sources": [{"doc_id": c["doc_id"], "chunk_index": c["chunk_index"], "excerpt": c["text"][:200]} for c in relevant_chunks],
            "confidence": 0.9,
            "processing_time_ms": int((time.time() - start) * 1000),
        }
