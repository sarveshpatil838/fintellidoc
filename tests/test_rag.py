"""
Tests for the RAG service.
"""

import pytest
import numpy as np


class TestRAGChunking:
    def test_chunking_splits_long_text(self):
        """Chunking should split text that exceeds chunk_size."""
        from app.core.config import get_settings
        settings = get_settings()

        # Create text longer than one chunk
        words = ["word"] * (settings.chunk_size * 3)
        text = " ".join(words)

        # Import and instantiate service (no API calls needed)
        # We test the chunking logic directly
        from app.services.rag import RAGService
        import unittest.mock as mock

        with mock.patch("app.services.rag.SentenceTransformer"), \
             mock.patch("app.services.rag.chromadb"):
            service = RAGService.__new__(RAGService)
            service.embedding_dim = 384
            service.index = None
            service.chunks = []

            chunks = service._chunk_text(text)
            assert len(chunks) > 1

    def test_chunking_single_chunk_for_short_text(self):
        from app.services.rag import RAGService
        import unittest.mock as mock

        with mock.patch("app.services.rag.SentenceTransformer"), \
             mock.patch("app.services.rag.chromadb"):
            service = RAGService.__new__(RAGService)
            service.embedding_dim = 384
            service.index = None
            service.chunks = []

            text = "This is a short document."
            chunks = service._chunk_text(text)
            assert len(chunks) == 1
            assert chunks[0] == text

    def test_chunks_have_overlap(self):
        """Overlapping chunks ensure context isn't lost at boundaries."""
        from app.core.config import get_settings
        from app.services.rag import RAGService
        import unittest.mock as mock

        settings = get_settings()

        with mock.patch("app.services.rag.SentenceTransformer"), \
             mock.patch("app.services.rag.chromadb"):
            service = RAGService.__new__(RAGService)
            service.embedding_dim = 384
            service.index = None
            service.chunks = []

            # Generate text where overlap matters
            words = [f"word{i}" for i in range(settings.chunk_size * 2)]
            text = " ".join(words)
            chunks = service._chunk_text(text)

            if len(chunks) >= 2:
                # Last words of chunk 0 should appear in chunk 1 (overlap)
                chunk0_words = set(chunks[0].split()[-settings.chunk_overlap:])
                chunk1_words = set(chunks[1].split()[:settings.chunk_overlap])
                overlap = chunk0_words & chunk1_words
                assert len(overlap) > 0, "Expected overlap between consecutive chunks"
