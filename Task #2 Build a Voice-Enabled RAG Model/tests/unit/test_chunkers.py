"""Unit tests for chunking strategies."""
import pytest
from ingestion.chunking import (
    SentenceChunker,
    SlidingWindowChunker,
    SemanticChunker,
    MetadataAwareChunker,
    ChunkerFactory,
)

SAMPLE_RECORD = {
    "document_id": "doc_001",
    "language": "en",
    "query": "What is photosynthesis?",
    "passage": (
        "Photosynthesis is the process by which plants convert sunlight into energy. "
        "It occurs primarily in the leaves using chlorophyll. "
        "Carbon dioxide and water are converted to glucose and oxygen. "
        "This process is fundamental to life on Earth. "
        "It forms the base of most food chains and provides the oxygen we breathe."
    ),
    "answer": "Plants use sunlight, CO2, and water to produce glucose.",
    "source": "test",
}

HINDI_RECORD = {
    "document_id": "doc_002",
    "language": "hi",
    "query": "प्रकाश संश्लेषण क्या है?",
    "passage": (
        "प्रकाश संश्लेषण वह प्रक्रिया है जिसके द्वारा पौधे सूर्य प्रकाश को ऊर्जा में परिवर्तित करते हैं। "
        "यह मुख्यतः पत्तियों में क्लोरोफिल की सहायता से होता है।"
    ),
    "answer": "पौधे सूर्य प्रकाश, CO2 और पानी से ग्लूकोज बनाते हैं।",
    "source": "test",
}


class TestSentenceChunker:
    def test_basic_chunking(self):
        chunker = SentenceChunker(sentences_per_chunk=2, overlap=0)
        chunks = chunker.chunk(SAMPLE_RECORD)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.strategy == "sentence"
            assert chunk.document_id == "doc_001"
            assert chunk.language == "en"
            assert len(chunk.text) > 0
            assert chunk.token_count > 0

    def test_overlap(self):
        no_overlap = SentenceChunker(sentences_per_chunk=2, overlap=0).chunk(SAMPLE_RECORD)
        with_overlap = SentenceChunker(sentences_per_chunk=2, overlap=1).chunk(SAMPLE_RECORD)
        # With overlap, we get more chunks
        assert len(with_overlap) >= len(no_overlap)

    def test_unique_chunk_ids(self):
        chunker = SentenceChunker()
        chunks = chunker.chunk(SAMPLE_RECORD)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_empty_passage(self):
        record = {**SAMPLE_RECORD, "passage": ""}
        chunks = SentenceChunker().chunk(record)
        assert chunks == []

    def test_hindi_passage(self):
        chunker = SentenceChunker(sentences_per_chunk=1, overlap=0)
        chunks = chunker.chunk(HINDI_RECORD)
        assert len(chunks) > 0
        assert chunks[0].language == "hi"


class TestSlidingWindowChunker:
    def test_basic_chunking(self):
        chunker = SlidingWindowChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk(SAMPLE_RECORD)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.strategy == "sliding_window"
            assert chunk.token_count <= 20

    def test_no_overlap(self):
        chunker = SlidingWindowChunker(chunk_size=15, overlap=0)
        chunks = chunker.chunk(SAMPLE_RECORD)
        assert len(chunks) > 0

    def test_chunk_covers_full_text(self):
        passage = SAMPLE_RECORD["passage"]
        words = passage.split()
        chunker = SlidingWindowChunker(chunk_size=len(words) + 10, overlap=0)
        chunks = chunker.chunk(SAMPLE_RECORD)
        assert len(chunks) == 1
        assert len(chunks[0].text.split()) == len(words)


class TestMetadataAwareChunker:
    def test_metadata_propagated(self):
        inner = SlidingWindowChunker(chunk_size=20, overlap=5)
        chunker = MetadataAwareChunker(inner)
        chunks = chunker.chunk(SAMPLE_RECORD)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "original_query" in chunk.parent_metadata
            assert chunk.parent_metadata["original_query"] == SAMPLE_RECORD["query"]


class TestChunkerFactory:
    def test_all_strategies(self):
        for strategy in ["sentence", "sliding_window"]:
            chunker = ChunkerFactory.create(strategy)
            chunks = chunker.chunk(SAMPLE_RECORD)
            assert len(chunks) > 0

    def test_unknown_strategy(self):
        with pytest.raises(ValueError):
            ChunkerFactory.create("nonexistent_strategy")
