import pytest

from docmind.chunking import Chunk, _split_text, chunk_document
from docmind.loader import Document


def test_splits_on_headings():
    doc = Document(text="Intro line\n# Setup\nInstall it\nRun it\n# Usage\nCall it\n", metadata={"source": "test1.md"})
    chunks = chunk_document(doc)

    expected = [
        Chunk(text="Intro line", metadata={"source": "test1.md", "heading" : "", "position" : 1}),
        Chunk(text="Install it\nRun it", metadata={"source": "test1.md", "heading" : "Setup", "position" : 2}),
        Chunk(text="Call it", metadata={"source": "test1.md", "heading" : "Usage", "position" : 3})
        ]

    assert chunks == expected 


def test_empty_document_returns_no_chunks():
    doc = Document(text="", metadata={"source": "test_2.md"})
    chunks = chunk_document(doc)
    expected = []
    assert chunks == expected 


def test_document_starting_with_heading_has_no_empty_chunk():
    doc = Document("# Setup\nInstall it\n# Usage\nCall it", metadata={"source": "test_3.md"})
    chunks = chunk_document(doc)

    expected = [
        Chunk(text="Install it", metadata={"source": "test_3.md", "heading" : "Setup", "position" : 1}),
        Chunk(text="Call it", metadata={"source": "test_3.md", "heading" : "Usage", "position" : 2})
        ]
    assert chunks == expected 


def test_document_without_headings_is_one_chunk():
    doc = Document("just text\nmore text", metadata={"source": "test_4.md"})
    chunks = chunk_document(doc)

    expected = [
        Chunk(text="just text\nmore text", metadata={"source": "test_4.md", "heading" : "", "position" : 1})
        ]
    assert chunks == expected

def test_split_text_with_overlap():
    assert _split_text("ABCDEFGH", 4, 1) == ["ABCD", "DEFG", "GH"]
    assert _split_text("ABCDEFG", 4, 1) == ["ABCD", "DEFG"]
    assert _split_text("ABC", 4, 1) == ["ABC"]

def test_error():
    with pytest.raises(ValueError):
        _split_text("ABCDEFGH", 0, 0)

def test_chunk_document():
    doc = Document(text="# Big\n" + "x" * 25, metadata={"source": "big.md"})
    chunks = chunk_document(doc, size=10, overlap=2)

    assert len(chunks) == 3
    assert chunks[0].metadata["heading"] == "Big"
    assert chunks[0].metadata["position"] == 1
    assert chunks[1].metadata["position"] == 2
    assert chunks[2].metadata["position"] == 3

