from docmind.loader import Document, load_documents


def test_loader(tmp_path):
    file1 = tmp_path / "test_1.md"
    file2 = tmp_path / "test_2.md"

    file1.write_text("test1", encoding="utf-8")    
    file2.write_text("test2", encoding="utf-8")
    
    docs = load_documents(tmp_path)

    expected = [
        Document(text="test1", metadata={"source": "test_1.md"}),
        Document(text="test2", metadata={"source": "test_2.md"})
    ]

    assert docs == expected 

