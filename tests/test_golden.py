from pathlib import Path
from docmind.loader import load_documents
from docmind.chunking import chunk_document, Chunk
import json


def test_golden_headings_exist() -> None:
    # 1. load docs from data/ and chunk every one of them
    docs = load_documents(Path("data"))
    chunks: list[Chunk] = []
    for doc in docs: 
        chunks.extend(chunk_document(doc))
    
    # 2. build a set of (source, heading) pairs from the chunks
    pairs = set()
    for chunk in chunks:
        pairs.add((chunk.metadata['source'], chunk.metadata['heading']))

    # 3. load golden.json with json (like in the REPL above)
    golden = json.loads(Path("eval/golden.json").read_text(encoding="utf-8"))

    # 4. loop over each question, then over its "expected" list,
    #    and assert the (source, heading) pair is in your set

    for question in golden:
        for expected in question["expected"]:
            expected_pair = (expected["source"], expected["heading"])
            assert expected_pair in pairs, f"{question['id']}: {expected_pair} not found"

