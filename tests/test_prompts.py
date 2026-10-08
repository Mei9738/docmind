from docmind.prompts import build_prompt
from docmind.retrieval import RetrievedChunk

def test_build_prompt_numbers_chunks_and_includes_question() :
    chunks = [
        RetrievedChunk(text= "Use None as default.", metadata={"source": "a.md", "heading":"Optional"}, 
                       distance=0.3),
    ]

    prompt = build_prompt("How?", chunks)

    assert "[1] (a.md - Optional)" in prompt
    assert "Use None as default." in prompt
    assert "How?" in prompt
    