from dataclasses import dataclass

from groq import Groq

from docmind import config
from docmind.prompts import build_prompt
from docmind.retrieval import RetrievedChunk

NO_ANSWER = "I don't have enough information to answer that."

@dataclass
class Answer:
    text: str
    sources: list[str]

def generate_answer(question: str, chunks: list[RetrievedChunk], client: Groq) -> Answer:
    if not chunks: 
        return Answer(NO_ANSWER, [])
    
    prompt = build_prompt(question=question, chunks=chunks)

    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role":"user", "content" : prompt}]
    )
    sources : list[str] = []
    for chunk in chunks:
        sources.append(f"{chunk.metadata['source']} - {chunk.metadata['heading']}")

    return Answer(text=resp.choices[0].message.content, sources=sources)