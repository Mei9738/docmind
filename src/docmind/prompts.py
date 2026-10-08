from docmind.retrieval import RetrievedChunk

TEMPLATE = """Answer using only the context below. If the context doesn't contain the answer, 
say you don't know. Cite your sources.
Context : 
{context}
Question : {question}
Answer:"""

def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    blocks : list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        block = f" [{i}] ({chunk.metadata['source']} - {chunk.metadata['heading']}) {chunk.text}"
        blocks.append(block)
    context = "\n\n".join(blocks)
    return TEMPLATE.format(context = context, question = question)
