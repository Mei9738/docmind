from docmind.retrieval import RetrievedChunk

TEMPLATE = """Answer using only the context below. If the context doesn't contain the answer, 
say you don't know. Cite your sources.
Context : 
{context}
Question : {question}
Answer:"""

GRADE_TEMPLATE = """Your role is to act as an assistant that will be grading chunks below, 
based on the question we have.

good = the chunks contain the information needed to answer the question
bad = they don’t, even if they’re about a related topic

Question: {question}

Chunks:
{context}

Answer with exactly one word: good or bad. No explanation."""

REWRITE_TEMPLATE = """Rewrite the question using the question below. Use technical FastAPI
terms and keep the same meaning.
These versions were already tried and did not find good results. Write a different one.

Here is the original question: {question}

Here is the previous versions we searched with: {previous_questions}

Answer with only the new question. No explanation."""

def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    blocks : list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        block = f" [{i}] ({chunk.metadata['source']} - {chunk.metadata['heading']}) {chunk.text}"
        blocks.append(block)
    context = "\n\n".join(blocks)
    return context

def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = _format_chunks(chunks)
    return TEMPLATE.format(context = context, question = question)

def build_grade_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = _format_chunks(chunks)
    return GRADE_TEMPLATE.format(context = context, question = question)

def build_rewrite_prompt(question: str, previous_questions: list[str]) -> str:
    blocks : list[str] = []
    for i, q in enumerate(previous_questions, start=1):
        block = f" [question {i}] - {q}"
        blocks.append(block)
    q_list = "\n\n".join(blocks)
    return REWRITE_TEMPLATE.format(question = question, previous_questions = q_list)
