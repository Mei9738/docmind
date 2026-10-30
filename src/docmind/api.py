import logging
from contextlib import asynccontextmanager

import chromadb
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import APIError, Groq
from pydantic import BaseModel

from docmind import config
from docmind.generate import generate_answer
from docmind.graph import build_graph
from docmind.retrieval import retrieve

logger = logging.getLogger(__name__)

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ONNXMiniLM_L6_V2()
    chroma_client = chromadb.PersistentClient(config.DB_PATH)
    try:
        app.state.collection = chroma_client.get_collection(config.COLLECTION_NAME)
    except NotFoundError:
        app.state.collection = None
    app.state.client = Groq()
    app.state.graph = build_graph(app.state.collection, app.state.model, app.state.client)
    yield


app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"status" : "OK"}

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieval_scores: list[float]


@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    collection = app.state.collection
    if collection is None or collection.count() == 0 :
        raise HTTPException(status_code=503, detail="Index is empty. Run python -m docmind.ingest first.")
    
    chunks = retrieve(request.question, collection, app.state.model)
    
    try: 
        answer = generate_answer(request.question, chunks, app.state.client)
    except APIError as e:
        logger.error("LLM call failed: %s", e)
        raise HTTPException(
                    status_code=503,
                    detail="The answer service is temporarily unavailable. Please try again later."
                )

    scores: list[float] = []
    for chunk in chunks:
        scores.append(chunk.distance)

    return AskResponse(answer=answer.text, sources=answer.sources, retrieval_scores=scores)

@app.post("/ask-graph")
def ask_graph(request: AskRequest) -> AskResponse:
    collection = app.state.collection
    if collection is None or collection.count() == 0 :
        raise HTTPException(status_code=503, detail="Index is empty. Run python -m docmind.ingest first.")
    
    starting_notebook = {"question": request.question, "search_queries": [request.question], "tries": 0}
    graph = app.state.graph
    try: 
        result = graph.invoke(starting_notebook)
    except APIError as e:
            logger.error("LLM call failed: %s", e)
            raise HTTPException(
                        status_code=503,
                        detail="The answer service is temporarily unavailable. Please try again later."
                    )

    scores: list[float] = []
    for chunk in result["chunks"]:
        scores.append(chunk.distance)


    return AskResponse(answer=result["answer"], sources=result["sources"], retrieval_scores=scores)

