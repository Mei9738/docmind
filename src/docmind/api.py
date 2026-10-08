from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
import chromadb
from docmind import config
from groq import Groq, APIError
from docmind.retrieval import retrieve
from docmind.generate import generate_answer
from chromadb.errors import NotFoundError
import logging

logger = logging.getLogger(__name__)

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = SentenceTransformer(config.EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(config.DB_PATH)
    try:
        app.state.collection = chroma_client.get_collection(config.COLLECTION_NAME)
    except NotFoundError:
        app.state.collection = None
    app.state.client = Groq()
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


