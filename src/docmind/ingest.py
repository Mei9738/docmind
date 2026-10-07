from pathlib import Path
from docmind.loader import load_documents
from docmind.chunking import chunk_document, Chunk
from sentence_transformers import SentenceTransformer
import chromadb, logging

logger = logging.getLogger(__name__)

def ingest(data_dir: Path, db_path: str) -> None:
    documents = load_documents(data_dir)
    chunks: list[Chunk] = []
    chunk_ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, str | int]] = []

    model = SentenceTransformer("all-MiniLM-L6-v2")

    for doc in documents:
        chunks.extend(chunk_document(doc))
   
    for chunk in chunks:
        chunk_ids.append(f"{chunk.metadata['source']}:{chunk.metadata['position']}")
        texts.append(chunk.text)
        metadatas.append(chunk.metadata)

    embeddings = model.encode(texts)

    client = chromadb.PersistentClient(db_path)

    existing_col = [c.name for c in client.list_collections()]
    if "docmind" in existing_col:
        client.delete_collection("docmind")

    col = client.get_or_create_collection("docmind")    
    col.add(
        ids = chunk_ids,
        documents = texts,
        metadatas = metadatas,
        embeddings = embeddings
    )

    logger.info("Indexed %d chunks from %d files", len(chunks), len(documents))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest(Path("data"), "chroma_db")