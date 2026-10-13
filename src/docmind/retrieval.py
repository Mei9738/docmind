from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import chromadb

@dataclass
class RetrievedChunk:
    text : str
    metadata: dict[str, str | int]
    distance: float

def retrieve(question: str, collection:chromadb.Collection, model: SentenceTransformer, k:int=5, max_distance: float = 0.5) -> list[RetrievedChunk]:
    results : list[RetrievedChunk] = []
    
    query_embeddings = model.encode(question)
    res = collection.query(query_embeddings, n_results= k)
    res_doc = res["documents"][0]
    res_metadatas = res["metadatas"][0]
    res_distances = res["distances"][0]
    
    for doc, meta, dist in zip(res_doc, res_metadatas, res_distances):
        if dist <= max_distance:
            results.append(RetrievedChunk(text=doc, metadata=meta, distance=dist))

    return results