from docmind import config
from docmind.retrieval import retrieve
from docmind.metrics import recall_at_k, reciprocal_rank
import json 
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
from docmind.graph  import build_graph
import time
import mlflow


def save_results(results: dict) -> Path:
    Path("eval/results").mkdir(parents=True, exist_ok=True)

    filename = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = Path(f"eval/results/{filename}.json")

    file_path.write_text(json.dumps(results, indent=2))
    return file_path

note = "Graph, full run"
pipeline = "graph"

def run_eval() -> Path:
    load_dotenv()
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(config.DB_PATH)
    collection = chroma_client.get_collection(config.COLLECTION_NAME)
    client = Groq(max_retries=5)
    graph = build_graph(collection, model, client, config.EVAL_MAX_DISTANCE)

    golden = json.loads(Path("eval/golden.json").read_text(encoding="utf-8"))
    
    recall_1 = []
    recall_3 = []
    recall_5 = []
    mrr = []
    per_question: dict[str, list[bool]] = {}

    for item in golden:
    # 1. expected_pairs = set of (source, heading) from item["expected"]
        expected_pair = set()
        for expected in item["expected"]:
            expected_pair.add((expected["source"], expected["heading"]))
    # 2. chunks = retrieve(item["question"], collection, model, k=5, max_distance=2.0)

        if pipeline == "raw": 
            chunks = retrieve(item["question"], collection, model, config.K, config.EVAL_MAX_DISTANCE )
        else : 
            starting_notebook = {"question": item["question"], "search_queries": [item["question"]], "tries": 0}
            result = graph.invoke(starting_notebook)
            chunks = result["chunks"]
            time.sleep(3)


    # 3. hits = []  then loop over chunks, appending True/False
        hits = []
        for chunk in chunks:
            pair = (chunk.metadata["source"], chunk.metadata["heading"])
            if pair in expected_pair:
                hits.append(True)
            else:
                hits.append(False)

        per_question[item["id"]] = hits

        recall_1.append(recall_at_k(hits, 1))
        recall_3.append(recall_at_k(hits, 3))
        recall_5.append(recall_at_k(hits, 5))
        mrr.append(reciprocal_rank(hits))


    avg_r1 = sum(recall_1)/len(recall_1)
    avg_r3 = sum(recall_3)/len(recall_3)
    avg_r5 = sum(recall_5)/len(recall_5)
    avg_mrr = sum(mrr)/len(mrr)
    print(f"Average recall@1 {avg_r1:.3f}")
    print(f"Average recall@3 {avg_r3:.3f}")
    print(f"Average recall@5 {avg_r5:.3f}")
    print(f"Average MRR {avg_mrr:.3f}")

    results = {
                "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
                "note": note,
                "params": {
                    "embedding_model": config.EMBEDDING_MODEL, 
                    "chunk_size": config.CHUNK_SIZE, 
                    "overlap": config.CHUNK_OVERLAP, 
                    "k": config.K, 
                    "max_distance": config.EVAL_MAX_DISTANCE,
                    "pipeline" : pipeline },
                "metrics": { 
                    "recall_at_1": round(avg_r1, 3), 
                    "recall_at_3": round(avg_r3, 3), 
                    "recall_at_5": round(avg_r5, 3), 
                    "mrr": round(avg_mrr, 3) },
                "per_question": per_question
                    }

    path = save_results(results)


    mlflow.set_experiment("docmind-retrieval")      # a folder that groups your runs

    with mlflow.start_run(run_name=note):            # one run = one eval
        mlflow.log_params(results["params"])                    # a dict of settings
        mlflow.log_metrics(results["metrics"])                   # a dict of scores
        mlflow.log_artifact(str(path))                    # the path of a file to attach

    return path
    
if __name__ == "__main__":
    run_eval()