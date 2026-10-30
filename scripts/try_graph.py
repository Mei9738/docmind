import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from docmind import config
from docmind.graph import build_graph

load_dotenv()
model = SentenceTransformer(config.EMBEDDING_MODEL)
chroma_client = chromadb.PersistentClient(config.DB_PATH)
collection = chroma_client.get_collection(config.COLLECTION_NAME)
client = Groq()

question = "how do I grab a value from the URL?"
starting_notebook = {"question": question, "search_queries": [question], "tries": 0}

graph = build_graph(collection, model, client)

result = graph.invoke(starting_notebook)

print("Answer:", result["answer"])
print("Sources:", result["sources"])
print("Search Queries:", result["search_queries"])
print("Tries:", result["tries"])