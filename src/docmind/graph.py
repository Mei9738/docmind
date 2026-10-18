from typing import TypedDict
from docmind.retrieval import RetrievedChunk, retrieve
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from docmind.prompts import build_grade_prompt, build_rewrite_prompt
from docmind import config
from docmind.generate import generate_answer, NO_ANSWER
from langgraph.graph import StateGraph, START, END

class GraphState(TypedDict):
    question : str
    chunks : list[RetrievedChunk]
    verdict : str
    tries : int
    search_queries: list[str]
    answer : str
    sources : list[str]

def build_graph(collection:chromadb.Collection, model: SentenceTransformer, client: Groq, max_distance : float = 0.5):
        
    def retrieve_node(state: GraphState) -> dict:
        chunks = retrieve(state["search_queries"][-1], collection, model, config.K, max_distance)
        return {"chunks": chunks}

    def check_node(state: GraphState) -> dict:
        if not state["chunks"]: 
            return {"verdict": "bad"}
        
        prompt = build_grade_prompt(state["question"], state["chunks"])

        resp = client.chat.completions.create(
            model = config.LLM_MODEL,
            messages = [{"role": "user", "content": prompt}]
        )
        
        verdict = resp.choices[0].message.content.lower().strip().rstrip(".")
        
        if verdict.startswith("good"):
            return {"verdict": "good"}

        else:
            return {"verdict": "bad"}

    def rewrite_node(state:GraphState) -> dict:
        prompt = build_rewrite_prompt(state["question"], state["search_queries"])

        resp = client.chat.completions.create(
                    model = config.LLM_MODEL,
                    messages = [{"role": "user", "content": prompt}]
                )
                
        new_query = resp.choices[0].message.content.strip()
        return {"search_queries": state["search_queries"] + [new_query], "tries": state["tries"] + 1}

    def answer_node (state: GraphState) -> dict:
        answer = generate_answer(state["question"], state["chunks"], client)
        return {"answer": answer.text, "sources" : answer.sources}

    def decide(state: GraphState) -> str: 
        if state["verdict"] == "good":
            return "answer"
        elif state["tries"] < config.MAX_TRIES:
            return "rewrite"
        else:
            return "give_up"

    def give_up_node(state: GraphState) -> dict:
        return {"answer": NO_ANSWER, "sources" : []}


    # (inside build_graph, after all the boxes)
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("check", check_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("answer", answer_node)
    graph.add_node("give_up", give_up_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "check")
    graph.add_conditional_edges("check", decide)
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("give_up", END)
    graph.add_edge("answer", END)

    return graph.compile()