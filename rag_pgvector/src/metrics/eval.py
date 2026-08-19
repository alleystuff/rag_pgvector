
import pandas as pd
from time import perf_counter

from rag_pgvector.src.metrics.semantic import (
    measure_answer_relevance,
    measure_context_relevance,
    measure_faithfulness,
    measure_hallucination_risk,
)
from rag_pgvector.src.utils.utils import get_token_provider
from rag_pgvector.src.models.embeddings import generate_embedding
from rag_pgvector.src.data.query_pg import query_similar_chunks
from rag_pgvector.src.models.chat_completion import chat_completion

from rag_pgvector.src.data import synthetic_qs

from pathlib import Path
from dotenv import load_dotenv

dotenv_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv()

def get_response(query: str, temperature: float=1.0, top_k: int=5) -> str:
    """
    Get a response from the chat completion model based on the query.

    Parameters:
    query (str): The user query.

    Returns:
    str: The response from the chat completion model.
    """
    token_provider = get_token_provider()

    emb = generate_embedding(
        token_provider, 
        dimensions=786,
        text=query
    )

    semantic_chunks = query_similar_chunks(
        embedding=emb,
        top_k=top_k
    )

    context = "\n\n".join(semantic_chunks["text"].dropna().astype(str))

    response = chat_completion(
        token_provider=token_provider,
        user_prompt=query,
        context=context,
        temperature=temperature
    )
    return response, context

def run_metrics_evaluation(    
    model_param_grid = {
        "temperature": [0.5, 0.7, 1.0],
        "top_k": [3, 5]}
):
    """
    Run metrics evaluation for synthetic questions.
    """
    token_provider = get_token_provider()
    results = []
    for query in synthetic_qs.SYNTHETIC_QUESTIONS:
        for temperature in model_param_grid["temperature"]:
            for top_k in model_param_grid["top_k"]:
                response_start_time = perf_counter()
                response, context = get_response(
                    query=query, 
                    temperature=temperature, 
                    top_k=top_k
                )
                response_end_time = perf_counter()
                response_time_seconds = response_end_time - response_start_time
                answer_relevance = measure_answer_relevance(
                    user_query=query,
                    model_response=response,
                    token_provider=token_provider,
                )
                context_relevance = measure_context_relevance(
                    user_query=query,
                    retrieved_chunks=context,
                    token_provider=token_provider,
                )
                faithfulness = measure_faithfulness(
                    retrieved_chunks=context,
                    model_response=response,
                    token_provider=token_provider,
                )
                results.append({
                    "query": query,
                    "response": response,
                    "temperature": temperature,
                    "top_k": top_k,
                    "response_time_seconds": response_time_seconds,
                    "answer_relevance": answer_relevance,
                    "context_relevance": context_relevance,
                    "faithfulness": faithfulness,
                    "hallucination_risk": measure_hallucination_risk(faithfulness),
                })
    return pd.DataFrame(results)