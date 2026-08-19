import numpy as np

from rag_pgvector.src.models.embeddings import generate_embedding


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors.

    Uses the same normalization approach as ragas.metrics.SemanticSimilarity._ascore:
    normalize each vector then take the dot product.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    a_norm = a / np.linalg.norm(a, keepdims=True)
    b_norm = b / np.linalg.norm(b, keepdims=True)
    return float((a_norm @ b_norm.T).item())


def measure_answer_relevance(
    user_query: str, model_response: str, token_provider
) -> float:
    """Measure semantic relevance between a user query and model response.

    A higher score indicates that the response more closely addresses the query.
    """
    query_embedding = generate_embedding(token_provider, user_query)
    response_embedding = generate_embedding(token_provider, model_response)
    return cosine_similarity(query_embedding, response_embedding)


def measure_context_relevance(
    user_query: str, retrieved_chunks: str, token_provider
) -> float:
    """Measure semantic relevance between a user query and retrieved chunks.

    A higher score indicates that retrieval returned context relevant to the query.
    """
    query_embedding = generate_embedding(token_provider, user_query)
    chunks_embedding = generate_embedding(token_provider, retrieved_chunks)
    return cosine_similarity(query_embedding, chunks_embedding)


def measure_faithfulness(
    retrieved_chunks: str, model_response: str, token_provider
) -> float:
    """Measure semantic similarity between retrieved chunks and model response.

    A higher score indicates that the response is more grounded in the context.
    """
    chunks_embedding = generate_embedding(token_provider, retrieved_chunks)
    response_embedding = generate_embedding(token_provider, model_response)
    return cosine_similarity(chunks_embedding, response_embedding)


def measure_hallucination_risk(faithfulness: float) -> float:
    """Measure hallucination risk as one minus the faithfulness score.

    A higher score indicates a response that is less grounded in retrieved context.
    """
    return 1.0 - faithfulness
