import numpy as np


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
