import pandas as pd

from rag_pgvector.src.utils.db_utils import query_db


def query_similar_chunks(embedding: list[float], top_k: int) -> pd.DataFrame:
	"""Return the top-k chunks nearest to an embedding by cosine distance."""
	if not embedding:
		raise ValueError("embedding must not be empty")
	if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
		raise ValueError("top_k must be a positive integer")

	return query_db(
		"""
		SELECT id, page, text, timestamp, embedding,
			   embedding <=> %s::vector AS cosine_distance
		FROM rag.embeddings
		ORDER BY embedding <=> %s::vector
		LIMIT %s
		""",
		(str(embedding), str(embedding), top_k),
	)
