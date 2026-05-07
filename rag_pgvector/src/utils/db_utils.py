import psycopg2
import pandas as pd
from psycopg2.extensions import connection
from psycopg2.extras import execute_values
from pydantic import BaseModel, field_validator
from typing import Callable, Optional

from dotenv import load_dotenv

# load env var for the database connection
load_dotenv()


class EmbeddingRow(BaseModel):
    id: str
    page: Optional[str] = None
    text: Optional[str] = None
    timestamp: Optional[str] = None
    embedding: list[float]

    @field_validator("id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("id must not be empty")
        return v


def db_connection(user: str, password: str, host: str, dbname: str = "postgres", port: int = 5432) -> connection:
    """Establish and return a connection to a PostgreSQL database."""
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
    )
    return conn


def query_db(query: str) -> pd.DataFrame:
    """Execute a SQL query using psycopg2 and return the results as a DataFrame."""
    import os
    conn = db_connection(
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PWD"),
        host=os.getenv("PG_HOST"),
        dbname=os.getenv("PG_DBNAME", "postgres"),
        port=int(os.getenv("PG_PORT", 5432))
    )
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=columns)
    conn.close()
    return df


def _default_index_fn(schema: str, table: str) -> str:
    return f"CREATE INDEX IF NOT EXISTS {table}_embedding_idx ON {schema}.{table} USING hnsw (embedding vector_cosine_ops);"


def create_embedding_table(
    schema: str,
    table: str,
    embedding_dim: int,
    index_fn: Callable[[str, str], str] = _default_index_fn,
) -> None:
    """Create a pgvector table with id, page, text, and embedding columns.

    Enables the pgvector extension if not already installed, then creates the
    table under the given schema if it does not already exist. An index on the
    embedding column is created using ``index_fn``, which accepts ``schema`` and
    ``table`` and returns the SQL statement to execute.
    """
    import os
    conn = db_connection(
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PWD"),
        host=os.getenv("PG_HOST"),
        dbname=os.getenv("PG_DBNAME", "postgres"),
        port=int(os.getenv("PG_PORT", 5432))
    )
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.{table} (
                id      TEXT        NOT NULL,
                page    TEXT,
                text    TEXT,
                timestamp TIMESTAMPTZ NOT NULL,
                embedding VECTOR({embedding_dim})
            );
            """
        )
        cur.execute(index_fn(schema, table))
    conn.commit()
    conn.close()


def load_embeddings(schema: str, table: str, df: pd.DataFrame) -> int:
    """Validate a DataFrame against the embeddings table schema and bulk-insert rows.

    Each row is validated with EmbeddingRow before insertion. Raises
    ValidationError if any row does not conform to the schema.
    Returns the number of rows inserted.
    """
    import os
    
    rows = [EmbeddingRow(**row) for row in df.to_dict(orient="records")]
    records = [
        (r.id, r.page, r.text, r.timestamp, str(r.embedding))
        for r in rows
    ]
    conn =  db_connection(
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PWD"),
        host=os.getenv("PG_HOST"),
        dbname=os.getenv("PG_DBNAME", "postgres"),
        port=int(os.getenv("PG_PORT", 5432))
    )
    with conn.cursor() as cur:
        execute_values(
            cur,
            f"INSERT INTO {schema}.{table} (id, page, text, timestamp, embedding) VALUES %s",
            records,
            template="(%s, %s, %s, %s, %s::vector)",
        )
    conn.commit()
    conn.close()
    return f"Inserted {len(records)} rows into {schema}.{table}"