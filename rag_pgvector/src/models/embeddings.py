import os
from typing import Optional
import uuid
from dotenv import load_dotenv
from openai import AzureOpenAI

import pytz
from datetime import datetime

load_dotenv()

ENDPOINT = os.getenv("ENDPOINT")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
API_VERSION = os.getenv("EMBEDDING_API_VERSION")


def generate_embeddings(token_provider, texts: list[str], page: Optional[int], dimensions: int = 786) -> list[dict]:
    client = AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        azure_ad_token_provider=token_provider,
    )
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
        dimensions=dimensions
    )
    timestamp = datetime.now(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # print(f"Embeddings generated at timestamp: {timestamp}")
    return [
        {
            "id": str(uuid.uuid4()),
            "page": str(page) if page is not None else None,
            "text": texts[item.index],
            "timestamp": timestamp,
            "embedding": item.embedding,
        }
        for item in response.data
    ]
