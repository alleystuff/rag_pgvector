import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

ENDPOINT = os.getenv("ENDPOINT")
DEPLOYMENT = os.getenv("DEPLOYMENT")
API_VERSION = os.getenv("API_VERSION")


def chat_completion(token_provider, user_prompt: str, context: str=None, temperature: float=1.0) -> str:
    client = AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        azure_ad_token_provider=token_provider,
    )

    system_instructions = """
    You are a financial analyst expert in analyzing financial documents, including SEC filings, annual reports, and other financial statements.
    Your task is to provide accurate and concise answers to user queries based on the provided context.
    If the answer is not present in the context, respond with "I don't know."
    """
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt + ("\n\nContext:\n" + context if context else "")},
        ],
        max_completion_tokens=500,
        temperature=temperature,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=DEPLOYMENT,
    )
    return response.choices[0].message.content
