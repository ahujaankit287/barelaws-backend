import os
import time
import openai
from utils.file_io import read_jsonl, write_jsonl

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text):
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print("❌ Embedding error:", e)
        return None

def main():
    chunks = read_jsonl("bls_data/chunks.jsonl")

    for chunk in chunks:
        if "embedding" not in chunk:
            chunk["embedding"] = get_embedding(chunk["text"])

    write_jsonl("bls_data/chunked_documents_with_embeddings.jsonl", chunks)

if __name__ == "__main__":
    main()