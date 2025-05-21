import os
import openai
from qdrant_client import QdrantClient

openai.api_key = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "barelaws_judgments"

# --- Embed Query ---
def embed_query(text):
    response = openai.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

# --- Query Qdrant ---
def query_vector_db(client, query_text, top_k=5):
    vector = embed_query(query_text)

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k,
        with_payload=True
    )

    return results


# --- Display Results ---
def show_results(results):
    print("\n🔍 Top Results:\n")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Score: {result.score:.4f}")
        print(f"Text: {result.payload.get('text')[:300]}...")
        print(f"Case Name: {result.payload.get('case_name')}")
        print(f"Court: {result.payload.get('court')}")
        print(f"URL: {result.payload.get('url')}")
        print("-" * 80)

# --- Main ---
def main():
    client = QdrantClient(host="localhost", port=6333)

    query_text = input("🧠 Enter your legal question or topic: ")
    results = query_vector_db(client, query_text)
    show_results(results)

if __name__ == "__main__":
    main()
