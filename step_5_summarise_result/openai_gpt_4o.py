import os
import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchText, SearchRequest

openai.api_key = os.getenv("OPENAI_API_KEY")

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "barelaws_judgments"

# ---- Embed query using OpenAI ---- #
def get_query_embedding(query):
    response = openai.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    return response.data[0].embedding

# ---- Retrieve similar chunks from Qdrant ---- #
def retrieve_top_chunks(client, query, top_k=5):
    query_vector = get_query_embedding(query)

    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )

    return hits

# ---- Build context from retrieved chunks ---- #
def build_context_from_hits(hits):
    context_blocks = []
    for i, hit in enumerate(hits):
        text = hit.payload.get("text", "")
        case_name = hit.payload.get("case_name", "Unknown Case")
        court = hit.payload.get("court", "")
        citation = f"[{case_name} - {court}]"
        url = f"{hit.payload.get("url", "no URL")}"
        context_blocks.append(f"{citation}\n{text}\n\nURL: {url}\n")
    return "\n\n".join(context_blocks)

# ---- Run GPT-4o with RAG ---- #
def ask_llm_with_context(prompt, context):
    system_message = (
        "You are a legal assistant helping summarize and interpret Indian case law. "
        "Use only the context provided. Always cite the case name, court and URL. Never use a case that is not in the context. "
        "If the context is not sufficient to answer the question, say 'I don't know'. "
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Context:\n{context}\n\nQuery:\n{prompt}"}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3
    )

    return response.choices[0].message.content.strip()

# ---- Main Flow ---- #
def main():
    prompt = input("🧠 Enter your legal query: ")

    print("🔍 Searching Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    hits = retrieve_top_chunks(client, prompt)

    print(f"📄 Retrieved {len(hits)} chunks. Preparing context...")
    context = build_context_from_hits(hits)

    print("🤖 Asking GPT-4o...")
    answer = ask_llm_with_context(prompt, context)

    print("\n💡 GPT-4o Response:\n")
    print(answer)

if __name__ == "__main__":
    main()