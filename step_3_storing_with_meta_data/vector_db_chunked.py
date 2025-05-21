import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from utils.file_io import read_jsonl
from utils.vector_db import upload_documents

# ---- Config ---- #
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "barelaws_judgments"
EMBEDDING_DIM = 3072  # for text-embedding-3-large

# ---- Main ---- #
def main():
    print(f"🚀 Connecting to Qdrant at http://{QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if not client.collection_exists(collection_name=COLLECTION_NAME):
        print("📦 Creating new collection...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    else:
        print("📁 Collection already exists.")

    chunks = read_jsonl("bls_data/chunked_documents_with_embeddings.jsonl")
    print(f"📤 Uploading {len(chunks)} chunks to Qdrant...")
    upload_documents(client, chunks, COLLECTION_NAME)
    print("✅ Upload complete.")

if __name__ == "__main__":
    main()