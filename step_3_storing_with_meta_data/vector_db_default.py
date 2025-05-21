# step_3_vectorstore/store_qdrant.py

import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from utils.file_io import read_jsonl

COLLECTION_NAME = "barelaws_embeddings"
EMBEDDING_DIM = 3072  # for `text-embedding-3-large`
VECTOR_STORE_PATH = "bls_data/chunked_documents_with_embeddings.jsonl"

def init_qdrant():
    print("🚀 Connecting to Qdrant at http://localhost:6333")
    client = QdrantClient(url="http://localhost:6333")
    
    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        print("📦 Creating new collection...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    else:
        print("📦 Collection already exists.")

    return client

def upload_documents(client, chunks):
    print(f"📤 Uploading {len(chunks)} chunks to Qdrant...")

    points = []
    for chunk in chunks:
        points.append(PointStruct(
            id=chunk.get("id", str(uuid.uuid4())),
            vector=chunk["embedding"],
            payload=chunk["metadata"]
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("✅ Upload complete.")

def main():
    chunks = read_jsonl(VECTOR_STORE_PATH)
    chunks = [chunk for chunk in chunks if "embedding" in chunk]

    client = init_qdrant()
    upload_documents(client, chunks)

if __name__ == "__main__":
    main()