from tqdm import tqdm
from qdrant_client.models import PointStruct
import uuid

def upload_documents(client, chunks, collection_name, batch_size=100):
    batch_size = int(batch_size)  # 👈 Ensure it's an integer
    for i in tqdm(range(0, len(chunks), batch_size), desc="🚀 Uploading"):
        batch = chunks[i:i + batch_size]
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk["embedding"],
                payload=chunk["metadata"] | {"text": chunk["text"]}
            )
            for chunk in batch
        ]
        client.upsert(collection_name=collection_name, points=points)