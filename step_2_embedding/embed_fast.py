import os
import openai
from utils.file_io import read_jsonl, write_jsonl
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")

BATCH_SIZE = 50  # Tweak based on chunk size

def get_batch_embeddings(text_list):
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-large",
            input=text_list
        )
        return [r.embedding for r in response.data]
    except Exception as e:
        print("❌ Embedding error:", e)
        return [None] * len(text_list)

def main():
    chunks = read_jsonl("bls_data/chunks.jsonl")
    chunks_to_embed = [c for c in chunks if "embedding" not in c]

    print(f"📦 Embedding {len(chunks_to_embed)} chunks in batches of {BATCH_SIZE}")

    for i in tqdm(range(0, len(chunks_to_embed), BATCH_SIZE), desc="🚀 Batching"):
        batch = chunks_to_embed[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = get_batch_embeddings(texts)

        for chunk, emb in zip(batch, embeddings):
            if emb:
                chunk["embedding"] = emb

    # Replace only updated chunks
    embedded_map = {c["id"]: c for c in chunks_to_embed if "embedding" in c}
    for i, chunk in enumerate(chunks):
        if chunk["id"] in embedded_map:
            chunks[i] = embedded_map[chunk["id"]]

    write_jsonl("bls_data/chunked_documents_with_embeddings.jsonl", chunks)
    print("✅ Embeddings complete.")

if __name__ == "__main__":
    main()