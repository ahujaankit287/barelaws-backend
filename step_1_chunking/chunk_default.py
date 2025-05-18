# chunk_default.py

from datasets import load_dataset
import os
import json
import uuid
from textwrap import wrap

# ---- Config ---- #
OUTPUT_FILE = "chunks.json"
CHUNK_SIZE = 500  # in words
SPLIT_BY_PARAGRAPH = True

# ---- Load Dataset ---- #
def load_judgments():
    print("📥 Loading dataset...")
    dataset = load_dataset("opennyaiorg/InJudgements_dataset", split="train[:50]")
    print(f"✅ Loaded {len(dataset)} documents.")
    return dataset

# ---- Chunk Text ---- #
def chunk_text(text, size=CHUNK_SIZE):
    words = text.split()
    chunks = wrap(' '.join(words), width=size * 5)  # approximate size in chars
    return chunks

# ---- Process and Chunk ---- #
def chunk_dataset(dataset):
    chunks = []
    print("✂️ Chunking documents...")
    
    for idx, doc in enumerate(dataset):
        judgment_text = doc.get("Text", "")
        if not judgment_text:
            continue

        doc_id = f"doc_{idx:05d}"
        base_metadata = {
            "source_id": doc_id,
            "case_name": doc.get("Titles", ""),
            "court": doc.get("Court_Name", ""),
            "cites": doc.get("Cites", ""),
            "cited_by": doc.get("Cited_by", ""),
            "url": doc.get("Doc_url", ""),
            "case_ttype": doc.get("Case_Type", ""),
            "court_name_normalized": doc.get("Court_Name_Normalized", ""),
        }

        # Split by paragraph or chunk by words
        if SPLIT_BY_PARAGRAPH:
            raw_chunks = [para.strip() for para in judgment_text.split("\n") if para.strip()]
        else:
            raw_chunks = chunk_text(judgment_text)

        for i, chunk in enumerate(raw_chunks):
            chunk_id = str(uuid.uuid4())
            chunks.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    **base_metadata,
                    "chunk_index": i
                }
            })

    print(f"✅ Chunked into {len(chunks)} chunks.")
    return chunks

# ---- Save ---- #
def save_chunks(chunks):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved {len(chunks)} chunks to {OUTPUT_FILE}")

# ---- Main ---- #
def main():
    dataset = load_judgments()
    chunks = chunk_dataset(dataset)
    save_chunks(chunks)

if __name__ == "__main__":
    main()