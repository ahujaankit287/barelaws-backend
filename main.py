import os
import pickle
import openai
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm

# ------------ Configuration ------------
SAVE_DIR = "bare_embeddings_output"
os.makedirs(SAVE_DIR, exist_ok=True)

EMBED_MODEL = "text-embedding-3-large"  # or text-embedding-3-small for cheaper, faster
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 128

# ------------ OpenAI API Setup ------------
openai.api_key = os.getenv("OPENAI_API_KEY")  # or set directly

# ------------ Load Dataset (1% Sample) ------------
print("📦 Loading 1% of dataset...")
dataset = load_dataset("opennyaiorg/InJudgements_dataset", split="train[:50]")

# ------------ Split and Prepare Documents ------------
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
documents = []

print("🧹 Splitting documents...")
for item in tqdm(dataset, desc="Processing"):
    text = item.get("Text") or item.get("text")
    if not text:
        continue
    chunks = splitter.split_text(text)
    for chunk in chunks:
        documents.append(Document(page_content=chunk))

# ------------ Embed Documents using OpenAI ------------
texts = [doc.page_content for doc in documents]
embeddings = []

print(f"🔢 Embedding {len(texts)} chunks...")

for i in tqdm(range(0, len(texts), 100), desc="Embedding"):
    batch = texts[i:i+100]
    try:
        response = openai.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)
    except Exception as e:
        print(f"⚠️ Error embedding batch {i}: {e}")
        embeddings.extend([[0.0] * 3072] * len(batch))  # fallback zero vector if failed

# ------------ Save Output ------------
with open(os.path.join(SAVE_DIR, "texts.pkl"), "wb") as f:
    pickle.dump(texts, f)

with open(os.path.join(SAVE_DIR, "embeddings.pkl"), "wb") as f:
    pickle.dump(embeddings, f)

print("✅ Done. Saved to", SAVE_DIR)
