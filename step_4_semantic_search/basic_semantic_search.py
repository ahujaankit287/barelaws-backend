import os
import pickle
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ------------ Config ------------
SAVE_DIR = "bare_embeddings_output"
EMBED_MODEL = "text-embedding-3-large"
openai.api_key = os.getenv("OPENAI_API_KEY")  # or set directly

# ------------ Load Saved Embeddings and Texts ------------
with open(os.path.join(SAVE_DIR, "texts.pkl"), "rb") as f:
    texts = pickle.load(f)

with open(os.path.join(SAVE_DIR, "embeddings.pkl"), "rb") as f:
    embeddings = pickle.load(f)

embeddings = np.array(embeddings)  # shape: (N, 3072)

# ------------ Query Input and Embedding ------------
query = input("🔍 Enter your search query: ")

response = openai.embeddings.create(
    model=EMBED_MODEL,
    input=[query]
)
query_embedding = np.array(response.data[0].embedding).reshape(1, -1)

# ------------ Compute Similarities ------------
similarities = cosine_similarity(query_embedding, embeddings)[0]
top_k = 5
top_indices = similarities.argsort()[-top_k:][::-1]

# ------------ Display Results ------------
print("\n📄 Top Matches:")
for i, idx in enumerate(top_indices):
    print(f"\n#{i+1} (Score: {similarities[idx]:.4f}):\n{texts[idx]}")