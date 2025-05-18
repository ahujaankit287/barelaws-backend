import pickle
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load your data
with open("bare_embeddings_output/texts.pkl", "rb") as f:
    texts = pickle.load(f)

with open("bare_embeddings_output/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

embeddings = np.array(embeddings)

# 🔁 Normalize if not already
from sklearn.preprocessing import normalize
embeddings = normalize(embeddings)

# 🔍 Semantic Search + MMR
def mmr_search(query, embeddings, texts, top_k=10, lambda_param=0.5):
    # Get query embedding using OpenAI
    import openai
    from openai import OpenAI
    openai.api_key = "your-openai-api-key"  # Replace with secure method
    client = OpenAI()

    print("📡 Getting query embedding from OpenAI...")
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=[query]
    )
    query_embedding = np.array(response.data[0].embedding).reshape(1, -1)

    # Cosine similarities
    similarities = cosine_similarity(embeddings, query_embedding).flatten()

    selected = []
    candidates = list(range(len(embeddings)))

    while len(selected) < top_k and candidates:
        if not selected:
            idx = np.argmax(similarities)
            selected.append(idx)
            candidates.remove(idx)
            continue

        candidate_sims = similarities[candidates]
        diversity = np.max(cosine_similarity(embeddings[candidates], embeddings[selected]), axis=1)

        mmr_scores = lambda_param * candidate_sims - (1 - lambda_param) * diversity
        selected_idx = candidates[np.argmax(mmr_scores)]
        selected.append(selected_idx)
        candidates.remove(selected_idx)

    results = [(texts[i], similarities[i]) for i in selected]
    return results

# 🎯 Run
query = "Power of High Court to quash FIR under Article 226"
results = mmr_search(query, embeddings, texts, top_k=10, lambda_param=0.6)

print("\n📚 Top MMR Results:")
for i, (text, score) in enumerate(results):
    print(f"\n--- Result #{i+1} (score: {score:.4f}) ---")
    print(text[:500])
    print("...")