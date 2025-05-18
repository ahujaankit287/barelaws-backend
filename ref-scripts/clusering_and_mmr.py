from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Example legal case summaries for Section 125 CrPC
case_summaries = [
    "The Supreme Court held that a Muslim husband is liable to provide maintenance to his divorced wife beyond the iddat period under Section 125 CrPC, emphasizing that the provision applies irrespective of religion.",
    "The Court interpreted the Muslim Women (Protection of Rights on Divorce) Act, 1986, stating that a reasonable and fair provision for the future of the divorced Muslim woman must be made within the iddat period, but the provision itself is not limited to that period.",
    "It was held that a Muslim woman can seek maintenance under Section 125 CrPC even after the enactment of the 1986 Act, reinforcing the secular nature of the provision.",
    "The Court emphasized that a divorced Muslim woman is entitled to claim maintenance under Section 125 CrPC if she has not remarried and is unable to maintain herself.",
    "The Supreme Court upheld the rights of divorced Muslim women to claim maintenance under Section 125 CrPC, reinforcing the provision's applicability across religions."
]

# Load the best-performing model
model = SentenceTransformer("hkunlp/instructor-large")
embeddings = model.encode(case_summaries, convert_to_tensor=True)

# Convert to numpy for clustering
embeddings_np = embeddings.cpu().numpy()

# KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(embeddings_np)

# Cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings_np)

# Show cluster groups
for i, label in enumerate(labels):
    print(f"Cluster {label}: {case_summaries[i][:100]}...")

# Show redundancy insight (mean similarity)
avg_similarity = np.mean(similarity_matrix)
print(f"\n🔁 Average Redundancy Score (Cosine Similarity): {avg_similarity:.4f}")