import json
import pickle
import math

# Load data dan index dari file
with open('data/tfidf_vectors.pkl', 'rb') as f:
    tfidf_vectors = pickle.load(f)

with open('data/idf.pkl', 'rb') as f:
    idf = pickle.load(f)

with open('data/meta.pkl', 'rb') as f:
    metadata = pickle.load(f)

with open('./data_berita_bersih.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Fungsi cosine similarity
def cosine_similarity(vec1, vec2):
    dot = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in vec1)
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0

# Fungsi cek relevansi otomatis (heuristik)
def is_relevant(query, doc_text, threshold=0.7):
    query_tokens = set(query.lower().split())
    doc_tokens = set(doc_text.lower().split())
    intersection = query_tokens.intersection(doc_tokens)
    similarity = len(intersection) / len(query_tokens) if query_tokens else 0
    return similarity >= threshold

# Fungsi buat query tf-idf vector
def query_to_tfidf(query, idf):
    tokens = query.lower().split()
    q_freq = {}
    for token in tokens:
        if token in idf:
            q_freq[token] = q_freq.get(token, 0) + 1
    return {token: count * idf[token] for token, count in q_freq.items()}

# List query contoh (bisa diganti/diperbanyak)
queries = [
    "kylian mbappe sepatu emas eropa",
    "marc Marquez crash",
    "real madrid liga champions",
    "megawati korea"
]

k = 10  # Precision @ k

print("Evaluasi Precision@{} untuk beberapa query...\n".format(k))

for query in queries:
    # Buat relevance set pakai heuristik
    relevant_docs = set()
    for i, doc in enumerate(data):
        text = doc['judul_bersih'] + " " + doc['konten_bersih']
        if is_relevant(query, text):
            relevant_docs.add(i)

    # Buat query tf-idf vector
    q_tfidf = query_to_tfidf(query, idf)

    # Hitung cosine similarity tiap dokumen
    scores = []
    for i, vec in enumerate(tfidf_vectors):
        score = cosine_similarity(q_tfidf, vec)
        if score > 0:
            scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)

    # Ambil top-k hasil pencarian
    top_k = scores[:k]

    # Hitung precision@k
    if len(top_k) == 0:
        precision = 0
    else:
        relevant_in_top_k = sum(1 for doc_id, _ in top_k if doc_id in relevant_docs)
        precision = relevant_in_top_k / len(top_k)

    print(f"Query: {query}")
    print(f"Relevant docs (heuristik): {len(relevant_docs)}")
    print(f"Retrieved docs (top {k}): {len(top_k)}")
    print(f"Precision@{k}: {precision:.4f}\n")
