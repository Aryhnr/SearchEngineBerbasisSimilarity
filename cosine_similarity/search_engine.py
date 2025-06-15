import pickle
import math

# Fungsi Cosine Similarity
def cosine_similarity(vec1, vec2):
    dot = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in vec1)
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0

# Load data
with open('data/tfidf_vectors.pkl', 'rb') as f:
    tfidf_vectors = pickle.load(f)

with open('data/idf.pkl', 'rb') as f:
    idf = pickle.load(f)

with open('data/meta.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Input query
query = input("Masukkan kata kunci pencarian: ").lower()
query_tokens = query.split()

# TF query
q_freq = {}
for token in query_tokens:
    if token in idf:
        q_freq[token] = q_freq.get(token, 0) + 1

# TF-IDF query
q_tfidf = {token: count * idf[token] for token, count in q_freq.items()}

# Hitung skor cosine similarity
results = []
for i, vec in enumerate(tfidf_vectors):
    score = cosine_similarity(q_tfidf, vec)
    if score > 0:
        results.append({
            'judul': metadata[i]['judul'],
            'kategori': metadata[i]['kategori'],
            'tanggal': metadata[i]['tanggal'],
            'link': metadata[i]['link'],
            'penulis': metadata[i]['penulis'],
            'gambar': metadata[i]['gambar'],
            'score': round(score, 4)
        })

# Urutkan berdasarkan skor tertinggi
results.sort(key=lambda x: x['score'], reverse=True)

# Tampilkan hasil
if results:
    print(f"\nDitemukan {len(results)} hasil relevan:\n")
    for i, res in enumerate(results[:10], 1):  # tampilkan 10 teratas
        print(f"{i}. {res['judul']} ({res['score']})")
        print(f"   Kategori: {res['kategori']} | Tanggal: {res['tanggal']}")
        print(f"   Penulis: {res['penulis']}")
        print(f"   Gambar: {res['gambar']}")
        print(f"   Link: {res['link']}\n")
else:
    print("Tidak ditemukan hasil yang relevan.")
