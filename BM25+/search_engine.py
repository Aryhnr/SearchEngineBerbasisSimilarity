import pickle
import math

# Parameter BM25+
k1 = 1.5
b = 0.75
delta = 1.0  # offset tambahan BM25+

# Load data
with open('data/doc_term_freq_bm25plus.pkl', 'rb') as f:
    doc_term_freq = pickle.load(f)

with open('data/doc_lengths_bm25plus.pkl', 'rb') as f:
    doc_lengths = pickle.load(f)

with open('data/df_bm25plus.pkl', 'rb') as f:
    document_freq = pickle.load(f)

with open('data/avgdl_bm25plus.pkl', 'rb') as f:
    avgdl = pickle.load(f)

with open('data/meta_bm25plus.pkl', 'rb') as f:
    metadata = pickle.load(f)

N = len(doc_term_freq)

# Hitung skor BM25+
def bm25plus_score(query_tokens, freq, doc_len):
    score = 0.0
    for token in query_tokens:
        if token in document_freq:
            df = document_freq[token]
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            f = freq.get(token, 0)
            denom = f + k1 * (1 - b + b * (doc_len / avgdl))
            score += idf * ((f + delta) * (k1 + 1)) / (denom + delta)
    return score

# Input query
query = input("Masukkan kata kunci pencarian: ").lower()
query_tokens = query.split()

# Hitung skor
results = []
for i, freq in enumerate(doc_term_freq):
    score = bm25plus_score(query_tokens, freq, doc_lengths[i])
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
    for i, res in enumerate(results[:10], 1):
        print(f"{i}. {res['judul']} ({res['score']})")
        print(f"   Kategori: {res['kategori']} | Tanggal: {res['tanggal']}")
        print(f"   Penulis: {res['penulis']}")
        print(f"   Gambar: {res['gambar']}")
        print(f"   Link: {res['link']}\n")
else:
    print("Tidak ditemukan hasil yang relevan.")
