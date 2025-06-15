import pickle
import math

# Fungsi BM25 scoring
def bm25_score(query_tokens, tf_doc, df, N, dl, avgdl, k1=1.5, b=0.75):
    score = 0
    for term in query_tokens:
        if term not in tf_doc or term not in df:
            continue
        f = tf_doc[term]
        df_term = df[term]
        idf = math.log((N - df_term + 0.5) / (df_term + 0.5) + 1)
        denom = f + k1 * (1 - b + b * dl / avgdl)
        score += idf * ((f * (k1 + 1)) / denom)
    return score

# Load data
with open('data/bm25_term_freq.pkl', 'rb') as f:
    doc_term_freq = pickle.load(f)

with open('data/bm25_doc_freq.pkl', 'rb') as f:
    document_freq = pickle.load(f)

with open('data/bm25_doc_lengths.pkl', 'rb') as f:
    doc_lengths = pickle.load(f)

with open('data/bm25_meta.pkl', 'rb') as f:
    metadata = pickle.load(f)

with open('data/bm25_params.pkl', 'rb') as f:
    params = pickle.load(f)

N = params['N']
avgdl = params['avgdl']
k1 = params['k1']
b = params['b']

# Input query
query = input("Masukkan kata kunci pencarian: ").lower()
query_tokens = query.split()

# Hitung skor BM25
results = []
for i, tf_doc in enumerate(doc_term_freq):
    score = bm25_score(query_tokens, tf_doc, document_freq, N, doc_lengths[i], avgdl, k1, b)
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

# Urutkan hasil
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
