import json
import pickle
import os
import math
from collections import defaultdict

# Membaca data
with open('./data_berita_bersih.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Membuat folder data
os.makedirs('data', exist_ok=True)

# Parameter BM25
k1 = 1.5
b = 0.75

# Tokenisasi dan indexing
doc_term_freq = []
document_freq = defaultdict(int)
doc_lengths = []
metadata = []

for item in data:
    text = item['judul_bersih'] + ' ' + item['konten_bersih']
    tokens = text.split()
    doc_lengths.append(len(tokens))

    freq = defaultdict(int)
    for token in tokens:
        freq[token] += 1
    doc_term_freq.append(freq)

    for word in set(tokens):
        document_freq[word] += 1

    metadata.append({
        'judul': item['judul'],
        'kategori': item['kategori'],
        'tanggal': item['tanggal'],
        'link': item['link'],
        'penulis': item['penulis'],
        'gambar': item['gambar']
    })

N = len(data)
avgdl = sum(doc_lengths) / N

# Simpan semua ke file
with open('data/bm25_term_freq.pkl', 'wb') as f:
    pickle.dump(doc_term_freq, f)

with open('data/bm25_doc_freq.pkl', 'wb') as f:
    pickle.dump(document_freq, f)

with open('data/bm25_doc_lengths.pkl', 'wb') as f:
    pickle.dump(doc_lengths, f)

with open('data/bm25_meta.pkl', 'wb') as f:
    pickle.dump(metadata, f)

with open('data/bm25_params.pkl', 'wb') as f:
    pickle.dump({'N': N, 'avgdl': avgdl, 'k1': k1, 'b': b}, f)

print("Indexing BM25 selesai dan data disimpan.")
