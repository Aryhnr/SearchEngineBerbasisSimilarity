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

# Tokenisasi dan indexing
doc_term_freq = []
document_freq = defaultdict(int)
doc_lengths = []
N = len(data)
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

avgdl = sum(doc_lengths) / len(doc_lengths)

# Simpan ke file
with open('data/doc_term_freq_bm25plus.pkl', 'wb') as f:
    pickle.dump(doc_term_freq, f)

with open('data/doc_lengths_bm25plus.pkl', 'wb') as f:
    pickle.dump(doc_lengths, f)

with open('data/df_bm25plus.pkl', 'wb') as f:
    pickle.dump(document_freq, f)

with open('data/avgdl_bm25plus.pkl', 'wb') as f:
    pickle.dump(avgdl, f)

with open('data/meta_bm25plus.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("Indexing BM25+ selesai dan data disimpan.")
