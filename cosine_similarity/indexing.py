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
N = len(data)

metadata = []

for item in data:
    text = item['judul_bersih'] + ' ' + item['konten_bersih']
    tokens = text.split()

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

# Hitung IDF
idf = {}
for word, df in document_freq.items():
    idf[word] = math.log(N / (1 + df))

# Hitung TF-IDF
tfidf_vectors = []
for freq in doc_term_freq:
    tfidf = {}
    for word, count in freq.items():
        tfidf[word] = count * idf[word]
    tfidf_vectors.append(tfidf)

# Simpan ke file
with open('data/tfidf_vectors.pkl', 'wb') as f:
    pickle.dump(tfidf_vectors, f)

with open('data/idf.pkl', 'wb') as f:
    pickle.dump(idf, f)

with open('data/meta.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("Indexing selesai dan data disimpan.")
