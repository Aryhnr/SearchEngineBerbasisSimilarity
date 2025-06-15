import json
import pickle
import math

# Parameter BM25+
k1 = 1.5
b = 0.75
delta = 1.0

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

with open('./data_berita_bersih.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

N = len(doc_term_freq)

# Fungsi BM25+
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

# Fungsi relevansi heuristik
def is_relevant(query, doc_text, threshold=0.7):
    query_tokens = set(query.lower().split())
    doc_tokens = set(doc_text.lower().split())
    intersection = query_tokens.intersection(doc_tokens)
    similarity = len(intersection) / len(query_tokens) if query_tokens else 0
    return similarity >= threshold

# Evaluasi
queries = [
    "kylian mbappe sepatu emas eropa",
    "marc Marquez crash",
    "real madrid liga champions",
    "megawati korea"
]

k = 10
print(f"Evaluasi Precision@{k} untuk beberapa query...\n")

for query in queries:
    relevant_docs = set()
    for i, doc in enumerate(data):
        text = doc['judul_bersih'] + " " + doc['konten_bersih']
        if is_relevant(query, text):
            relevant_docs.add(i)

    query_tokens = query.lower().split()
    scores = []
    for i, freq in enumerate(doc_term_freq):
        score = bm25plus_score(query_tokens, freq, doc_lengths[i])
        if score > 0:
            scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)

    top_k = scores[:k]

    if len(top_k) == 0:
        precision = 0
    else:
        relevant_in_top_k = sum(1 for doc_id, _ in top_k if doc_id in relevant_docs)
        precision = relevant_in_top_k / len(top_k)

    print(f"Query: {query}")
    print(f"Relevant docs (heuristik): {len(relevant_docs)}")
    print(f"Retrieved docs (top {k}): {len(top_k)}")
    print(f"Precision@{k}: {precision:.4f}\n")
