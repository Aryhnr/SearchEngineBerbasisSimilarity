import json
import pickle
import math

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

with open('./data_berita_bersih.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

N = params['N']
avgdl = params['avgdl']
k1 = params['k1']
b = params['b']

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

# Fungsi cek relevansi otomatis (heuristik)
def is_relevant(query, doc_text, threshold=0.7):
    query_tokens = set(query.lower().split())
    doc_tokens = set(doc_text.lower().split())
    intersection = query_tokens.intersection(doc_tokens)
    similarity = len(intersection) / len(query_tokens) if query_tokens else 0
    return similarity >= threshold

# Evaluasi beberapa query
queries = [
    "kylian mbappe sepatu emas eropa",
    "marc Marquez crash",
    "real madrid liga champions",
    "megawati korea"
]

k = 10

print(f"Evaluasi Precision@{k} untuk BM25\n")

for query in queries:
    query_tokens = query.lower().split()

    relevant_docs = set()
    for i, doc in enumerate(data):
        text = doc['judul_bersih'] + " " + doc['konten_bersih']
        if is_relevant(query, text):
            relevant_docs.add(i)

    scores = []
    for i, tf_doc in enumerate(doc_term_freq):
        score = bm25_score(query_tokens, tf_doc, document_freq, N, doc_lengths[i], avgdl, k1, b)
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
