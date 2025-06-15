import streamlit as st
import pickle
import math
import json
from PIL import Image
import requests
from streamlit_option_menu import option_menu

# --- Load metadata dan data mentah ---
with open('data/meta.pkl', 'rb') as f:
    metadata_cosine = pickle.load(f)
with open('data/bm25_meta.pkl', 'rb') as f:
    metadata_bm25 = pickle.load(f)
with open('data/meta_bm25plus.pkl', 'rb') as f:
    metadata_bm25plus = pickle.load(f)
with open('data_berita_bersih.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# --- FUNGSI COSINE ---
def cosine_similarity(vec1, vec2):
    dot = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in vec1)
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0

def cosine_search(query):
    with open('data/tfidf_vectors.pkl', 'rb') as f:
        vectors = pickle.load(f)
    with open('data/idf.pkl', 'rb') as f:
        idf = pickle.load(f)
    tokens = query.lower().split()
    q_freq = {t: tokens.count(t) for t in tokens if t in idf}
    q_tfidf = {t: c * idf[t] for t, c in q_freq.items()}
    results = [(i, cosine_similarity(q_tfidf, v)) for i, v in enumerate(vectors)]
    results = [(i, s) for i, s in results if s > 0]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:20], metadata_cosine

# --- FUNGSI BM25 ---
def bm25_score(qt, tf_doc, df, N, dl, avgdl, k1=1.5, b=0.75):
    score = 0
    for t in qt:
        if t not in tf_doc or t not in df: continue
        f, df_t = tf_doc[t], df[t]
        idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
        denom = f + k1 * (1 - b + b * dl / avgdl)
        score += idf * ((f * (k1 + 1)) / denom)
    return score

def bm25_search(query):
    with open('data/bm25_term_freq.pkl', 'rb') as f:
        doc_term_freq = pickle.load(f)
    with open('data/bm25_doc_freq.pkl', 'rb') as f:
        document_freq = pickle.load(f)
    with open('data/bm25_doc_lengths.pkl', 'rb') as f:
        doc_lengths = pickle.load(f)
    with open('data/bm25_params.pkl', 'rb') as f:
        params = pickle.load(f)
    N, avgdl = params['N'], params['avgdl']
    qt = query.lower().split()
    results = [(i, bm25_score(qt, tf, document_freq, N, doc_lengths[i], avgdl))
               for i, tf in enumerate(doc_term_freq)]
    results = [(i, s) for i, s in results if s > 0]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:20], metadata_bm25

# --- FUNGSI BM25+ ---
def bm25plus_score(qt, tf_doc, df, N, dl, avgdl, k1=1.5, b=0.75, delta=1.0):
    score = 0
    for t in qt:
        if t not in tf_doc or t not in df: continue
        f, df_t = tf_doc[t], df[t]
        idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
        denom = f + k1 * (1 - b + b * dl / avgdl)
        score += idf * ((f + delta) * (k1 + 1)) / (denom + delta)
    return score

def bm25plus_search(query):
    with open('data/doc_term_freq_bm25plus.pkl', 'rb') as f:
        doc_term_freq = pickle.load(f)
    with open('data/df_bm25plus.pkl', 'rb') as f:
        document_freq = pickle.load(f)
    with open('data/doc_lengths_bm25plus.pkl', 'rb') as f:
        doc_lengths = pickle.load(f)
    with open('data/avgdl_bm25plus.pkl', 'rb') as f:
        avgdl = pickle.load(f)
    N = len(doc_term_freq)
    qt = query.lower().split()
    results = [(i, bm25plus_score(qt, tf, document_freq, N, doc_lengths[i], avgdl))
               for i, tf in enumerate(doc_term_freq)]
    results = [(i, s) for i, s in results if s > 0]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:20], metadata_bm25plus

# --- UI ---
st.set_page_config(page_title="Search Engine Berita", layout="wide")

st.markdown("<h1 style='text-align: center; margin-bottom: 50px;'>Pencarian Berita</h1>", unsafe_allow_html=True)
# 1. MENU HORIZONTAL
selected = option_menu(
    menu_title=None,
    options=["Cosine Similarity", "BM25", "BM25+"],
    icons=["list-task", "list-task", "list-task"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f8f9fa"},
        "icon": {"color": "black", "font-size": "20px"},
        "nav-link": {"font-size": "18px", "text-align": "center", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#ff4b4b", "color": "white"},
    }
)

# 2. FORM DI TENGAH
center = st.columns([1, 4, 1])
with center[1]:
    c1, c2 = st.columns([5, 1])
    with c1:
        query = st.text_input("", placeholder="Masukkan kata kunci...", key="query_input")
    with c2:
        st.markdown("<div style='margin-top: 30px;'>", unsafe_allow_html=True)
        search = st.button("🔍", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# 3. HASIL PENCARIAN
# 3. HASIL PENCARIAN
if search and query.strip():
    if selected == "Cosine Similarity":
        results, meta = cosine_search(query)
    elif selected == "BM25":
        results, meta = bm25_search(query)
    else:
        results, meta = bm25plus_search(query)

    center = st.columns([1, 4, 1])
    with center[1]:  # hasil juga ditaruh di tengah
        if results:
            st.success(f"Ditemukan {len(results)} hasil relevan:")
            for idx, score in results:
                item = meta[idx]
                berita = raw_data[idx]
                col1, col2 = st.columns([1, 3])
                with col1:
                    try:
                        img = Image.open(requests.get(item['gambar'], stream=True).raw)
                        st.image(img, width=300)
                    except:
                        st.write("📷 [Gambar tidak tersedia]")
                with col2:
                    st.markdown(f"### [{item['judul']}]({item['link']})")
                    st.markdown(f"**Penulis**: {item['penulis']} | **Tanggal**: {item['tanggal']}")
                    st.markdown(f"**Score**: `{round(score, 4)}`")
                    st.markdown(berita['konten_bersih'][:150] + "...")
                    st.markdown("---")
        else:
            st.warning("Tidak ditemukan hasil relevan.")

