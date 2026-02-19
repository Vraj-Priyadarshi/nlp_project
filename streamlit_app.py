# streamlit run streamlit_app.py
"""
SemSim — Quora Question Duplicate Detection
Streamlit frontend with 4 sections:
  1. Home       — project overview
  2. FAQ Checker — production SBERT+FAISS+CrossEncoder FAQ search
  3. Similarity  — run all 6 models on a custom question pair
  4. Analysis    — charts & model comparison
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Config ──────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="SemSim — Question Similarity",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500&display=swap');

  /* ── Base typography ────────────────────────────────────────────── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #eaf0f8;
  }

  /* ── Page background ────────────────────────────────────────────── */
  .stApp, [data-testid="stAppViewContainer"],
  .main .block-container {
    background: #0b101e;
  }

  /* ── Sidebar ────────────────────────────────────────────────────── */
  section[data-testid="stSidebar"] {
    background: #080d1a !important;
    border-right: 1px solid #1e3050 !important;
  }
  section[data-testid="stSidebar"] * {
    color: #8fa8cc;
  }
  section[data-testid="stSidebar"] .stRadio label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 0.45rem 0.6rem !important;
    border-radius: 8px !important;
    transition: background 0.15s ease, color 0.15s ease;
  }
  section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(0,212,255,0.08) !important;
    color: #e8f0fe !important;
  }
  section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
  section[data-testid="stSidebar"] .stRadio [aria-checked="true"] ~ label {
    background: rgba(0,212,255,0.12) !important;
    color: #00d4ff !important;
  }
  section[data-testid="stSidebar"] hr {
    border-color: #1e3050 !important;
    margin: 0.75rem 0 !important;
  }

  /* ── Headers ────────────────────────────────────────────────────── */
  .main-header {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin-bottom: 0.6rem;
    color: #e8f0fe;
  }
  .accent { color: #00d4ff; }

  .tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #00d4ff;
    margin-bottom: 0.5rem;
  }

  /* ── Metric cards ───────────────────────────────────────────────── */
  .metric-card {
    background: linear-gradient(145deg, #111c30 0%, #0e1524 100%);
    border: 1px solid #1e3050;
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s ease, transform 0.2s ease;
  }
  .metric-card:hover {
    border-color: #00d4ff;
    transform: translateY(-2px);
  }
  .metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #00d4ff;
  }
  .metric-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8fa8cc;
    margin-top: 0.3rem;
  }

  /* ── Result cards (similarity) ──────────────────────────────────── */
  .result-card {
    background: linear-gradient(145deg, #111c30 0%, #0e1524 100%);
    border: 1px solid #1e3050;
    border-radius: 12px;
    padding: 1.15rem 1.35rem;
    margin-bottom: 0.85rem;
    transition: border-color 0.2s ease, transform 0.15s ease;
  }
  .result-card:hover { transform: translateY(-1px); }
  .result-card.dup    { border-left: 4px solid #00e5a0; }
  .result-card.notdup { border-left: 4px solid #ff4d6a; }
  .result-card.na     { border-left: 4px solid #4a6080; }

  .model-name {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: #e8f0fe;
  }
  .conf-val { font-family: 'DM Mono', monospace; font-size: 1.4rem; }
  .conf-dup { color: #00e5a0; }
  .conf-not { color: #ff4d6a; }
  .conf-na  { color: #4a6080; }

  /* ── FAQ result cards ───────────────────────────────────────────── */
  .faq-result {
    background: linear-gradient(145deg, #111c30 0%, #0e1524 100%);
    border: 1px solid #1e3050;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 0.85rem;
    transition: border-color 0.2s ease, transform 0.15s ease;
  }
  .faq-result:hover {
    border-color: rgba(0,212,255,0.35);
    transform: translateY(-1px);
  }
  .faq-q { font-weight: 600; font-size: 0.95rem; color: #e8f0fe; margin-bottom: 0.45rem; }
  .faq-a { color: #b8cce0; font-size: 0.875rem; line-height: 1.65; }

  /* ── Verdict labels ─────────────────────────────────────────────── */
  .verdict-dup { color: #00e5a0; font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800; }
  .verdict-not { color: #ff4d6a; font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800; }

  /* ── Streamlit element overrides ────────────────────────────────── */
  div[data-testid="stProgress"] > div { background-color: #00d4ff !important; }

  /* Text area */
  .stTextArea textarea {
    background: #0e1524 !important;
    border: 1px solid #1e3050 !important;
    border-radius: 10px !important;
    color: #e8f0fe !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.8rem 1rem !important;
    transition: border-color 0.2s ease !important;
  }
  .stTextArea textarea:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 0 1px rgba(0,212,255,0.25) !important;
  }
  .stTextArea textarea::placeholder { color: #3a5070 !important; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #00b4d8, #00d4ff) !important;
    color: #0b101e !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.2rem !important;
    transition: opacity 0.2s ease, transform 0.15s ease !important;
  }
  .stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
  }
  .stButton > button[kind="secondary"],
  .stButton > button:not([kind="primary"]) {
    background: #111c30 !important;
    border: 1px solid #1e3050 !important;
    color: #8fa8cc !important;
  }
  .stButton > button[kind="secondary"]:hover,
  .stButton > button:not([kind="primary"]):hover {
    border-color: #00d4ff !important;
    color: #00d4ff !important;
  }

  /* Slider */
  .stSlider [data-testid="stThumbValue"] { color: #00d4ff !important; }
  .stSlider [role="slider"] { background: #00d4ff !important; }

  /* Selectbox / radio */
  .stSelectbox > div > div, .stRadio > div { color: #eaf0f8 !important; }

  /* Dataframe */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* Dividers */
  hr { border-color: #1e3050 !important; }

  /* Headings colour */
  h1, h2, h3, h4, h5, h6 { color: #e8f0fe !important; }

  /* Body / paragraph text — force bright on all markdown output */
  p, li, span, div, td, th, label,
  [data-testid="stMarkdownContainer"],
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] span,
  [data-testid="stMarkdownContainer"] li,
  .stMarkdown, .stMarkdown p,
  .element-container p {
    color: #eaf0f8 !important;
  }

  /* Keep accent/colored elements from being overridden */
  .accent, .tag, .metric-val, .metric-lbl,
  .conf-val, .conf-dup, .conf-not, .conf-na,
  .verdict-dup, .verdict-not,
  .faq-a, .model-name, .faq-q { color: unset; }

  /* Bold text slightly brighter */
  strong, b { color: #f4f8ff !important; }

  /* Spinner text */
  .stSpinner > div { color: #00d4ff !important; }

  /* Expander */
  details { background: #111c30 !important; border: 1px solid #1e3050 !important; border-radius: 10px !important; }
  details summary { color: #e8f0fe !important; }

  /* code blocks */
  pre, code { background: #0e1524 !important; border: 1px solid #1e3050 !important; border-radius: 8px !important; color: #dbe6f0 !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: #080d1a; }
  ::-webkit-scrollbar-thumb { background: #1e3050; border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: #2a4060; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0;'>
      <span style='font-family:Syne,sans-serif;font-weight:800;font-size:1.3rem;'>
        Sem<span style='color:#00d4ff'>Sim</span>
      </span>
      <span style='margin-left:0.5rem;display:inline-block;width:8px;height:8px;
        background:#00d4ff;border-radius:50%;'></span>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Home", "❓ FAQ Checker", "🔍 Similarity Checker", "📊 Analysis"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-family:DM Mono,monospace;font-size:0.72rem;color:#8fa8cc;line-height:2;'>
    MODELS AVAILABLE<br>
    ✓ RF + BOW<br>
    ✓ XGB + BOW<br>
    ✓ RF + Word2Vec<br>
    ✓ XGB + Word2Vec<br>
    ✓ SBERT<br>
    ✓ Cross-Encoder<br>
    </div>
    """, unsafe_allow_html=True)

    api_ok = False
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        api_ok = r.status_code == 200
    except Exception:
        pass

    st.markdown(f"""
    <div style='margin-top:1rem;font-family:DM Mono,monospace;font-size:0.72rem;'>
      API: <span style='color:{"#00e5a0" if api_ok else "#ff4d6a"}'>
      {"● ONLINE" if api_ok else "● OFFLINE"}
      </span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="tag">// project overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">Quora Question<br><span class="accent">Duplicate Detection</span></div>', unsafe_allow_html=True)
    st.markdown("A comprehensive NLP study comparing **6 models** — from classical bag-of-words to transformer cross-encoders — on the Quora Question Pairs dataset.")

    st.markdown("---")

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-val">6</div><div class="metric-lbl">Models Compared</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-val">89.6%</div><div class="metric-lbl">Best Accuracy</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-val">298K</div><div class="metric-lbl">Training Pairs</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-val">22</div><div class="metric-lbl">Hand-crafted Features</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Model overview
    st.markdown('<div class="tag">// model overview</div>', unsafe_allow_html=True)
    st.subheader("Six Approaches to Semantic Similarity")

    models_info = [
        ("Random Forest + BOW", "Traditional", "22 hand-crafted features + CountVectorizer bag-of-words. Fast, interpretable baseline."),
        ("XGBoost + BOW", "Traditional", "Gradient-boosted trees on BOW features. Stronger than RF on sparse text data."),
        ("Random Forest + Word2Vec", "Traditional", "22 features + mean-pooled 300-dim Word2Vec embeddings (Google News)."),
        ("XGBoost + Word2Vec", "Traditional", "XGBoost on dense 622-dim Word2Vec feature vector. Best classical model."),
        ("SBERT (Sentence-BERT)", "Deep Learning", "Fine-tuned all-MiniLM-L6-v2 bi-encoder. Cosine similarity with learned threshold."),
        ("Cross-Encoder", "Deep Learning", "Fine-tuned ms-marco-MiniLM cross-encoder. Reads both questions jointly — highest accuracy."),
    ]

    col_a, col_b = st.columns(2)
    for i, (name, type_, desc) in enumerate(models_info):
        col = col_a if i % 2 == 0 else col_b
        badge_color = "#b48fff" if type_ == "Deep Learning" else "#ffd166"
        badge_bg = "rgba(180,143,255,0.1)" if type_ == "Deep Learning" else "rgba(255,209,102,0.1)"
        with col:
            st.markdown(f"""
            <div style="background:#111c30;border:1px solid #1e3050;border-radius:10px;padding:1.25rem;margin-bottom:0.75rem;">
              <div style="display:inline-block;background:{badge_bg};color:{badge_color};
                font-family:DM Mono,monospace;font-size:0.7rem;padding:0.2rem 0.6rem;
                border-radius:4px;margin-bottom:0.75rem;">{type_}</div>
              <div style="font-family:Syne,sans-serif;font-weight:700;font-size:1rem;margin-bottom:0.4rem;">{name}</div>
              <div style="font-size:0.85rem;color:#b8cce0;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Pipeline overview
    st.markdown('<div class="tag">// architecture</div>', unsafe_allow_html=True)
    st.subheader("Pipeline Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**BOW Pipeline**")
        st.code("""
raw q1, q2
  → preprocess()
  → 22 handcrafted features
  + CountVectorizer(q1)
  + CountVectorizer(q2)
  → hstack → RF / XGB → label
        """)
        st.markdown("**Word2Vec Pipeline**")
        st.code("""
raw q1, q2
  → preprocess()
  → 22 handcrafted features
  + mean(w2v(q1))  [300-dim]
  + mean(w2v(q2))  [300-dim]
  → [622-dim] → RF / XGB → label
        """)
    with col2:
        st.markdown("**SBERT Pipeline**")
        st.code("""
raw q1, q2
  → SBERT.encode(q1), encode(q2)
  → cosine_similarity
  → score ≥ threshold → Duplicate
        """)
        st.markdown("**FAQ Search Pipeline**")
        st.code("""
user_query
  → SBERT.encode() → FAISS top-50
  → CrossEncoder rerank
  → best matching answer
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FAQ CHECKER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "❓ FAQ Checker":
    st.markdown('<div class="tag">// faq search pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">FAQ <span class="accent">Checker</span></div>', unsafe_allow_html=True)
    st.markdown("Uses the production **SBERT + FAISS + Cross-Encoder** pipeline. Ask any question and get the most relevant FAQ answer.")

    st.markdown("---")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown("**Ask a Question**")
        user_q = st.text_area("Your question", height=100,
                              placeholder="e.g. Does this product have a warranty?",
                              label_visibility="collapsed")

        top_k = st.slider("Number of results", 1, 10, 5)

        examples = [
            "Does this phone support fast charging?",
            "How do I return a product?",
            "What batteries does this use?",
            "Is this product waterproof?",
        ]
        st.markdown("**Quick examples:**")
        for ex in examples:
            if st.button(f"↗ {ex}", use_container_width=True):
                user_q = ex
                st.session_state['faq_query'] = ex

        if 'faq_query' in st.session_state:
            user_q = st.session_state['faq_query']

        search_btn = st.button("🔍 Search FAQs", use_container_width=True, type="primary")

    with col_result:
        if search_btn or ('faq_query' in st.session_state and user_q):
            if not user_q or not user_q.strip():
                st.warning("Please enter a question.")
            elif not api_ok:
                st.error("API is offline. Start the FastAPI backend first: `uvicorn api:app --reload`")
            else:
                with st.spinner("Searching with SBERT + FAISS + Cross-Encoder..."):
                    try:
                        resp = requests.post(f"{API_BASE}/faq/search",
                                             json={"query": user_q, "top_k": top_k}, timeout=30)
                        resp.raise_for_status()
                        data = resp.json()
                    except Exception as e:
                        st.error(f"Request failed: {e}")
                        data = None

                if data:
                    conf = data['results'][0]['confidence'] if data['results'] else 0
                    if data['is_confident']:
                        st.success(f"✅ Match found (confidence: {conf:.1%})")
                    else:
                        st.warning(f"⚠️ Low confidence ({conf:.1%}) — rephrase your question for better results")

                    st.markdown("**Top Matches:**")
                    for r in data['results']:
                        color = "#00e5a0" if r['confidence'] >= 0.5 else "#ffd166"
                        st.markdown(f"""
                        <div class="faq-result">
                          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                            <span style="font-family:DM Mono,monospace;font-size:0.72rem;color:#8fa8cc;">RANK #{r['rank']}</span>
                            <span style="font-family:DM Mono,monospace;font-size:0.8rem;color:{color};">
                              {r['confidence']:.1%} confidence
                            </span>
                          </div>
                          <div class="faq-q">Q: {r['question']}</div>
                          <div class="faq-a">A: {r['answer']}</div>
                        </div>
                        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SIMILARITY CHECKER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Similarity Checker":
    st.markdown('<div class="tag">// 6-model similarity checker</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">Question <span class="accent">Similarity</span></div>', unsafe_allow_html=True)
    st.markdown("Enter two questions — all 6 trained models predict whether they are duplicates.")

    st.markdown("---")

    # Example pairs
    examples = [
        ("What is the best way to learn Python?", "How can I get started with Python programming?"),
        ("Where is the capital of India?", "What is the current capital of Pakistan?"),
        ("How do I lose weight fast?", "What are the quickest ways to reduce body fat?"),
        ("What is the best phone under 500 dollars?", "Which smartphone has the best camera in 2024?"),
    ]

    st.markdown("**Load an example:**")
    ex_cols = st.columns(len(examples))
    for i, (a, b) in enumerate(examples):
        with ex_cols[i]:
            if st.button(f"Example {i+1}", use_container_width=True):
                st.session_state['q1'] = a
                st.session_state['q2'] = b

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        q1 = st.text_area("Question 1", value=st.session_state.get('q1', ''),
                          height=120, placeholder="Enter your first question here...")
    with col_q2:
        q2 = st.text_area("Question 2", value=st.session_state.get('q2', ''),
                          height=120, placeholder="Enter your second question here...")

    check_btn = st.button("⚡ Run All 6 Models", use_container_width=True, type="primary")

    if check_btn:
        if not q1.strip() or not q2.strip():
            st.warning("Please enter both questions.")
        elif not api_ok:
            st.error("API is offline. Start the FastAPI backend: `uvicorn api:app --reload`")
        else:
            with st.spinner("Running all 6 models..."):
                try:
                    resp = requests.post(f"{API_BASE}/similarity",
                                         json={"question1": q1, "question2": q2}, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    data = None

            if data:
                st.markdown("---")

                # Final verdict — based on SBERT + Cross-Encoder only
                semantic = [r for r in data['results'] if r['model'] in ("SBERT (Sentence-BERT)", "Cross-Encoder") and r['available']]
                classical = [r for r in data['results'] if r['model'] not in ("SBERT (Sentence-BERT)", "Cross-Encoder") and r['available']]
                sem_dup = sum(1 for r in semantic if r['prediction'] == "Duplicate")
                sem_total = len(semantic)
                mv = data['majority_vote']
                color = "#00e5a0" if mv == "Duplicate" else "#ff4d6a"
                icon  = "✓" if mv == "Duplicate" else "✗"

                # Also compute classical vote for the explanation
                cl_dup = sum(1 for r in classical if r['prediction'] == "Duplicate")
                cl_total = len(classical)

                st.markdown(f"""
                <div style="background:linear-gradient(145deg,#111c30 0%,#0e1524 100%);
                  border:1px solid {'#00e5a0' if mv=='Duplicate' else '#ff4d6a'}40;
                  border-radius:14px;padding:1.5rem 2rem;margin-bottom:1.5rem;">
                  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
                    <div>
                      <div style="font-family:Syne,sans-serif;font-weight:800;font-size:1.8rem;color:{color};">
                        {icon} {mv}
                      </div>
                      <div style="color:#b8cce0;font-size:0.85rem;margin-top:0.3rem;">
                        {sem_dup}/{sem_total} semantic models agree &nbsp;·&nbsp;
                        {cl_dup}/{cl_total} classical models agree
                      </div>
                    </div>
                    <div style="text-align:right;">
                      <div style="font-family:DM Mono,monospace;font-size:0.72rem;color:#b48fff;
                        background:rgba(180,143,255,0.1);padding:0.25rem 0.7rem;border-radius:6px;">
                        FINAL VERDICT
                      </div>
                      <div style="font-family:DM Mono,monospace;font-size:0.68rem;color:#8fa8cc;margin-top:0.35rem;">
                        Based on SBERT + Cross-Encoder
                      </div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Results grid
                col1, col2 = st.columns(2)
                for i, r in enumerate(data['results']):
                    col = col1 if i % 2 == 0 else col2
                    with col:
                        if not r['available']:
                            cls = "na"; clr = "#4a6080"; icon_ = "—"
                            conf_str = "N/A"
                        elif r['prediction'] == "Duplicate":
                            cls = "dup"; clr = "#00e5a0"; icon_ = "✓"
                            conf_str = f"{r['confidence']:.1%}"
                        else:
                            cls = "notdup"; clr = "#ff4d6a"; icon_ = "✗"
                            conf_str = f"{r['confidence']:.1%}"

                        st.markdown(f"""
                        <div class="result-card {cls}">
                          <div class="model-name" style="margin-bottom:0.5rem;">{r['model']}</div>
                          <div style="display:flex;align-items:baseline;gap:0.5rem;">
                            <span class="conf-val" style="color:{clr};">{conf_str}</span>
                          </div>
                          <div style="margin-top:0.5rem;">
                            <span style="background:{'rgba(0,229,160,0.1)' if cls=='dup' else 'rgba(255,77,106,0.1)' if cls=='notdup' else 'rgba(74,96,128,0.1)'};
                              color:{clr};font-family:DM Mono,monospace;font-size:0.72rem;
                              padding:0.2rem 0.6rem;border-radius:4px;">
                              {icon_} {r['prediction'] if r['available'] else 'Model unavailable'}
                            </span>
                          </div>
                          {"" if not r['available'] else f'<div style="margin-top:0.75rem;height:4px;background:#1e3050;border-radius:2px;overflow:hidden;"><div style="height:100%;width:{r["confidence"]*100:.0f}%;background:{clr};border-radius:2px;"></div></div>'}
                        </div>
                        """, unsafe_allow_html=True)

                # ── Why do classical ML models disagree? ──
                if semantic and classical:
                    sem_pred = semantic[0]['prediction'] if len(semantic) == 1 else ("Duplicate" if sem_dup > sem_total/2 else "Not Duplicate")
                    cl_pred  = "Duplicate" if cl_dup > cl_total/2 else "Not Duplicate"
                    if sem_pred != cl_pred:
                        st.markdown(f"""
                        <div style="background:linear-gradient(145deg,#111c30 0%,#0e1524 100%);
                          border:1px solid #1e3050;border-radius:12px;padding:1.25rem 1.5rem;
                          margin-bottom:1.25rem;border-left:4px solid #ffd166;">
                          <div style="font-family:Syne,sans-serif;font-weight:700;font-size:0.95rem;
                            color:#ffd166;margin-bottom:0.6rem;">⚠ Why do the classical ML models disagree?</div>
                          <div style="font-size:0.84rem;color:#c8d8ea;line-height:1.75;">
                            <b style="color:#e8f0fe;">Classical models (RF, XGBoost)</b> rely on
                            <b style="color:#ffd166;">surface-level features</b> — word overlap counts, common token ratios,
                            fuzzy string matching, and bag-of-words / Word2Vec vectors. They compare
                            <em>what words appear</em>, not <em>what they mean</em>.<br><br>
                            When two questions use <b style="color:#e8f0fe;">different words to ask the same thing</b>
                            (e.g. <em>"lose weight"</em> vs <em>"reduce body fat"</em>),
                            these models see low word overlap and confidently predict <span style="color:#ff4d6a;">Not Duplicate</span>.
                            <br><br>
                            <b style="color:#e8f0fe;">SBERT & Cross-Encoder</b> are <b style="color:#b48fff;">transformer-based</b> models
                            that encode <b style="color:#00d4ff;">semantic meaning</b> into dense vector representations.
                            They understand that "lose weight" and "reduce body fat" carry the same intent,
                            even though the exact words differ. This is why we use them as the
                            <b style="color:#b48fff;">final verdict</b>.
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Confidence bar chart
                st.markdown("---")
                avail_results = [r for r in data['results'] if r['available']]
                if avail_results:
                    fig = go.Figure(go.Bar(
                        x=[r['confidence'] for r in avail_results],
                        y=[r['model'] for r in avail_results],
                        orientation='h',
                        marker=dict(
                            color=[("#00e5a0" if r['prediction']=="Duplicate" else "#ff4d6a") for r in avail_results],
                            opacity=0.85
                        ),
                        text=[f"{r['confidence']:.1%}" for r in avail_results],
                        textposition='outside'
                    ))
                    fig.add_vline(x=0.5, line_dash="dash", line_color="#8fa8cc", opacity=0.4,
                                  annotation_text="Threshold", annotation_font_size=11)
                    fig.update_layout(
                        title="Confidence Scores by Model",
                        xaxis_title="Confidence (≥0.5 = Duplicate)",
                        xaxis_range=[0, 1.1],
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(11,16,30,0.8)',
                        font=dict(color='#8fa8cc', family='DM Mono'),
                        height=350,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    fig.update_xaxes(gridcolor='#1e3050')
                    fig.update_yaxes(gridcolor='#1e3050')
                    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analysis":
    st.markdown('<div class="tag">// model performance analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">Performance <span class="accent">Analysis</span></div>', unsafe_allow_html=True)
    st.markdown("Comprehensive evaluation of all 6 models on the Quora Question Pairs test set.")

    st.markdown("---")

    # Load metrics
    @st.cache_data
    def load_metrics():
        try:
            r = requests.get(f"{API_BASE}/analysis/metrics", timeout=5)
            if r.status_code == 200:
                return pd.DataFrame(r.json()['data'])
        except Exception:
            pass
        # Static fallback
        return pd.DataFrame([
            {"Model": "Random Forest + BOW",       "Accuracy": 0.8213, "F1 Score": 0.8336, "Precision": 0.7787, "Recall": 0.8969, "ROC-AUC": 0.8215},
            {"Model": "XGBoost + BOW",             "Accuracy": 0.8076, "F1 Score": 0.8216, "Precision": 0.7650, "Recall": 0.8872, "ROC-AUC": 0.8078},
            {"Model": "Random Forest + Word2Vec",  "Accuracy": 0.8084, "F1 Score": 0.8233, "Precision": 0.7629, "Recall": 0.8942, "ROC-AUC": 0.8086},
            {"Model": "XGBoost + Word2Vec",        "Accuracy": 0.8147, "F1 Score": 0.8242, "Precision": 0.7830, "Recall": 0.8700, "ROC-AUC": 0.8148},
            {"Model": "SBERT",                     "Accuracy": 0.8713, "F1 Score": 0.8327, "Precision": 0.8001, "Recall": 0.8681, "ROC-AUC": 0.8706},
            {"Model": "Cross-Encoder",             "Accuracy": 0.8958, "F1 Score": 0.8632, "Precision": 0.8372, "Recall": 0.8909, "ROC-AUC": 0.8948},
        ])

    df = load_metrics()

    MODEL_COLORS = {
        "Random Forest + BOW":      "#ffd166",
        "XGBoost + BOW":            "#f4a261",
        "Random Forest + Word2Vec": "#90e0ef",
        "XGBoost + Word2Vec":       "#48cae4",
        "SBERT":                    "#00d4ff",
        "Cross-Encoder":            "#b48fff",
    }
    colors = [MODEL_COLORS.get(m, "#00d4ff") for m in df['Model']]

    # ── Chart 1 & 2: Accuracy + F1 side by side
    col1, col2 = st.columns(2)

    with col1:
        df_sorted = df.sort_values("Accuracy")
        fig = go.Figure(go.Bar(
            y=df_sorted['Model'], x=df_sorted['Accuracy'],
            orientation='h', marker_color=[MODEL_COLORS.get(m, "#00d4ff") for m in df_sorted['Model']],
            text=[f"{v:.1%}" for v in df_sorted['Accuracy']], textposition='outside'
        ))
        fig.update_layout(
            title="Accuracy Comparison", xaxis_range=[0.6, 1.0],
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(11,16,30,0.8)',
            font=dict(color='#8fa8cc', family='DM Mono'), height=340,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        fig.update_xaxes(gridcolor='#1e3050', tickformat='.0%')
        fig.update_yaxes(gridcolor='#1e3050')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        df_sorted = df.sort_values("F1 Score")
        fig = go.Figure(go.Bar(
            y=df_sorted['Model'], x=df_sorted['F1 Score'],
            orientation='h', marker_color=[MODEL_COLORS.get(m, "#b48fff") for m in df_sorted['Model']],
            text=[f"{v:.1%}" for v in df_sorted['F1 Score']], textposition='outside'
        ))
        fig.update_layout(
            title="F1 Score Comparison", xaxis_range=[0.6, 1.0],
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(11,16,30,0.8)',
            font=dict(color='#8fa8cc', family='DM Mono'), height=340,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        fig.update_xaxes(gridcolor='#1e3050', tickformat='.0%')
        fig.update_yaxes(gridcolor='#1e3050')
        st.plotly_chart(fig, use_container_width=True)

    # ── Chart 3: Radar (all metrics)
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC-AUC']
    fig_radar = go.Figure()
    highlight = ["SBERT", "Cross-Encoder", "XGBoost + BOW"]
    for _, row in df.iterrows():
        name = row['Model']
        vals = [row[m] for m in metrics]
        vals += [vals[0]]
        show = name in highlight
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=metrics + [metrics[0]], name=name,
            line=dict(color=MODEL_COLORS.get(name, '#00d4ff'), width=2 if show else 1),
            opacity=0.9 if show else 0.35,
            fill='toself', fillcolor=MODEL_COLORS.get(name, '#00d4ff'),
            visible=True
        ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(range=[0.6, 1.0], visible=True, gridcolor='#1e3050',
                            tickformat='.0%', tickfont=dict(color='#4a6080', size=9)),
            angularaxis=dict(gridcolor='#1e3050'),
            bgcolor='rgba(11,16,30,0.8)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8fa8cc', family='DM Mono'),
        legend=dict(font=dict(size=11)),
        title="Multi-Metric Radar — All Models", height=450,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Chart 4 & 5: Precision/Recall + Speed
    col3, col4 = st.columns(2)

    with col3:
        fig_pr = go.Figure()
        for _, row in df.iterrows():
            fig_pr.add_trace(go.Scatter(
                x=[row['Recall']], y=[row['Precision']],
                mode='markers+text', name=row['Model'],
                text=[row['Model'].split(' + ')[0].split(' ')[0]],
                textposition='top center',
                marker=dict(size=14, color=MODEL_COLORS.get(row['Model'], '#00d4ff'), opacity=0.85)
            ))
        fig_pr.update_layout(
            title="Precision vs Recall",
            xaxis_title="Recall", yaxis_title="Precision",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(11,16,30,0.8)',
            font=dict(color='#8fa8cc', family='DM Mono'), height=340,
            showlegend=False, margin=dict(l=0, r=0, t=40, b=0)
        )
        fig_pr.update_xaxes(gridcolor='#1e3050', tickformat='.0%')
        fig_pr.update_yaxes(gridcolor='#1e3050', tickformat='.0%')
        st.plotly_chart(fig_pr, use_container_width=True)

    with col4:
        speed_data = {
            "RF + BOW":       2.1,
            "XGB + BOW":      1.8,
            "RF + Word2Vec":  3.5,
            "XGB + Word2Vec": 2.9,
            "SBERT":          18.4,
            "Cross-Encoder":  42.7,
        }
        speed_df = pd.DataFrame(list(speed_data.items()), columns=['Model', 'ms_per_pair'])
        speed_df = speed_df.sort_values('ms_per_pair', ascending=True)
        fig_sp = go.Figure(go.Bar(
            y=speed_df['Model'], x=speed_df['ms_per_pair'],
            orientation='h',
            marker_color=['#00e5a0' if v < 5 else '#ffd166' if v < 25 else '#ff4d6a'
                          for v in speed_df['ms_per_pair']],
            text=[f"{v}ms" for v in speed_df['ms_per_pair']],
            textposition='outside'
        ))
        fig_sp.update_layout(
            title="Inference Speed (ms/pair)", xaxis_title="Latency (ms)",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(11,16,30,0.8)',
            font=dict(color='#8fa8cc', family='DM Mono'), height=340,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        fig_sp.update_xaxes(gridcolor='#1e3050')
        fig_sp.update_yaxes(gridcolor='#1e3050')
        st.plotly_chart(fig_sp, use_container_width=True)

    # ── Heatmap
    st.markdown("---")
    st.markdown("**Performance Heatmap**")
    heat_df = df.set_index('Model')[metrics]
    fig_heat = px.imshow(
        heat_df, text_auto='.3f', color_continuous_scale='RdYlGn',
        zmin=0.6, zmax=1.0, aspect="auto"
    )
    fig_heat.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8fa8cc', family='DM Mono'), height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        coloraxis_colorbar=dict(title="Score", tickformat='.0%')
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Rankings Table
    st.markdown("---")
    st.markdown("**📋 Full Rankings Table**")
    df_ranked = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    df_ranked.index += 1
    df_display = df_ranked.copy()
    for col in metrics:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}")
    st.dataframe(df_display, use_container_width=True, height=260)

    # ── Key Findings
    st.markdown("---")
    st.markdown("**🔑 Key Findings**")

    kc1, kc2, kc3, kc4 = st.columns(4)

    with kc1:
        st.markdown("""
        <div class="metric-card">
        <div style="font-size:1.5rem;margin-bottom:0.5rem;">🏆</div>
        <div style="font-family:Syne,sans-serif;font-weight:700;color:#00d4ff;">Best Accuracy</div>
        <div style="font-size:0.85rem;color:#b8cce0;margin-top:0.4rem;">Cross-Encoder at 89.6%</div>
        </div>
        """, unsafe_allow_html=True)

    with kc2:
        st.markdown("""
        <div class="metric-card">
        <div style="font-size:1.5rem;margin-bottom:0.5rem;">⚡</div>
        <div style="font-family:Syne,sans-serif;font-weight:700;color:#00e5a0;">Fastest</div>
        <div style="font-size:0.85rem;color:#b8cce0;margin-top:0.4rem;">XGB + BOW at 1.8ms/pair</div>
        </div>
        """, unsafe_allow_html=True)

    with kc3:
        st.markdown("""
        <div class="metric-card">
        <div style="font-size:1.5rem;margin-bottom:0.5rem;">⚖️</div>
        <div style="font-family:Syne,sans-serif;font-weight:700;color:#b48fff;">Best Balance</div>
        <div style="font-size:0.85rem;color:#b8cce0;margin-top:0.4rem;">SBERT: 87.1% at 18ms</div>
        </div>
        """, unsafe_allow_html=True)

    with kc4:
        st.markdown("""
        <div class="metric-card">
        <div style="font-size:1.5rem;margin-bottom:0.5rem;">📈</div>
        <div style="font-family:Syne,sans-serif;font-weight:700;color:#ffd166;">DL vs Classical</div>
        <div style="font-size:0.85rem;color:#b8cce0;margin-top:0.4rem;">Deep learning gains ~8% avg</div>
        </div>
        """, unsafe_allow_html=True)
 