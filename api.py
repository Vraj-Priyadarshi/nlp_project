# uvicorn api:app --reload

"""
FastAPI backend for Quora Question Duplicate Detection
Loads all 6 trained models from joblib/directory artifacts.

Expected files in same directory:
  - count_vectorizer1.joblib
  - randomForest_BOW.joblib
  - XGBClassifier_BOW.joblib
  - RandomForestClassifier_WTV.joblib
  - XGBClassifier_WTV.joblib
  - sbert_quora_model/          (directory)
  - sbert_results.joblib        (contains best_threshold)
  - cross_encoder_quora/        (directory — optional, falls back to HF)
  - cross_encoder_results.joblib (contains best_threshold)
  - faq_faiss.index             (built offline)
  - faq_metadata.joblib         (questions + answers)
  - dataset/faq_data.csv        (FAQ source)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import re
import os
import joblib
import uvicorn

app = FastAPI(title="SemSim API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Globals (loaded once at startup) ─────────────────────────────────────────
cv = None           # CountVectorizer
rf_bow = None       # RF + BOW
xgb_bow = None      # XGB + BOW
rf_wtv = None       # RF + Word2Vec
xgb_wtv = None      # XGB + Word2Vec
w2v_model = None    # Word2Vec model (gensim)
sbert_model = None  # SBERT bi-encoder
sbert_thresh = 0.5  # best threshold from training
cross_model = None  # CrossEncoder
cross_thresh = 0.5  # best threshold from training
faiss_index = None  # FAISS index for FAQ search
faq_meta = None     # FAQ questions + answers dict

MODEL_STATUS = {}   # tracks which models loaded successfully


# ─── Preprocessing (must match notebook exactly) ──────────────────────────────

def preprocess(q):
    q = str(q).lower().strip()
    q = q.replace('%', ' percent').replace('$', ' dollar ').replace('₹', ' rupee ')
    q = q.replace('€', ' euro ').replace('@', ' at ').replace('[math]', '')
    q = q.replace(',000,000,000 ', 'b ').replace(',000,000 ', 'm ').replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    contractions = {
        "ain't": "am not", "aren't": "are not", "can't": "cannot", "could've": "could have",
        "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not",
        "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
        "he'll": "he will", "he's": "he is", "i'd": "i would", "i'll": "i will",
        "i'm": "i am", "i've": "i have", "isn't": "is not", "it's": "it is",
        "let's": "let us", "shouldn't": "should not", "that's": "that is", "there's": "there is",
        "they'd": "they would", "they'll": "they will", "they're": "they are",
        "they've": "they have", "wasn't": "was not", "we'd": "we would", "we'll": "we will",
        "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
        "what're": "what are", "what's": "what is", "what've": "what have", "where's": "where is",
        "who'll": "who will", "who's": "who is", "won't": "will not", "wouldn't": "would not",
        "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have",
    }
    q_decontracted = []
    for word in q.split():
        word = contractions.get(word, word)
        q_decontracted.append(word)
    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have").replace("n't", " not").replace("'re", " are").replace("'ll", " will")
    q = re.sub(r'\W', ' ', q)
    q = re.sub(r'\s+', ' ', q).strip()
    return q


# ─── Feature Engineering (mirrors notebook) ───────────────────────────────────

def safe_divide(a, b, fallback=0.0):
    return a / b if b != 0 else fallback

def get_handcrafted_features(q1, q2):
    """Returns the 22-feature dense vector from the notebook."""
    from nltk.corpus import stopwords
    STOP = set(stopwords.words("english"))
    SAFE = 0.0001

    feats = []
    # Basic
    feats.append(len(q1))
    feats.append(len(q2))
    feats.append(len(q1.split()))
    feats.append(len(q2.split()))
    w1 = set(q1.split()); w2 = set(q2.split())
    common = len(w1 & w2)
    total  = len(w1) + len(w2)
    feats.append(common)
    feats.append(total)
    feats.append(round(common / total, 2) if total else 0)

    # Token features (8)
    q1_tok = q1.split(); q2_tok = q2.split()
    q1_w = set(w for w in q1_tok if w not in STOP)
    q2_w = set(w for w in q2_tok if w not in STOP)
    q1_s = set(w for w in q1_tok if w in STOP)
    q2_s = set(w for w in q2_tok if w in STOP)
    common_wc = len(q1_w & q2_w)
    common_sc = len(q1_s & q2_s)
    feats.append(common_wc / (min(len(q1_w), len(q2_w)) + SAFE))
    feats.append(common_wc / (max(len(q1_w), len(q2_w)) + SAFE))
    feats.append(common_sc / (min(len(q1_s), len(q2_s)) + SAFE))
    feats.append(common_sc / (max(len(q1_s), len(q2_s)) + SAFE))
    common_tc = len(set(q1_tok) & set(q2_tok))
    feats.append(common_tc / (min(len(q1_tok), len(q2_tok)) + SAFE))
    feats.append(common_tc / (max(len(q1_tok), len(q2_tok)) + SAFE))
    feats.append(int(q1_tok[-1] == q2_tok[-1]) if q1_tok and q2_tok else 0)
    feats.append(int(q1_tok[0]  == q2_tok[0])  if q1_tok and q2_tok else 0)

    # Length features (3)
    feats.append(abs(len(q1_tok) - len(q2_tok)))
    feats.append((len(q1_tok) + len(q2_tok)) / 2)
    try:
        import distance
        strs = list(distance.lcsubstrings(q1, q2))
        feats.append(len(strs[0]) / (min(len(q1), len(q2)) + 1) if strs else 0.0)
    except Exception:
        feats.append(0.0)

    # Fuzzy features (4)
    try:
        from fuzzywuzzy import fuzz
        feats.append(fuzz.QRatio(q1, q2))
        feats.append(fuzz.partial_ratio(q1, q2))
        feats.append(fuzz.token_sort_ratio(q1, q2))
        feats.append(fuzz.token_set_ratio(q1, q2))
    except Exception:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(feats)


def build_bow_vector(q1, q2):
    """Build the full feature vector for BOW models."""
    from scipy.sparse import hstack, csr_matrix
    dense = csr_matrix(get_handcrafted_features(q1, q2).reshape(1, -1))
    q1_bow = cv.transform([q1])
    q2_bow = cv.transform([q2])
    return hstack([dense, q1_bow, q2_bow], format='csr')


def sentence_vector_w2v(sentence, vector_size=300):
    """Mean pooling over Word2Vec word vectors."""
    words = re.sub(r'[^a-z ]', '', sentence.lower()).split()
    vecs = [w2v_model[w] for w in words if w in w2v_model]
    return np.mean(vecs, axis=0) if vecs else np.zeros(vector_size)


def build_w2v_vector(q1, q2):
    """Build full feature vector for Word2Vec models (22 + 300 + 300 = 622 dims)."""
    dense = get_handcrafted_features(q1, q2)
    v1 = sentence_vector_w2v(q1)
    v2 = sentence_vector_w2v(q2)
    return np.concatenate([dense, v1, v2]).reshape(1, -1).astype(np.float32)


# ─── Model Loading ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_models():
    global cv, rf_bow, xgb_bow, rf_wtv, xgb_wtv, w2v_model
    global sbert_model, sbert_thresh, cross_model, cross_thresh
    global faiss_index, faq_meta

    import nltk
    try: nltk.data.find('corpora/stopwords')
    except: nltk.download('stopwords', quiet=True)

    # 1. CountVectorizer
    try:
        cv = joblib.load("count_vectorizer1.joblib")
        MODEL_STATUS["CountVectorizer"] = "✅ Loaded"
    except Exception as e:
        MODEL_STATUS["CountVectorizer"] = f"❌ {e}"

    # 2. RF + BOW
    try:
        rf_bow = joblib.load("randomForest_BOW.joblib")
        MODEL_STATUS["RF_BOW"] = "✅ Loaded"
    except Exception as e:
        MODEL_STATUS["RF_BOW"] = f"❌ {e}"

    # 3. XGB + BOW
    try:
        xgb_bow = joblib.load("XGBClassifier_BOW.joblib")
        MODEL_STATUS["XGB_BOW"] = "✅ Loaded"
    except Exception as e:
        MODEL_STATUS["XGB_BOW"] = f"❌ {e}"

    # 4. RF + Word2Vec
    try:
        rf_wtv = joblib.load("RandomForestClassifier_WTV.joblib")
        MODEL_STATUS["RF_WTV"] = "✅ Loaded"
    except Exception as e:
        MODEL_STATUS["RF_WTV"] = f"❌ {e}"

    # 5. XGB + Word2Vec
    try:
        xgb_wtv = joblib.load("XGBClassifier_WTV.joblib")
        MODEL_STATUS["XGB_WTV"] = "✅ Loaded"
    except Exception as e:
        MODEL_STATUS["XGB_WTV"] = f"❌ {e}"

    # 6. Word2Vec embeddings (trimmed local model — no internet needed)
    if rf_wtv is not None or xgb_wtv is not None:
        try:
            from gensim.models import KeyedVectors
            w2v_path = "word2vec_trimmed.kv"
            if os.path.exists(w2v_path):
                print("Loading trimmed Word2Vec from local file...")
                w2v_model = KeyedVectors.load(w2v_path)
            else:
                import gensim.downloader as gapi
                print("WARNING: Trimmed Word2Vec not found, downloading full model...")
                w2v_model = gapi.load("word2vec-google-news-300")
            MODEL_STATUS["Word2Vec"] = f"✅ Loaded ({len(w2v_model):,} words)"
        except Exception as e:
            MODEL_STATUS["Word2Vec"] = f"❌ {e}"

    # 7. SBERT
    try:
        from sentence_transformers import SentenceTransformer
        sbert_path = "./sbert_quora_model"
        sbert_model = SentenceTransformer(sbert_path if os.path.exists(sbert_path) else 'all-MiniLM-L6-v2')
        MODEL_STATUS["SBERT"] = "✅ Loaded"
    except Exception as e:
        MODEL_STATUS["SBERT"] = f"❌ {e}"

    try:
        s = joblib.load("sbert_results.joblib")
        sbert_thresh = float(s['best_threshold'])
    except Exception:
        sbert_thresh = 0.5

    # 8. Cross-Encoder
    try:
        from sentence_transformers import CrossEncoder
        ce_path = "./cross_encoder_quora"
        cross_model = CrossEncoder(ce_path if os.path.exists(ce_path) else 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        MODEL_STATUS["CrossEncoder"] = "✅ Loaded"
    except Exception as e:
        MODEL_STATUS["CrossEncoder"] = f"❌ {e}"

    try:
        c = joblib.load("cross_encoder_results.joblib")
        cross_thresh = float(c['best_threshold'])
    except Exception:
        cross_thresh = 0.5

    # 9. FAISS + FAQ metadata
    try:
        import faiss
        faiss_index = faiss.read_index("faq_faiss.index")
        faq_meta    = joblib.load("faq_metadata.joblib")
        MODEL_STATUS["FAISS_FAQ"] = f"✅ {faiss_index.ntotal} FAQs indexed"
    except Exception as e:
        MODEL_STATUS["FAISS_FAQ"] = f"❌ {e}"

    print("=== Model Status ===")
    for k, v in MODEL_STATUS.items():
        print(f"  {k}: {v}")


# ─── Request / Response Schemas ───────────────────────────────────────────────

class PairRequest(BaseModel):
    question1: str
    question2: str

class FAQRequest(BaseModel):
    query: str
    top_k: int = 5

class ModelResult(BaseModel):
    model: str
    prediction: str       # "Duplicate" | "Not Duplicate" | "N/A"
    confidence: Optional[float]
    available: bool

class SimilarityResponse(BaseModel):
    question1: str
    question2: str
    results: List[ModelResult]
    majority_vote: str

class FAQResult(BaseModel):
    question: str
    answer: str
    confidence: float
    rank: int

class FAQResponse(BaseModel):
    query: str
    is_confident: bool
    results: List[FAQResult]


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "SemSim API running", "status": MODEL_STATUS}

@app.get("/health")
def health():
    return {"status": "ok", "models": MODEL_STATUS}

@app.post("/similarity", response_model=SimilarityResponse)
def predict_similarity(req: PairRequest):
    q1 = preprocess(req.question1.strip())
    q2 = preprocess(req.question2.strip())
    if not q1 or not q2:
        raise HTTPException(400, "Both questions must be non-empty")

    results = []

    # ── Model 1: RF + BOW ──
    if rf_bow and cv:
        try:
            X = build_bow_vector(q1, q2)
            pred = int(rf_bow.predict(X)[0])
            prob = rf_bow.predict_proba(X)[0][pred]
            results.append(ModelResult(model="Random Forest + BOW", prediction="Duplicate" if pred else "Not Duplicate",
                                       confidence=round(float(prob), 4), available=True))
        except Exception as e:
            results.append(ModelResult(model="Random Forest + BOW", prediction="N/A", confidence=None, available=False))
    else:
        results.append(ModelResult(model="Random Forest + BOW", prediction="N/A", confidence=None, available=False))

    # ── Model 2: XGB + BOW ──
    if xgb_bow and cv:
        try:
            X = build_bow_vector(q1, q2)
            pred = int(xgb_bow.predict(X)[0])
            prob = xgb_bow.predict_proba(X)[0][pred]
            results.append(ModelResult(model="XGBoost + BOW", prediction="Duplicate" if pred else "Not Duplicate",
                                       confidence=round(float(prob), 4), available=True))
        except Exception as e:
            results.append(ModelResult(model="XGBoost + BOW", prediction="N/A", confidence=None, available=False))
    else:
        results.append(ModelResult(model="XGBoost + BOW", prediction="N/A", confidence=None, available=False))

    # ── Model 3: RF + Word2Vec ──
    if rf_wtv and w2v_model:
        try:
            X = build_w2v_vector(q1, q2)
            pred = int(rf_wtv.predict(X)[0])
            prob = rf_wtv.predict_proba(X)[0][pred]
            results.append(ModelResult(model="Random Forest + Word2Vec", prediction="Duplicate" if pred else "Not Duplicate",
                                       confidence=round(float(prob), 4), available=True))
        except Exception as e:
            results.append(ModelResult(model="Random Forest + Word2Vec", prediction="N/A", confidence=None, available=False))
    else:
        results.append(ModelResult(model="Random Forest + Word2Vec", prediction="N/A", confidence=None,
                                   available=False))

    # ── Model 4: XGB + Word2Vec ──
    if xgb_wtv and w2v_model:
        try:
            X = build_w2v_vector(q1, q2)
            pred = int(xgb_wtv.predict(X)[0])
            prob = xgb_wtv.predict_proba(X)[0][pred]
            results.append(ModelResult(model="XGBoost + Word2Vec", prediction="Duplicate" if pred else "Not Duplicate",
                                       confidence=round(float(prob), 4), available=True))
        except Exception as e:
            results.append(ModelResult(model="XGBoost + Word2Vec", prediction="N/A", confidence=None, available=False))
    else:
        results.append(ModelResult(model="XGBoost + Word2Vec", prediction="N/A", confidence=None, available=False))

    # ── Model 5: SBERT ──
    if sbert_model:
        try:
            import torch
            e1 = sbert_model.encode(req.question1, convert_to_tensor=True)
            e2 = sbert_model.encode(req.question2, convert_to_tensor=True)
            score = float(torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item())
            score_norm = (score + 1) / 2
            pred = "Duplicate" if score_norm >= sbert_thresh else "Not Duplicate"
            results.append(ModelResult(model="SBERT (Sentence-BERT)", prediction=pred,
                                       confidence=round(score_norm, 4), available=True))
        except Exception as e:
            results.append(ModelResult(model="SBERT (Sentence-BERT)", prediction="N/A", confidence=None, available=False))
    else:
        results.append(ModelResult(model="SBERT (Sentence-BERT)", prediction="N/A", confidence=None, available=False))

    # ── Model 6: Cross-Encoder ──
    if cross_model:
        try:
            from scipy.special import expit
            score = float(cross_model.predict([(req.question1, req.question2)])[0])
            prob = float(expit(score))
            pred = "Duplicate" if prob >= cross_thresh else "Not Duplicate"
            results.append(ModelResult(model="Cross-Encoder", prediction=pred,
                                       confidence=round(prob, 4), available=True))
        except Exception as e:
            results.append(ModelResult(model="Cross-Encoder", prediction="N/A", confidence=None, available=False))
    else:
        results.append(ModelResult(model="Cross-Encoder", prediction="N/A", confidence=None, available=False))

    # Final verdict — based on SBERT + Cross-Encoder (semantic models) only
    semantic_models = [r for r in results if r.model in ("SBERT (Sentence-BERT)", "Cross-Encoder") and r.available]
    if semantic_models:
        sem_dup = sum(1 for r in semantic_models if r.prediction == "Duplicate")
        majority = "Duplicate" if sem_dup > len(semantic_models) / 2 else "Not Duplicate"
    else:
        # Fallback to all-model majority vote if neither semantic model is available
        dup_votes = sum(1 for r in results if r.prediction == "Duplicate")
        total_avail = sum(1 for r in results if r.available)
        majority = "Duplicate" if dup_votes > total_avail / 2 else "Not Duplicate"

    return SimilarityResponse(question1=req.question1, question2=req.question2,
                               results=results, majority_vote=majority)


@app.post("/faq/search", response_model=FAQResponse)
def faq_search(req: FAQRequest):
    if not faiss_index or not faq_meta or not sbert_model:
        raise HTTPException(503, "FAQ search pipeline not ready (FAISS/SBERT not loaded)")

    import faiss as _faiss
    from scipy.special import expit

    query_emb = sbert_model.encode(req.query, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)
    top_k_faiss = min(50, faiss_index.ntotal)
    scores, indices = faiss_index.search(query_emb.astype(np.float32), top_k_faiss)

    faq_qs = faq_meta["questions"]
    faq_as = faq_meta["answers"]
    retrieved_q = [faq_qs[i] for i in indices[0]]
    retrieved_a = [faq_as[i] for i in indices[0]]

    if cross_model:
        pairs = [(req.query, q) for q in retrieved_q]
        ce_scores = cross_model.predict(pairs)
        ce_probs  = expit(ce_scores)
        order = np.argsort(-ce_probs)
        confidence = ce_probs
    else:
        order = np.arange(len(retrieved_q))
        confidence = scores[0] / (scores[0].max() + 1e-6)

    top_results = []
    for rank, idx in enumerate(order[:req.top_k]):
        top_results.append(FAQResult(
            question=retrieved_q[idx], answer=retrieved_a[idx],
            confidence=round(float(confidence[idx]), 4), rank=rank + 1
        ))

    is_confident = top_results[0].confidence >= 0.5 if top_results else False
    return FAQResponse(query=req.query, is_confident=is_confident, results=top_results)


@app.get("/analysis/metrics")
def get_metrics():
    """Pre-computed evaluation metrics from model_comparison_results.csv or hardcoded from notebook."""
    import os
    if os.path.exists("model_comparison_results.csv"):
        import pandas as pd
        df = pd.read_csv("model_comparison_results.csv")
        return {"source": "csv", "data": df.to_dict(orient="records")}

    # Hardcoded from notebook output (fallback)
    return {
        "source": "updated",
        "data": [
            {"Model": "Random Forest + BOW",       "Accuracy": 0.8213, "F1 Score": 0.8336, "Precision": 0.7787, "Recall": 0.8969, "ROC-AUC": 0.8215},
            {"Model": "XGBoost + BOW",             "Accuracy": 0.8076, "F1 Score": 0.8216, "Precision": 0.7650, "Recall": 0.8872, "ROC-AUC": 0.8078},
            {"Model": "Random Forest + Word2Vec",  "Accuracy": 0.8084, "F1 Score": 0.8233, "Precision": 0.7629, "Recall": 0.8942, "ROC-AUC": 0.8086},
            {"Model": "XGBoost + Word2Vec",        "Accuracy": 0.8147, "F1 Score": 0.8242, "Precision": 0.7830, "Recall": 0.8700, "ROC-AUC": 0.8148},
            {"Model": "SBERT",                     "Accuracy": 0.8713, "F1 Score": 0.8327, "Precision": 0.8001, "Recall": 0.8681, "ROC-AUC": 0.8706},
            {"Model": "Cross-Encoder",             "Accuracy": 0.8958, "F1 Score": 0.8632, "Precision": 0.8372, "Recall": 0.8909, "ROC-AUC": 0.8948},
        ]
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)