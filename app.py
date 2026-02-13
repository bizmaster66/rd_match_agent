import os
import re
import json
import hashlib
import tempfile
import zipfile
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi
import faiss

try:
    import google.generativeai as genai
except Exception:
    genai = None

# -----------------------------
# Config
# -----------------------------
DEFAULT_WEIGHTS = {
    "대표과제명": 1.0,
    "연구목표요약": 1.0,
    "연구개발내용": 3.0,
    "기대효과요약": 1.0,
}

REQUIRED_COLUMNS = [
    "기업명",
    "R&D과제번호",
    "대표과제명",
    "연구목표요약",
    "연구개발내용",
    "기대효과요약",
]

# -----------------------------
# Helpers
# -----------------------------

def safe_text(val: object) -> str:
    if isinstance(val, (list, tuple, dict)):
        return ""
    if pd.isna(val):
        return ""
    return str(val).strip()


def truncate(text: str, max_chars: int) -> str:
    text = safe_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def extract_json_array(text: str):
    if not text:
        return None
    text = text.strip()
    # Strip code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    # Fast path: try full text
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass

    # Try to find a JSON array substring with bracket matching
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                snippet = text[start : i + 1]
                try:
                    return json.loads(snippet)
                except Exception:
                    break

    # Try to extract one or more JSON objects
    objects = []
    start = 0
    while True:
        s = text.find("{", start)
        if s == -1:
            break
        depth = 0
        for i in range(s, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[s : i + 1]
                    try:
                        obj = json.loads(snippet)
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except Exception:
                        pass
                    start = i + 1
                    break
        else:
            break
    if objects:
        return objects

    # Regex fallback (last resort)
    match = re.search(r"\[.*\]", text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def tokenize_simple(text: str):
    text = safe_text(text)
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9]+|[가-힣]+", text)


def expand_variants(term: str):
    t = safe_text(term)
    if not t:
        return []
    variants = {t}
    pairs = [
        ("컴퓨팅", "컴퓨터"),
        ("분석", "분석기"),
        ("진단", "진단기"),
        ("최적화", "최적"),
        ("예측", "예측기"),
        ("추천", "추천기"),
        ("인공지능", "AI"),
        ("빅데이터", "빅 데이터"),
    ]
    for a, b in pairs:
        if a in t:
            variants.add(t.replace(a, b))
        if b in t:
            variants.add(t.replace(b, a))
    return list(variants)


def preprocess_query(query: str) -> str:
    q = safe_text(query)
    if not q:
        return q
    # Remove quotes/brackets and normalize separators
    q = re.sub(r"[\\[\\]\"']", " ", q)
    q = re.sub(r"[,:/|]+", " ", q)
    q = re.sub(r"\\s+", " ", q).strip()

    # If user provided comma-separated keywords, append a compact token string
    tokens = tokenize_simple(q)
    if tokens:
        q = f"{q} " + " ".join(tokens)
    return q


@st.cache_data(show_spinner=False)
def load_rnd_data(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(BytesIO(file_bytes), sheet_name=0)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(missing)}")

    # Keep only columns we need, but preserve originals if present
    df = df.copy()
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].apply(safe_text)

    return df


def combined_text_series(df: pd.DataFrame) -> pd.Series:
    base = (
        df["대표과제명"].fillna("")
        + " "
        + df["연구목표요약"].fillna("")
        + " "
        + df["연구개발내용"].fillna("")
        + " "
        + df["기대효과요약"].fillna("")
    ).astype(str)
    if "tags_expanded" in df.columns:
        base = base + " " + df["tags_expanded"].fillna("").astype(str)
    return base


def extract_tags_simple(df: pd.DataFrame, top_k: int = 3):
    texts = combined_text_series(df).tolist()
    tokenized = [tokenize_simple(t) for t in texts]
    # Build document frequency
    df_counts = {}
    for toks in tokenized:
        for tok in set([t for t in toks if len(t) >= 2]):
            df_counts[tok] = df_counts.get(tok, 0) + 1
    n_docs = len(tokenized)

    tags = []
    tags_expanded = []
    for toks in tokenized:
        tf = {}
        for t in toks:
            if len(t) < 2:
                continue
            tf[t] = tf.get(t, 0) + 1
        # simple tf-idf scoring
        scores = []
        for t, cnt in tf.items():
            idf = np.log((n_docs + 1) / (df_counts.get(t, 1) + 1)) + 1.0
            scores.append((t, cnt * idf))
        scores.sort(key=lambda x: x[1], reverse=True)
        top = [t for t, _ in scores[:top_k]]
        tags.append(", ".join(top))

        expanded = []
        for t in top:
            expanded.extend(expand_variants(t))
        expanded = list(dict.fromkeys([t for t in expanded if t]))
        tags_expanded.append(", ".join(expanded))

    return tags, tags_expanded


def build_faiss_index(embeddings: np.ndarray):
    if embeddings is None or len(embeddings) == 0:
        return None
    emb = embeddings.astype("float32")
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def save_bundle(df: pd.DataFrame, embeddings: np.ndarray, index, file_hash: str):
    manifest = {
        "rows": len(df),
        "hash": file_hash,
        "created_at": datetime.now().isoformat(),
        "embedding_model": "gemini-embedding-001",
    }
    with tempfile.TemporaryDirectory() as tmp:
        meta_path = os.path.join(tmp, "meta.csv")
        emb_path = os.path.join(tmp, "embeddings.npy")
        idx_path = os.path.join(tmp, "faiss.index")
        manifest_path = os.path.join(tmp, "manifest.json")

        df.to_csv(meta_path, index=False)
        np.save(emb_path, embeddings.astype("float32"))
        if index is not None:
            faiss.write_index(index, idx_path)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(meta_path, "meta.csv")
            zf.write(emb_path, "embeddings.npy")
            if index is not None:
                zf.write(idx_path, "faiss.index")
            zf.write(manifest_path, "manifest.json")
        zip_buf.seek(0)
        return zip_buf


@st.cache_resource(show_spinner=False)
def load_bundle(bundle_bytes: bytes):
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "bundle.zip")
        with open(zip_path, "wb") as f:
            f.write(bundle_bytes)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)

        meta_path = os.path.join(tmp, "meta.csv")
        emb_path = os.path.join(tmp, "embeddings.npy")
        idx_path = os.path.join(tmp, "faiss.index")
        manifest_path = os.path.join(tmp, "manifest.json")

        df = pd.read_csv(meta_path)
        embeddings = np.load(emb_path)
        index = faiss.read_index(idx_path) if os.path.exists(idx_path) else None
        manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

    # restore columns to expected types
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(safe_text)
    return df, embeddings, index, manifest


@st.cache_resource(show_spinner=False)
def build_index(file_hash: str, df: pd.DataFrame):
    # Use char n-grams to better capture Korean text similarity per field
    vectorizers = {}
    matrices = {}

    for col in ["대표과제명", "연구목표요약", "연구개발내용", "기대효과요약"]:
        texts = df[col].fillna("").astype(str).tolist()
        vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=50000,
        )
        X = vec.fit_transform(texts)
        X = normalize(X)
        vectorizers[col] = vec
        matrices[col] = X

    # Combined word-level index for keyword-heavy queries
    combined = combined_text_series(df).tolist()

    word_vec = TfidfVectorizer(
        tokenizer=tokenize_simple,
        lowercase=False,
        max_features=80000,
    )
    word_X = word_vec.fit_transform(combined)
    word_X = normalize(word_X)

    # BM25 index
    tokenized = [tokenize_simple(t) for t in combined]
    bm25 = BM25Okapi(tokenized)

    return vectorizers, matrices, word_vec, word_X, bm25, tokenized


def compute_weighted_scores(query: str, vectorizers, matrices, word_vec, word_X, weights):
    scores = None
    per_field = {}
    for col, weight in weights.items():
        vec = vectorizers[col]
        X = matrices[col]
        q = vec.transform([query])
        q = normalize(q)
        col_scores = X @ q.T
        col_scores = np.asarray(col_scores.todense()).ravel()
        per_field[col] = col_scores
        if scores is None:
            scores = weight * col_scores
        else:
            scores += weight * col_scores

    if scores is None:
        scores = np.zeros(word_X.shape[0], dtype=float)

    # Add word-level signal for keyword-focused queries
    q_word = word_vec.transform([query])
    q_word = normalize(q_word)
    word_scores = word_X @ q_word.T
    word_scores = np.asarray(word_scores.todense()).ravel()

    # Blend signals
    blended = scores * 0.7 + word_scores * 0.3
    return blended, per_field


def normalize_0_100(arr: np.ndarray):
    if arr.size == 0:
        return arr
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < 1e-9:
        return np.full_like(arr, 50.0)
    return (arr - min_v) / (max_v - min_v) * 100.0


def _extract_embedding(response):
    if response is None:
        return None
    if isinstance(response, dict):
        if "embedding" in response:
            return response["embedding"]
        if "embeddings" in response:
            return response["embeddings"]
    if hasattr(response, "embedding"):
        return response.embedding
    if hasattr(response, "embeddings"):
        return response.embeddings
    return None


def embed_texts(texts: list, api_key: str):
    if genai is None:
        return None
    genai.configure(api_key=api_key)
    model = "gemini-embedding-001"
    # Try batch embedding
    try:
        resp = genai.embed_content(
            model=model,
            content=texts,
        )
        emb = _extract_embedding(resp)
        if isinstance(emb, list) and emb and isinstance(emb[0], (list, np.ndarray)):
            return np.array(emb, dtype=float)
        if isinstance(emb, list) and emb and isinstance(emb[0], dict) and "values" in emb[0]:
            return np.array([e["values"] for e in emb], dtype=float)
    except Exception:
        pass

    # Fallback: single call per text
    vectors = []
    for t in texts:
        try:
            resp = genai.embed_content(model=model, content=t)
            emb = _extract_embedding(resp)
            if isinstance(emb, dict) and "values" in emb:
                vectors.append(emb["values"])
            else:
                vectors.append(emb)
        except Exception:
            vectors.append(None)
    if any(v is None for v in vectors):
        return None
    return np.array(vectors, dtype=float)


def embed_texts_batched(texts: list, api_key: str, batch_size: int = 64, progress=None):
    vectors = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        emb = embed_texts(batch, api_key)
        if emb is None:
            return None
        vectors.append(emb)
        if progress:
            progress.progress(min((i + batch_size) / total, 1.0))
    return np.vstack(vectors) if vectors else None


def make_prompt(query: str, candidates: pd.DataFrame, top_n: int, strict: bool = False) -> str:
    items = []
    for _, row in candidates.iterrows():
        items.append(
            {
                "idx": int(row["_idx"]),
                "기업명": truncate(row["기업명"], 60),
                "R&D과제번호": truncate(row["R&D과제번호"], 40),
                "대표과제명": truncate(row["대표과제명"], 160),
                "연구목표요약": truncate(row["연구목표요약"], 300),
                "연구개발내용": truncate(row["연구개발내용"], 500),
                "기대효과요약": truncate(row["기대효과요약"], 300),
            }
        )

    prompt = (
        "너는 R&D 과제 추천 심사관이다.\n"
        f"입력 과제: \"{query}\"\n\n"
        "후보 목록(JSON):\n"
        f"{json.dumps(items, ensure_ascii=False)}\n\n"
        "각 후보에 대해 입력 과제와의 관련성을 평가해라.\n"
        f"출력은 상위 {top_n}개 결과만 포함한다. (부족하면 가능한 만큼만)\n"
        "출력은 반드시 JSON 배열만 반환한다.\n"
        "형식: [{\"idx\":123, \"is_relevant\":true, \"score\":87, \"rationale\":\"...\", \"evidence_fields\":[\"연구개발내용\", \"대표과제명\"]}]\n"
        "조건:\n"
        "- is_relevant는 입력 과제와 관련 있으면 true, 아니면 false\n"
        "- score는 0~100 정수\n"
        "- rationale은 한국어 1문장(60자 이내)\n"
        "- evidence_fields는 다음 중에서만 선택: 대표과제명, 연구목표요약, 연구개발내용, 기대효과요약\n"
        "- evidence_fields는 최대 2개만 포함\n"
    )
    if strict:
        prompt += (
            "\n중요: JSON 외에 다른 텍스트를 출력하지 마라. "
            "코드블록(```)이나 설명을 추가하지 마라.\n"
        )
    return prompt


def make_keyword_prompt(query: str) -> str:
    prompt = (
        "너는 입력 과제에서 핵심 키워드를 추출하고 동의어/유의어를 제시하는 역할이다.\n"
        f"입력 과제: \"{query}\"\n"
        "다음 JSON 형식으로만 출력하라.\n"
        "{\"keywords\": [\"...\", \"...\"], \"synonyms\": {\"키워드1\": [\"...\", \"...\"]}}\n"
        "조건:\n"
        "- keywords는 1~3개\n"
        "- 한국어 중심, 의미상 핵심 단어만\n"
        "- synonyms는 각 키워드당 0~4개, 너무 일반적인 단어는 제외\n"
        "- 같은 개념의 표현 변형(예: 컴퓨팅/컴퓨터, 분석/분석기, 진단/진단기, 최적화/최적 등)은 반드시 포함\n"
        "- 키워드의 한자/영문 약어도 있으면 포함 (예: 인공지능/AI)\n"
    )
    return prompt


def extract_keywords_with_llm(query: str, api_key: str):
    if genai is None:
        return [], {}
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    try:
        response = model.generate_content(
            make_keyword_prompt(query),
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )
    except TypeError:
        response = model.generate_content(
            make_keyword_prompt(query),
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=512,
            ),
        )
    raw = _response_text(response)
    try:
        data = json.loads(raw)
        keywords = [safe_text(k) for k in data.get("keywords", []) if safe_text(k)]
        synonyms = data.get("synonyms", {}) if isinstance(data.get("synonyms", {}), dict) else {}
        # normalize synonyms
        syn_norm = {}
        for k, vals in synonyms.items():
            if not safe_text(k):
                continue
            if isinstance(vals, list):
                syn_norm[safe_text(k)] = [safe_text(v) for v in vals if safe_text(v)]
        # Expand variants for each keyword and synonym
        expanded = {}
        for k in keywords:
            base = expand_variants(k)
            syns = syn_norm.get(k, [])
            all_terms = []
            for s in syns:
                all_terms.extend(expand_variants(s))
            all_terms.extend(base)
            expanded[k] = list(dict.fromkeys([t for t in all_terms if t]))
        return keywords[:3], expanded
    except Exception:
        return [], {}


def make_rewrite_prompt(query: str, keywords: list, synonyms: dict) -> str:
    kw = ", ".join(keywords) if keywords else ""
    prompt = (
        "너는 검색 쿼리를 재작성하는 역할이다.\n"
        f"입력 과제: \"{query}\"\n"
        f"핵심 키워드: {kw}\n"
        "키워드의 동의어/유의어를 고려해 검색용 쿼리를 확장해라.\n"
        "다음 JSON 형식으로만 출력하라.\n"
        "{\"rewrite\": \"...\"}\n"
        "조건:\n"
        "- 한국어 중심\n"
        "- 1문장 또는 키워드 나열 형태\n"
        "- 원문 의미를 왜곡하지 말 것\n"
    )
    return prompt


def rewrite_query_with_llm(query: str, keywords: list, synonyms: dict, api_key: str):
    if genai is None:
        return query
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    try:
        response = model.generate_content(
            make_rewrite_prompt(query, keywords, synonyms),
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=256,
                response_mime_type="application/json",
            ),
        )
    except TypeError:
        response = model.generate_content(
            make_rewrite_prompt(query, keywords, synonyms),
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=256,
            ),
        )
    raw = _response_text(response)
    try:
        data = json.loads(raw)
        rewrite = safe_text(data.get("rewrite", ""))
        return rewrite if rewrite else query
    except Exception:
        return query


def keyword_bonus_scores(df: pd.DataFrame, keywords: list, synonyms: dict, per_keyword_bonus: float):
    if not keywords:
        return np.zeros(len(df), dtype=float)

    # Build search terms
    terms = []
    for k in keywords:
        terms.append(k)
        for s in synonyms.get(k, []):
            terms.append(s)

    terms = [t for t in terms if t]
    if not terms:
        return np.zeros(len(df), dtype=float)

    combined = (
        df["대표과제명"].fillna("")
        + " "
        + df["연구목표요약"].fillna("")
        + " "
        + df["연구개발내용"].fillna("")
        + " "
        + df["기대효과요약"].fillna("")
    ).astype(str)

    bonuses = np.zeros(len(df), dtype=float)
    for k in keywords:
        candidates = [k] + synonyms.get(k, [])
        pattern = "|".join([re.escape(t) for t in candidates if t])
        if not pattern:
            continue
        mask = combined.str.contains(pattern, case=False, regex=True)
        bonuses[mask.values] += per_keyword_bonus

    return bonuses


def build_keyword_groups(keywords: list, synonyms: dict):
    groups = []
    for k in keywords:
        terms = [k] + synonyms.get(k, [])
        terms = [t for t in terms if t]
        if terms:
            groups.append((k, terms))
    return groups


def compute_keyword_matches(df: pd.DataFrame, groups: list):
    if not groups:
        return np.zeros(len(df), dtype=int), [[] for _ in range(len(df))]

    combined = (
        df["대표과제명"].fillna("")
        + " "
        + df["연구목표요약"].fillna("")
        + " "
        + df["연구개발내용"].fillna("")
        + " "
        + df["기대효과요약"].fillna("")
    ).astype(str)

    match_count = np.zeros(len(df), dtype=int)
    matched = [[] for _ in range(len(df))]
    for keyword, terms in groups:
        pattern = "|".join([re.escape(t) for t in terms if t])
        if not pattern:
            continue
        mask = combined.str.contains(pattern, case=False, regex=True)
        idxs = np.where(mask.values)[0]
        match_count[idxs] += 1
        for i in idxs:
            matched[i].append(keyword)

    return match_count, matched


def find_snippet(row: pd.Series, terms: list, window: int = 30) -> str:
    fields = ["연구개발내용", "대표과제명", "연구목표요약", "기대효과요약"]
    for col in fields:
        text = safe_text(row.get(col, ""))
        if not text:
            continue
        for term in terms:
            if not term:
                continue
            m = re.search(re.escape(term), text, flags=re.IGNORECASE)
            if m:
                start = max(m.start() - window, 0)
                end = min(m.end() + window, len(text))
                snippet = text[start:end]
                return f"{col}: {snippet}"
    return ""
def _response_text(response) -> str:
    if response is None:
        return ""
    text = getattr(response, "text", None)
    if text:
        return text
    # Fallback for SDKs that don't populate .text
    try:
        parts = response.candidates[0].content.parts
        return "".join([getattr(p, "text", "") for p in parts])
    except Exception:
        return ""


def rerank_with_llm(query: str, candidates: pd.DataFrame, api_key: str, top_n: int, batch_size: int):
    if genai is None:
        return None, "google-generativeai 패키지를 불러오지 못했습니다.", None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    def _call(prompt_text):
        try:
            return model.generate_content(
                prompt_text,
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                ),
            )
        except TypeError:
            # Older SDKs may not support response_mime_type
            return model.generate_content(
                prompt_text,
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=4096,
                ),
            )

    all_items = []
    raw_debug = []
    for start in range(0, len(candidates), batch_size):
        chunk = candidates.iloc[start : start + batch_size]
        chunk_top = min(top_n, len(chunk), batch_size)
        prompt = make_prompt(query, chunk, top_n=chunk_top, strict=False)
        try:
            response = _call(prompt)
        except Exception as e:
            return None, f"LLM 호출 실패: {e}", None

        raw = _response_text(response)
        data = extract_json_array(raw)
        if not data:
            # Retry with stricter prompt
            try:
                response2 = _call(make_prompt(query, chunk, top_n=chunk_top, strict=True))
            except Exception as e:
                return None, f"LLM 호출 실패: {e}", None
            raw2 = _response_text(response2)
            data = extract_json_array(raw2)
            if not data:
                raw_debug.append(raw2)
                continue

        all_items.extend(data)
        raw_debug.append(raw)

    if not all_items:
        return None, "LLM 응답에서 JSON 배열을 파싱하지 못했습니다.", "\n\n".join(raw_debug)
    return all_items, "", "\n\n".join(raw_debug)


def normalize_scores_to_100(scores: np.ndarray):
    if scores.size == 0:
        return scores
    min_v = float(scores.min())
    max_v = float(scores.max())
    if max_v - min_v < 1e-9:
        return np.full_like(scores, 50.0)
    return (scores - min_v) / (max_v - min_v) * 100.0


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="R&D 과제 매칭 에이전트", layout="wide")

st.title("R&D 과제 매칭 에이전트")
st.caption("입력 과제(문단/키워드)와 관련성이 높은 기업 및 R&D 과제를 찾아 점수화하고 근거를 요약합니다.")

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
is_cloud = os.getcwd().startswith("/mount/src")

if not api_key:
    st.error("GOOGLE_API_KEY가 설정되지 않았습니다. .env 또는 Streamlit Secrets에 설정하세요.")
    st.stop()

st.subheader("1) 데이터 소스")
tab_excel_src, tab_bundle_src = st.tabs(["엑셀 업로드", "번들 업로드"])

df = None
file_hash = None
bundle_index = None
bundle_embeddings = None
bundle_manifest = None

with tab_excel_src:
    rnd_file = st.file_uploader("R&D 과제 엑셀 파일을 업로드하세요", type=["xlsx"])
    if rnd_file:
        file_bytes = rnd_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        try:
            df = load_rnd_data(file_bytes)
        except Exception as e:
            st.error(str(e))
            st.stop()

        st.success(f"데이터 로드 완료: {len(df):,} rows")

        with st.expander("데이터 미리보기", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)

        with st.expander("인덱스/번들 생성 (로컬용)", expanded=False):
            st.caption("임베딩과 벡터 인덱스를 생성해 번들(zip)로 내려받습니다.")
            if is_cloud:
                st.warning("Streamlit Cloud에서는 메모리 한도 때문에 번들 생성을 비활성화했습니다. 로컬에서 생성 후 번들을 업로드하세요.")
                batch_size = st.slider("임베딩 배치 크기", min_value=16, max_value=128, value=64, step=16, disabled=True)
                build_bundle_btn = st.button("번들 생성", key="build_bundle_btn", disabled=True)
            else:
                batch_size = st.slider("임베딩 배치 크기", min_value=16, max_value=128, value=64, step=16)
                build_bundle_btn = st.button("번들 생성", key="build_bundle_btn")
            if build_bundle_btn:
                progress = st.progress(0.0)
                tags, tags_expanded = extract_tags_simple(df, top_k=3)
                df_bundle = df.copy()
                df_bundle["tags"] = tags
                df_bundle["tags_expanded"] = tags_expanded

                texts = combined_text_series(df_bundle).tolist()
                embeddings = embed_texts_batched(texts, api_key, batch_size=batch_size, progress=progress)
                if embeddings is None:
                    st.error("임베딩 생성에 실패했습니다.")
                else:
                    index = build_faiss_index(embeddings)
                    bundle_buf = save_bundle(df_bundle, embeddings, index, file_hash)
                    today = datetime.now().strftime("%Y%m%d")
                    bundle_name = f"rnd_index_bundle_{today}.zip"
                    st.download_button(
                        label=f"번들 다운로드 ({bundle_name})",
                        data=bundle_buf,
                        file_name=bundle_name,
                        mime="application/zip",
                    )
                progress.empty()

with tab_bundle_src:
    bundle_file = st.file_uploader("번들(zip) 파일을 업로드하세요", type=["zip"])
    if bundle_file:
        bundle_bytes = bundle_file.getvalue()
        df, bundle_embeddings, bundle_index, bundle_manifest = load_bundle(bundle_bytes)
        file_hash = bundle_manifest.get("hash") if bundle_manifest else hashlib.md5(bundle_bytes).hexdigest()
        st.success(f"번들 로드 완료: {len(df):,} rows")
        with st.expander("번들 정보", expanded=False):
            st.json(bundle_manifest or {})

if df is None:
    st.info("엑셀 또는 번들 파일을 업로드해주세요.")
    st.stop()

vectorizers, matrices, word_vec, word_X, bm25, tokenized = build_index(file_hash, df)

st.subheader("2) 입력 과제")
tab_text, tab_excel = st.tabs(["텍스트 입력", "엑셀 업로드"])

inputs = []

with tab_text:
    text_input = st.text_area(
        "과제를 줄 단위로 입력하세요 (최대 5개)",
        height=160,
        placeholder="예:\nAI 기반 폐배터리 진단\n디지털 헬스케어 데이터 분석 플랫폼",
    )
    if text_input:
        inputs.extend([line.strip() for line in text_input.splitlines() if line.strip()])

with tab_excel:
    input_file = st.file_uploader("입력 과제 엑셀을 업로드하세요", type=["xlsx"], key="input_file")
    if input_file:
        input_df = pd.read_excel(input_file)
        if input_df.shape[1] == 0:
            st.error("입력 엑셀에 컬럼이 없습니다.")
        else:
            col = st.selectbox("과제 텍스트 컬럼을 선택하세요", list(input_df.columns))
            inputs.extend([
                safe_text(v) for v in input_df[col].tolist() if safe_text(v)
            ])

# dedup + limit
inputs = [x for x in inputs if x]
if len(inputs) > 5:
    st.warning("입력은 최대 5개까지 지원합니다. 상위 5개만 사용합니다.")
    inputs = inputs[:5]

st.subheader("3) 결과 설정")
col1, col2 = st.columns(2)
with col1:
    top_n = st.slider("상위 결과 개수", min_value=5, max_value=50, value=20, step=1)
with col2:
    filename_base = st.text_input("결과 파일명(확장자 제외)", value="match_results")

with st.expander("고급 설정", expanded=False):
    candidate_k = st.slider("1차 후보 수(K)", min_value=50, max_value=500, value=200, step=10)
    rerank_m = st.slider("LLM 재랭킹 개수", min_value=20, max_value=100, value=40, step=5)
    rerank_batch = st.slider("LLM 배치 크기", min_value=3, max_value=15, value=5, step=1)
    keyword_bonus = st.slider("키워드 포함 가산점", min_value=0, max_value=30, value=10, step=1)
    if bundle_index is not None:
        faiss_k = st.slider("FAISS 후보 수", min_value=200, max_value=3000, value=1200, step=100)
    else:
        faiss_k = None
    st.markdown("필드 가중치")
    w_title = st.number_input("대표과제명", min_value=0.0, max_value=5.0, value=1.0, step=0.5)
    w_goal = st.number_input("연구목표요약", min_value=0.0, max_value=5.0, value=1.0, step=0.5)
    w_content = st.number_input("연구개발내용", min_value=0.0, max_value=5.0, value=3.0, step=0.5)
    w_effect = st.number_input("기대효과요약", min_value=0.0, max_value=5.0, value=1.0, step=0.5)

weights = {
    "대표과제명": w_title,
    "연구목표요약": w_goal,
    "연구개발내용": w_content,
    "기대효과요약": w_effect,
}

run = st.button("매칭 실행", type="primary", disabled=len(inputs) == 0)

if run:
    results = []

    for input_id, query in enumerate(inputs, start=1):
        st.write(f"처리 중: 입력 {input_id}")
        query_norm = preprocess_query(query)
        keywords, synonyms = extract_keywords_with_llm(query_norm, api_key)
        if keywords:
            st.caption(f"핵심 키워드: {', '.join(keywords)}")
        query_rewrite = rewrite_query_with_llm(query_norm, keywords, synonyms, api_key)
        scores, per_field_scores = compute_weighted_scores(
            query_rewrite, vectorizers, matrices, word_vec, word_X, weights
        )
        if scores.size == 0:
            continue

        # keyword bonus (hard signal)
        bonuses = keyword_bonus_scores(df, keywords, synonyms, per_keyword_bonus=keyword_bonus)
        scores = scores + bonuses

        # BM25 score
        bm25_scores = bm25.get_scores(tokenize_simple(query_rewrite))
        bm25_scores = normalize_0_100(np.asarray(bm25_scores, dtype=float))

        # keyword match filter (prefer >=2, fallback to >=1)
        groups = build_keyword_groups(keywords, synonyms)
        match_count, matched_keywords = compute_keyword_matches(df, groups)
        preferred_mask = match_count >= 2
        if preferred_mask.sum() == 0:
            preferred_mask = match_count >= 1
        if preferred_mask.sum() == 0:
            preferred_mask = np.ones(len(df), dtype=bool)
        threshold = 2 if (match_count >= 2).sum() > 0 else 1
        if (match_count >= threshold).sum() > top_n:
            st.warning("키워드 매칭 과제가 상위 결과 개수보다 많습니다. 일부만 표시됩니다.")

        # candidate selection
        candidate_pool = np.where(preferred_mask)[0]
        if bundle_index is not None and faiss_k:
            q_vec = embed_texts([query_rewrite], api_key)
            if q_vec is not None:
                q_emb = q_vec.astype("float32")
                faiss.normalize_L2(q_emb)
                _, I = bundle_index.search(q_emb, min(faiss_k, len(df)))
                faiss_ids = [i for i in I[0].tolist() if i >= 0]
                candidate_pool = np.unique(np.concatenate([candidate_pool, faiss_ids]))
        if len(candidate_pool) == 0:
            candidate_pool = np.arange(len(df))
        k = min(candidate_k, len(candidate_pool))
        # combine tfidf + bm25 + keyword bonus for pool scoring
        tfidf_norm = normalize_0_100(scores)
        pool_scores = tfidf_norm[candidate_pool] * 0.6 + bm25_scores[candidate_pool] * 0.4
        top_local = np.argpartition(-pool_scores, k - 1)[:k]
        idxs = candidate_pool[top_local]
        cand = df.iloc[idxs].copy()
        cand["_idx"] = idxs
        cand["_tfidf_score"] = scores[idxs]
        cand["_bm25_score"] = bm25_scores[idxs]
        cand["_kw_count"] = match_count[idxs]
        cand["_kw_list"] = [matched_keywords[i] for i in idxs]
        cand = cand.sort_values("_tfidf_score", ascending=False)

        # dense embedding rerank signal (on candidate set)
        max_embed = 1200
        embed_subset = cand
        if len(embed_subset) > max_embed:
            embed_subset = cand.head(max_embed).copy()
        embed_texts_list = (
            embed_subset["대표과제명"].fillna("")
            + " "
            + embed_subset["연구목표요약"].fillna("")
            + " "
            + embed_subset["연구개발내용"].fillna("")
            + " "
            + embed_subset["기대효과요약"].fillna("")
        ).astype(str).tolist()
        q_vec = embed_texts([query_rewrite], api_key)
        d_vecs = embed_texts(embed_texts_list, api_key) if q_vec is not None else None
        if q_vec is not None and d_vecs is not None:
            qn = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-9)
            dn = d_vecs / (np.linalg.norm(d_vecs, axis=1, keepdims=True) + 1e-9)
            sim = np.dot(dn, qn.T).ravel()
            embed_scores = normalize_0_100(sim)
            embed_map = dict(zip(embed_subset["_idx"].tolist(), embed_scores.tolist()))
            cand["_embed_score"] = cand["_idx"].apply(lambda i: embed_map.get(i, 0.0))
        else:
            cand["_embed_score"] = 0.0

        # rerank
        max_m = max(top_n * 2, top_n + 10)
        if rerank_m > max_m:
            st.info(f"LLM 재랭킹 후보 수를 {max_m}개로 제한했습니다(출력 안정성).")
        m = min(rerank_m, len(cand), max_m)
        # combine signals for rerank selection
        cand["_final_score"] = (
            normalize_0_100(cand["_tfidf_score"].values) * 0.4
            + normalize_0_100(cand["_bm25_score"].values) * 0.3
            + normalize_0_100(cand["_embed_score"].values) * 0.3
        )
        rerank_candidates = cand.sort_values("_final_score", ascending=False).head(m).copy()
        llm_data, llm_err, llm_raw = rerank_with_llm(
            query_norm, rerank_candidates, api_key, top_n=top_n, batch_size=rerank_batch
        )
        if llm_err:
            st.warning(llm_err)
            if llm_raw:
                with st.expander("LLM 원본 응답(디버그)", expanded=False):
                    st.code(llm_raw)
        else:
            if llm_raw:
                with st.expander("LLM 원본 응답(디버그)", expanded=False):
                    st.code(llm_raw)

        if llm_data:
            llm_df = pd.DataFrame(llm_data)
            if "idx" in llm_df.columns:
                merged = rerank_candidates.merge(llm_df, left_on="_idx", right_on="idx", how="left")
            else:
                merged = rerank_candidates.copy()
                merged["score"] = normalize_scores_to_100(merged["_final_score"].values)
                merged["rationale"] = ""
                merged["evidence_fields"] = ""
        else:
            merged = rerank_candidates.copy()
            merged["score"] = normalize_scores_to_100(merged["_final_score"].values)
            merged["rationale"] = ""
            merged["evidence_fields"] = ""

        # fill missing
        merged["score"] = merged["score"].fillna(0).astype(float)
        merged["rationale"] = merged["rationale"].fillna("")
        if "is_relevant" in merged.columns:
            merged["is_relevant"] = merged["is_relevant"].apply(
                lambda v: True if str(v).lower() in ["true", "1", "yes"] else False if str(v).lower() in ["false", "0", "no"] else bool(v)
            )
        else:
            merged["is_relevant"] = True
        # If LLM provided relevance, filter to relevant only when available
        if "is_relevant" in merged.columns and merged["is_relevant"].notna().any():
            rel = merged["is_relevant"] == True
            if rel.sum() > 0:
                merged = merged[rel]
        def fallback_fields(idx: int):
            # choose top 2 fields by per-field similarity
            items = []
            for col in ["연구개발내용", "대표과제명", "연구목표요약", "기대효과요약"]:
                score = per_field_scores.get(col, np.array([]))
                if score.size:
                    items.append((col, float(score[idx])))
            items.sort(key=lambda x: x[1], reverse=True)
            return [c for c, _ in items[:2]]

        def normalize_fields(val, idx):
            if isinstance(val, list) and val:
                return ", ".join(val)
            if safe_text(val):
                return safe_text(val)
            return ", ".join(fallback_fields(idx))

        merged["evidence_fields"] = merged.apply(
            lambda r: normalize_fields(r.get("evidence_fields", ""), int(r["_idx"])),
            axis=1,
        )

        # keyword match info + snippet
        def keyword_terms_for_row(row):
            terms = []
            for kw in row.get("_kw_list", []):
                terms.append(kw)
                for s in synonyms.get(kw, []):
                    terms.append(s)
            return [t for t in terms if t]

        merged["keyword_match_count"] = merged["_kw_count"].fillna(0).astype(int)
        merged["matched_keywords"] = merged["_kw_list"].apply(lambda v: ", ".join(v) if isinstance(v, list) else "")
        merged["keyword_match_passed"] = merged["keyword_match_count"] >= threshold
        merged["keyword_match_snippet"] = merged.apply(
            lambda r: find_snippet(r, keyword_terms_for_row(r)), axis=1
        )

        # One more pass for missing rationales (smaller batch)
        missing = merged[merged["rationale"].str.strip() == ""]
        if not missing.empty:
            st.info(f"근거요약이 비어 있는 {len(missing)}개 항목을 추가 생성합니다.")
            llm_data2, llm_err2, _ = rerank_with_llm(
                query_norm, missing, api_key, top_n=len(missing), batch_size=min(5, rerank_batch)
            )
            if llm_data2:
                llm_df2 = pd.DataFrame(llm_data2)
                if "idx" in llm_df2.columns:
                    merged = merged.merge(llm_df2, left_on="_idx", right_on="idx", how="left", suffixes=("", "_new"))
                    merged["rationale"] = merged["rationale"].where(
                        merged["rationale"].str.strip() != "",
                        merged["rationale_new"].fillna(""),
                    )
                    merged["score"] = merged["score"].where(
                        merged["score"] > 0,
                        merged["score_new"].fillna(0),
                    )
                    merged["evidence_fields"] = merged["evidence_fields"].where(
                        merged["evidence_fields"].str.strip() != "",
                        merged["evidence_fields_new"].fillna(""),
                    )
                    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_new")], errors="ignore")

        merged = merged.sort_values("score", ascending=False).head(top_n)

        for _, row in merged.iterrows():
            results.append(
                {
                    "input_id": input_id,
                    "input_text": query,
                    "추출키워드": ", ".join(keywords) if keywords else "",
                    "검색쿼리": query_rewrite,
                    "태그": row.get("tags", "") if "tags" in merged.columns else "",
                    "기업명": row["기업명"],
                    "R&D과제번호": row["R&D과제번호"],
                    "대표과제명": row["대표과제명"],
                    "점수(0-100)": int(round(row["score"])),
                    "근거요약": row["rationale"],
                    "근거필드": row["evidence_fields"],
                    "키워드매칭개수": int(row.get("keyword_match_count", 0)),
                    "매칭된키워드": row.get("matched_keywords", ""),
                    "키워드매칭통과": bool(row.get("keyword_match_passed", False)),
                    "키워드매칭스니펫": row.get("keyword_match_snippet", ""),
                }
            )

    if results:
        out_df = pd.DataFrame(results)
        st.subheader("결과 미리보기")
        st.dataframe(out_df, use_container_width=True)

        today = datetime.now().strftime("%Y%m%d")
        out_name = f"{filename_base}_result_{today}.xlsx"
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            out_df.to_excel(writer, index=False, sheet_name="Results")
        output.seek(0)

        st.download_button(
            label=f"엑셀 다운로드 ({out_name})",
            data=output,
            file_name=out_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.warning("결과가 없습니다. 입력이나 데이터 상태를 확인하세요.")
