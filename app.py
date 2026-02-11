import os
import re
import json
import hashlib
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

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
    combined = (
        df["대표과제명"].fillna("")
        + " "
        + df["연구목표요약"].fillna("")
        + " "
        + df["연구개발내용"].fillna("")
        + " "
        + df["기대효과요약"].fillna("")
    ).astype(str).tolist()

    word_vec = TfidfVectorizer(
        tokenizer=tokenize_simple,
        lowercase=False,
        max_features=80000,
    )
    word_X = word_vec.fit_transform(combined)
    word_X = normalize(word_X)

    return vectorizers, matrices, word_vec, word_X


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
        "형식: [{\"idx\":123, \"score\":87, \"rationale\":\"...\", \"evidence_fields\":[\"연구개발내용\", \"대표과제명\"]}]\n"
        "조건:\n"
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
        return keywords[:3], syn_norm
    except Exception:
        return [], {}


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

if not api_key:
    st.error("GOOGLE_API_KEY가 설정되지 않았습니다. .env 또는 Streamlit Secrets에 설정하세요.")
    st.stop()

st.subheader("1) R&D 데이터 업로드")
rnd_file = st.file_uploader("R&D 과제 엑셀 파일을 업로드하세요", type=["xlsx"])

if not rnd_file:
    st.info("먼저 R&D 과제 엑셀을 업로드해주세요.")
    st.stop()

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

vectorizers, matrices, word_vec, word_X = build_index(file_hash, df)

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
        scores, per_field_scores = compute_weighted_scores(
            query_norm, vectorizers, matrices, word_vec, word_X, weights
        )
        if scores.size == 0:
            continue

        # keyword bonus (hard signal)
        bonuses = keyword_bonus_scores(df, keywords, synonyms, per_keyword_bonus=keyword_bonus)
        scores = scores + bonuses

        # candidate selection
        k = min(candidate_k, len(df))
        idxs = np.argpartition(-scores, k - 1)[:k]
        cand = df.iloc[idxs].copy()
        cand["_idx"] = idxs
        cand["_tfidf_score"] = scores[idxs]
        cand = cand.sort_values("_tfidf_score", ascending=False)

        # rerank
        max_m = max(top_n * 2, top_n + 10)
        if rerank_m > max_m:
            st.info(f"LLM 재랭킹 후보 수를 {max_m}개로 제한했습니다(출력 안정성).")
        m = min(rerank_m, len(cand), max_m)
        rerank_candidates = cand.head(m).copy()
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
                merged["score"] = normalize_scores_to_100(merged["_tfidf_score"].values)
                merged["rationale"] = ""
                merged["evidence_fields"] = ""
        else:
            merged = rerank_candidates.copy()
            merged["score"] = normalize_scores_to_100(merged["_tfidf_score"].values)
            merged["rationale"] = ""
            merged["evidence_fields"] = ""

        # fill missing
        merged["score"] = merged["score"].fillna(0).astype(float)
        merged["rationale"] = merged["rationale"].fillna("")
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
                    "기업명": row["기업명"],
                    "R&D과제번호": row["R&D과제번호"],
                    "대표과제명": row["대표과제명"],
                    "점수(0-100)": int(round(row["score"])),
                    "근거요약": row["rationale"],
                    "근거필드": row["evidence_fields"],
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
