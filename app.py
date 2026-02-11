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
    # Fast path: try full text
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
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
    for col, weight in weights.items():
        vec = vectorizers[col]
        X = matrices[col]
        q = vec.transform([query])
        q = normalize(q)
        col_scores = X @ q.T
        col_scores = np.asarray(col_scores.todense()).ravel()
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
    return scores * 0.7 + word_scores * 0.3


def make_prompt(query: str, candidates: pd.DataFrame, strict: bool = False) -> str:
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
        "출력은 반드시 JSON 배열만 반환한다.\n"
        "형식: [{\"idx\":123, \"score\":87, \"rationale\":\"...\", \"evidence_fields\":[\"연구개발내용\", \"대표과제명\"]}]\n"
        "조건:\n"
        "- score는 0~100 정수\n"
        "- rationale은 한국어 1~2문장\n"
        "- evidence_fields는 다음 중에서만 선택: 대표과제명, 연구목표요약, 연구개발내용, 기대효과요약\n"
    )
    if strict:
        prompt += (
            "\n중요: JSON 외에 다른 텍스트를 출력하지 마라. "
            "코드블록(```)이나 설명을 추가하지 마라.\n"
        )
    return prompt


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


def rerank_with_llm(query: str, candidates: pd.DataFrame, api_key: str):
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
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                ),
            )
        except TypeError:
            # Older SDKs may not support response_mime_type
            return model.generate_content(
                prompt_text,
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=2048,
                ),
            )

    prompt = make_prompt(query, candidates, strict=False)
    try:
        response = _call(prompt)
    except Exception as e:
        return None, f"LLM 호출 실패: {e}", None

    raw = _response_text(response)
    data = extract_json_array(raw)
    if data:
        return data, "", None

    # Retry with stricter prompt
    try:
        response2 = _call(make_prompt(query, candidates, strict=True))
    except Exception as e:
        return None, f"LLM 호출 실패: {e}", None

    raw2 = _response_text(response2)
    data2 = extract_json_array(raw2)
    if not data2:
        return None, "LLM 응답에서 JSON 배열을 파싱하지 못했습니다.", raw2
    return data2, "", None


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
        scores = compute_weighted_scores(query_norm, vectorizers, matrices, word_vec, word_X, weights)
        if scores.size == 0:
            continue

        # candidate selection
        k = min(candidate_k, len(df))
        idxs = np.argpartition(-scores, k - 1)[:k]
        cand = df.iloc[idxs].copy()
        cand["_idx"] = idxs
        cand["_tfidf_score"] = scores[idxs]
        cand = cand.sort_values("_tfidf_score", ascending=False)

        # rerank
        m = min(rerank_m, len(cand))
        rerank_candidates = cand.head(m).copy()
        llm_data, llm_err, llm_raw = rerank_with_llm(query_norm, rerank_candidates, api_key)
        if llm_err:
            st.warning(llm_err)
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
        merged["evidence_fields"] = merged["evidence_fields"].apply(
            lambda v: ", ".join(v) if isinstance(v, list) else safe_text(v)
        )

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
