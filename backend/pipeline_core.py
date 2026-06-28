"""Streamlit-free extraction of the Research Agent pipeline.

This is a faithful port of the pipeline functions in the project's `app.py`,
with the Streamlit couplings removed:
  - `@st.cache_resource`  -> `functools.lru_cache` (module-level singletons)
  - `st.secrets`          -> environment variables only (`get_secret`)
  - `st.write/info/error/stop/progress` -> removed or replaced with raises/callbacks

The data model (`Paper`, `LLMConfig`) and the algorithms (3-stage hybrid
retrieval, LLM/heuristic classification, Moneyball scoring) match app.py so the
backend and the Streamlit app produce the same results. The eventual goal is to
have app.py import from here too, removing the duplication.
"""

from __future__ import annotations

import os
import json
import math
import time
import textwrap
import pathlib
import tempfile
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict, Any, Callable
from functools import lru_cache

import numpy as np
import requests

# Heavy/optional imports are done lazily inside functions where practical, but
# these are core to the backend so we import eagerly.
from openai import OpenAI
from groq import Groq

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:  # pragma: no cover
    SentenceTransformer = None
    CrossEncoder = None

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None

try:
    import bm25s
except ImportError:  # pragma: no cover
    bm25s = None


# =========================
# Constants
# =========================

MIN_FOR_PREDICTION = 20
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-3-large"
GEMINI_EMBEDDING_MODEL_NAME = "text-embedding-004"
LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_MONEYBALL_WEIGHTS = {
    "weight_fame": 0.84,
    "weight_hype": 0.0,
    "weight_sniper": 0.0,
    "weight_utility": 0.16,
}

CONFERENCE_KEYWORDS = [
    "EMNLP", "ACL", "NAACL", "EACL",
    "NeurIPS", "ICLR", "ICML",
    "CVPR", "ECCV",
    "ICASSP", "AAAI", "AISTATS",
]
JOURNAL_KEYWORDS = [
    "Nature", "Science",
    "JMLR", "Journal of Machine Learning Research",
    "TPAMI", "IEEE Transactions on Pattern Analysis",
    "Artificial Intelligence Journal",
    "IJCV", "International Journal of Computer Vision",
    "Nature Machine Intelligence", "Nature Communications",
]
NEGATIVE_VENUE_SIGNALS = ["submitted to", "under review", "preprint"]

ARXIV_CATEGORIES: Dict[str, List[str]] = {
    "Computer Science": [
        "cs.AI", "cs.LG", "cs.HC", "cs.CL", "cs.CV", "cs.RO", "cs.IR", "cs.NE", "cs.SE",
        "cs.CR", "cs.DS", "cs.DB", "cs.SI", "cs.MM", "cs.IT", "cs.PF", "cs.MA",
    ],
}
ARXIV_CODE_TO_NAME = {
    "cs.AI": "Artificial Intelligence", "cs.LG": "Machine Learning",
    "cs.HC": "Human-Computer Interaction", "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision and Pattern Recognition", "cs.RO": "Robotics",
    "cs.IR": "Information Retrieval", "cs.NE": "Neural and Evolutionary Computing",
    "cs.SE": "Software Engineering", "cs.CR": "Cryptography and Security",
    "cs.DS": "Data Structures and Algorithms", "cs.DB": "Databases",
    "cs.SI": "Social and Information Networks", "cs.MM": "Multimedia",
    "cs.IT": "Information Theory", "cs.PF": "Performance", "cs.MA": "Multiagent Systems",
}


# =========================
# Data structures
# =========================

@dataclass
class LLMConfig:
    api_key: str
    model: str
    api_base: Optional[str]
    provider: str = "openai"  # "openai", "gemini", "groq", or "free_local"


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: List[str]
    email_domains: List[str]
    abstract: str
    submitted_date: datetime
    pdf_url: str
    arxiv_url: str
    predicted_citations: Optional[float] = None
    prediction_explanations: Optional[List[str]] = None
    semantic_relevance: Optional[float] = None
    semantic_reason: Optional[str] = None
    focus_label: Optional[str] = None
    llm_relevance_score: Optional[float] = None
    venue: Optional[str] = None
    source: Optional[str] = None


# =========================
# Secrets / paths
# =========================

def get_secret(name: str, default: str = "") -> str:
    """Environment-variable-only secret lookup (backend has no st.secrets)."""
    value = os.getenv(name)
    if value:
        return str(value).strip().strip('\'"')
    return default


def get_corpus_dir() -> pathlib.Path:
    env_dir = os.environ.get("CORPUS_DATA_DIR")
    if env_dir:
        p = pathlib.Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    temp_path = pathlib.Path(tempfile.gettempdir()) / "researchagent_corpus"
    temp_path.mkdir(parents=True, exist_ok=True)
    return temp_path


def has_r2_credentials() -> bool:
    return all(get_secret(k) for k in
               ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET"))


# =========================
# Corpus sync from Cloudflare R2
# =========================

def download_corpus_artifacts(log: Callable[[str], None] = print) -> bool:
    """Sync corpus artifacts from R2 to the local corpus dir if remote is newer.

    Returns True if a usable corpus.db is present afterwards. Idempotent: skips
    download when local build_meta matches remote.
    """
    import boto3
    from botocore.config import Config
    from concurrent.futures import ThreadPoolExecutor, as_completed

    key_id = get_secret("R2_ACCESS_KEY_ID")
    access_key = get_secret("R2_SECRET_ACCESS_KEY")
    endpoint = get_secret("R2_ENDPOINT")
    bucket = get_secret("R2_BUCKET")
    if not all([key_id, access_key, endpoint, bucket]):
        log("R2 credentials not configured — corpus sync skipped.")
        return (get_corpus_dir() / "corpus.db").exists()

    corpus_dir = get_corpus_dir()
    meta_path = corpus_dir / "build_meta.json"

    s3 = boto3.client(
        "s3", endpoint_url=endpoint,
        aws_access_key_id=key_id, aws_secret_access_key=access_key,
        region_name="auto", config=Config(signature_version="s3v4"),
    )

    local_meta = {}
    if meta_path.exists():
        try:
            local_meta = json.load(open(meta_path, encoding="utf-8"))
        except Exception:
            pass

    remote_meta = json.loads(
        s3.get_object(Bucket=bucket, Key="corpus/build_meta.json")["Body"].read().decode("utf-8")
    )

    bm25_dir = corpus_dir / "bm25_index"
    up_to_date = (
        remote_meta.get("built_at", "") == local_meta.get("built_at", "_")
        and remote_meta.get("schema_version", 1) == local_meta.get("schema_version", 0)
        and (corpus_dir / "corpus.db").exists()
        and (corpus_dir / "index_minilm.faiss").exists()
        and bm25_dir.is_dir() and any(bm25_dir.iterdir())
    )
    if up_to_date:
        log("Corpus already up to date.")
        return True

    log(f"Syncing corpus from R2 ({remote_meta.get('corpus_size', '?')} papers, ~760 MB first time)...")
    core_files = ["corpus.db", "index_minilm.faiss", "embeddings_minilm.npy", "id_map.json", "build_meta.json"]

    def _dl(filename):
        dest = corpus_dir / filename
        # Skip files already present with the exact remote size (cheap re-sync).
        try:
            remote_size = s3.head_object(Bucket=bucket, Key=f"corpus/{filename}")["ContentLength"]
            if dest.exists() and dest.stat().st_size == remote_size:
                return filename + " (cached)"
        except Exception:
            pass
        s3.download_file(bucket, f"corpus/{filename}", str(dest))
        return filename

    failed = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_dl, f): f for f in core_files}
        for fut in as_completed(futures):
            try:
                name = fut.result()
            except Exception as exc:
                failed.append(f"{futures[fut]} ({exc})")
                continue
            log(f"  downloaded: {name}")
    if failed:
        log(f"Corpus sync failed for: {', '.join(failed)}")
        return (corpus_dir / "corpus.db").exists()

    # BM25 index dir (non-fatal)
    bm25_dir = corpus_dir / "bm25_index"
    bm25_dir.mkdir(exist_ok=True)
    try:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="corpus/bm25_index/"):
            for obj in page.get("Contents", []):
                fn = obj["Key"][len("corpus/bm25_index/"):]
                if fn:
                    s3.download_file(bucket, obj["Key"], str(bm25_dir / fn))
    except Exception as e:
        log(f"BM25 index sync failed (BM25 disabled): {e}")

    log("Corpus synced from R2.")
    return True


# =========================
# Query helpers / filters
# =========================

def get_date_range(option: str):
    today = date.today()
    return {
        "Last 3 Days": (today - timedelta(days=3), today),
        "Last Week": (today - timedelta(days=7), today),
        "Last Month": (today - timedelta(days=30), today),
        "All Time": (date(2000, 1, 1), today),
    }.get(option) or (_ for _ in ()).throw(ValueError(f"Unknown date range option: {option}"))


def build_query_brief(research_brief: str, not_looking_for: str) -> str:
    research_brief, not_looking_for = research_brief.strip(), not_looking_for.strip()
    parts = []
    if research_brief:
        parts.append("RESEARCH BRIEF:\n" + research_brief)
    if not_looking_for:
        parts.append("WHAT I AM NOT LOOKING FOR:\n" + not_looking_for)
    return "\n\n".join(parts) if parts else "The user did not provide any research brief."


def parse_not_terms(not_text: str) -> List[str]:
    import re
    not_text = not_text.strip()
    if not not_text:
        return []
    return [p.strip().lower() for p in re.split(r"[,\n;]+", not_text) if p.strip()]


def filter_papers_by_not_terms(papers: List[Paper], not_text: str):
    terms = parse_not_terms(not_text)
    if not terms or not papers:
        return papers, 0
    filtered, removed = [], 0
    for p in papers:
        text = f"{p.title} {p.abstract}".lower()
        if any(term in text for term in terms):
            removed += 1
        else:
            filtered.append(p)
    return filtered, removed


def filter_papers_by_venue(papers, venue_filter_type, selected_category, selected_venues):
    if venue_filter_type == "None":
        return papers
    filtered = []
    for p in papers:
        venue = (p.venue or "").lower()
        if venue_filter_type == "All Conferences":
            if any(c.lower() in venue for c in CONFERENCE_KEYWORDS):
                filtered.append(p)
        elif venue_filter_type == "All Journals":
            if any(j.lower() in venue for j in JOURNAL_KEYWORDS):
                filtered.append(p)
        elif venue_filter_type == "Specific Venue":
            if selected_venues and any(sel.lower() in venue for sel in selected_venues):
                filtered.append(p)
    return filtered


def fetch_papers_from_db(start_date, end_date, category_filter=None, subcats=None) -> List[Paper]:
    db_path = get_corpus_dir() / "corpus.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    query = "SELECT * FROM papers WHERE date(submitted_date) >= date(?) AND date(submitted_date) <= date(?)"
    params: List[Any] = [start_date.isoformat(), end_date.isoformat()]
    if category_filter and category_filter != "All":
        query += " AND fields_of_study LIKE ?"
        params.append(f"%{category_filter}%")
    if subcats and category_filter != "All":
        or_clauses = []
        for cat_code in subcats:
            cat_name = ARXIV_CODE_TO_NAME.get(cat_code, cat_code)
            words = [w for w in cat_name.split() if w.lower() not in ("and", "or", "of")]
            if words:
                keyword = " ".join(words[:2]) if len(words) > 1 else words[0]
                or_clauses.append("(title LIKE ? OR abstract LIKE ? OR fields_of_study LIKE ?)")
                params.extend([f"%{keyword}%", f"%{keyword}%", f"%{cat_code}%"])
        if or_clauses:
            query += " AND (" + " OR ".join(or_clauses) + ")"

    rows = conn.execute(query, params).fetchall()
    papers: List[Paper] = []
    for r in rows:
        d = dict(r)
        date_str = d.get("submitted_date", "2024-01-01")
        if "T" not in date_str:
            date_str = date_str + "T00:00:00"
        papers.append(Paper(
            arxiv_id=d["arxiv_id"], title=d["title"],
            authors=json.loads(d.get("authors") or "[]"), email_domains=[],
            abstract=d.get("abstract") or "",
            submitted_date=datetime.fromisoformat(date_str.replace("Z", "+00:00")),
            pdf_url=d.get("pdf_url") or "", arxiv_url=d.get("arxiv_url") or "",
            venue=d.get("venue"), source=d.get("source"),
        ))
    return papers


# =========================
# LLM call + JSON helper
# =========================

def call_llm(prompt: str, llm_config: LLMConfig, label: str = "") -> str:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if llm_config.provider == "openai":
                client_args = {"api_key": llm_config.api_key}
                if llm_config.api_base and llm_config.api_base.strip():
                    client_args["base_url"] = llm_config.api_base
                client = OpenAI(**client_args)
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt},
                ]
                kwargs: Dict[str, Any] = {"model": llm_config.model, "messages": messages}
                if not (llm_config.model.startswith("o1") or llm_config.model.startswith("gpt-5")):
                    kwargs["temperature"] = 0.2
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content

            elif llm_config.provider == "gemini":
                if genai is None:
                    raise RuntimeError("Gemini selected but google-genai is not installed.")
                client = genai.Client(api_key=llm_config.api_key)
                response = client.models.generate_content(model=llm_config.model, contents=prompt)
                if getattr(response, "candidates", None):
                    cand = response.candidates[0]
                    if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                        texts = [part.text for part in cand.content.parts
                                 if getattr(part, "text", None)]
                        if texts:
                            return "".join(texts)
                return getattr(response, "text", "")

            elif llm_config.provider == "groq":
                client = Groq(api_key=llm_config.api_key)
                response = client.chat.completions.create(
                    model=llm_config.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                return response.choices[0].message.content
            else:
                raise ValueError(f"Unknown provider: {llm_config.provider}")
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def safe_parse_json_array(raw: str) -> Optional[List[Dict[str, Any]]]:
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else None
    except Exception:
        return None


# =========================
# Embeddings + model singletons
# =========================

@lru_cache(maxsize=1)
def get_local_embed_model():
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed.")
    return SentenceTransformer(LOCAL_EMBED_MODEL)


def embed_texts_local(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    model = get_local_embed_model()
    return model.encode(texts, convert_to_numpy=True).tolist()


def cosine_similarity(vec1, vec2) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    n1 = sum(a * a for a in vec1)
    n2 = sum(b * b for b in vec2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (math.sqrt(n1) * math.sqrt(n2))


@lru_cache(maxsize=1)
def load_bm25_index():
    if not bm25s:
        return None, None
    base = get_corpus_dir()
    bm25_path, id_map_path = base / "bm25_index", base / "id_map.json"
    if not bm25_path.exists() or not id_map_path.exists():
        return None, None
    try:
        retriever = bm25s.BM25.load(str(bm25_path))
        id_map = json.load(open(id_map_path, encoding="utf-8"))
        arxiv_to_pos = {v: int(k) for k, v in id_map.items()}
        return retriever, arxiv_to_pos
    except Exception as e:
        print(f"Failed to load BM25 index: {e}")
        return None, None


def bm25_recall(papers, query_brief, retriever, arxiv_to_pos, n1: int = 600) -> List[Paper]:
    if not retriever or not papers:
        return papers
    paper_dict = {p.arxiv_id: p for p in papers}
    tokens = bm25s.tokenize([query_brief])
    try:
        res, _ = retriever.retrieve(tokens, k=n1)
    except Exception as e:
        print(f"BM25 retrieve error: {e}")
        return papers
    pos_to_arxiv = {v: k for k, v in arxiv_to_pos.items()}
    recalled, seen = [], set()
    for pos in res[0]:
        arxiv_id = pos_to_arxiv.get(int(pos))
        if arxiv_id and arxiv_id in paper_dict and arxiv_id not in seen:
            recalled.append(paper_dict[arxiv_id])
            seen.add(arxiv_id)
    if len(recalled) < 50:
        return papers  # too few intersecting; skip strict BM25 pruning
    return recalled


@lru_cache(maxsize=1)
def load_precomputed_embeddings():
    emb_path = get_corpus_dir() / "embeddings_minilm.npy"
    if not emb_path.exists():
        return None
    try:
        return np.load(str(emb_path), mmap_mode="r")
    except Exception as e:
        print(f"Failed to load precomputed embeddings: {e}")
        return None


def minilm_vector_rerank(papers, query_brief, embeddings, arxiv_to_pos, n2: int = 300) -> List[Paper]:
    if not papers:
        return []
    if embeddings is None or not arxiv_to_pos:
        texts = [p.title + "\n\n" + p.abstract for p in papers]
        paper_vecs = embed_texts_local(texts)
        q_vec = embed_texts_local([query_brief])[0]
        scored = []
        for p, vec in zip(papers, paper_vecs):
            p.semantic_relevance = cosine_similarity(q_vec, vec)
            scored.append((p.semantic_relevance, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:min(n2, len(scored))]]

    model = get_local_embed_model()
    q_vec = model.encode([query_brief], normalize_embeddings=True)[0]
    scored = []
    for p in papers:
        pos = arxiv_to_pos.get(p.arxiv_id)
        sim = float(np.dot(embeddings[pos], q_vec)) if (pos is not None and pos < embeddings.shape[0]) else 0.0
        p.semantic_relevance = sim
        scored.append((sim, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:min(n2, len(scored))]]


@lru_cache(maxsize=1)
def get_cross_encoder_model():
    if CrossEncoder is None:
        return None
    try:
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        print(f"CrossEncoder load error: {e}")
        return None


def cross_encoder_rerank(papers, query_brief, n3: int = 150) -> List[Paper]:
    if not papers:
        return []
    model = get_cross_encoder_model()
    if not model:
        return papers[:n3]
    pairs = [[query_brief, p.title + "\n\n" + p.abstract] for p in papers]
    try:
        scores = model.predict(pairs)
        scored = []
        for p, score in zip(papers, scores):
            s = float(score)
            p.semantic_relevance = 1 / (1 + math.exp(-s))
            scored.append((s, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:min(n3, len(scored))]]
    except Exception as e:
        print(f"CrossEncoder predict error: {e}")
        return papers[:n3]


# =========================
# Classification
# =========================

def classify_papers_with_llm(papers, query_brief, llm_config, batch_size: int = 15) -> List[Paper]:
    if not papers:
        return papers
    for batch_start in range(0, len(papers), batch_size):
        batch = papers[batch_start:batch_start + batch_size]
        paper_blocks = []
        for idx, p in enumerate(batch):
            paper_blocks.append(textwrap.dedent(f"""
            Paper {idx}:
            Title: {p.title}
            Abstract: {p.abstract}
            """).strip())
        instruction = textwrap.dedent(f"""
        You are given a user's research brief and a small set of papers.
        Brief: \"\"\"{query_brief}\"\"\"

        For each paper, decide:
          1. focus_label: "primary", "secondary", or "off-topic".
          2. relevance_score: float 0.0-1.0.
          3. reason: 1-2 sentence explanation.

        Return JSON array:
          [{{ "index": <int>, "focus_label": "...", "relevance_score": <float>, "reason": "..." }}]
        """).strip()
        prompt = "\n\n".join([instruction, "PAPERS:", *paper_blocks])
        try:
            raw = call_llm(prompt, llm_config, label="classification")
        except Exception:
            raw = ""
        parsed = safe_parse_json_array(raw)
        if parsed is None:
            continue
        idx_to_info = {}
        for item in parsed:
            try:
                idx = int(item["index"])
                label = str(item.get("focus_label", "")).strip().lower()
                if label not in ["primary", "secondary", "off-topic"]:
                    label = "off-topic"
                idx_to_info[idx] = {
                    "focus_label": label,
                    "relevance_score": float(item.get("relevance_score", 0.0)),
                    "reason": str(item.get("reason", "")).strip(),
                }
            except Exception:
                continue
        for idx, p in enumerate(batch):
            info = idx_to_info.get(idx)
            if info:
                p.focus_label = info["focus_label"]
                p.llm_relevance_score = info["relevance_score"]
                p.semantic_reason = info["reason"]
            else:
                p.focus_label = "off-topic"
                p.llm_relevance_score = 0.0
    return papers


def heuristic_classify_papers_free(candidates: List[Paper]) -> List[Paper]:
    if not candidates:
        return candidates
    ranked = sorted(candidates, key=lambda p: p.semantic_relevance or 0.0, reverse=True)
    n = len(ranked)
    top_k = max(1, min(n, max(10, int(0.3 * n))))
    for idx, p in enumerate(ranked):
        p.llm_relevance_score = p.semantic_relevance or 0.0
        p.focus_label = "primary" if idx < top_k else "secondary"
        if p.semantic_reason is None:
            p.semantic_reason = "Heuristic classification based on embedding similarity."
    return ranked


# =========================
# Moneyball impact scoring
# =========================

def get_s2_citation_stats(paper: Paper, api_key: Optional[str] = None) -> int:
    headers = {"x-api-key": api_key} if api_key else {}
    max_retries = 2

    def fetch(url, params):
        for attempt in range(max_retries + 1):
            try:
                r = requests.get(url, headers=headers, params=params, timeout=10)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
            except Exception:
                if attempt < max_retries:
                    time.sleep(1)
                    continue
        return None

    if paper.arxiv_id:
        clean_id = paper.arxiv_id.split("v")[0]
        data = fetch(f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{clean_id}",
                     {"fields": "authors.citationCount"})
        if data:
            auth_cites = [a.get("citationCount", 0) for a in data.get("authors", []) if a.get("citationCount")]
            if auth_cites:
                return max(auth_cites)

    data = fetch("https://api.semanticscholar.org/graph/v1/paper/search",
                 {"query": paper.title, "limit": 1, "fields": "title,citationCount,authors.citationCount"})
    if data and data.get("data"):
        auth_cites = [a.get("citationCount", 0) for a in data["data"][0].get("authors", []) if a.get("citationCount")]
        return max(auth_cites) if auth_cites else 0
    return 0


def predict_citations_direct(target_papers, llm_config, batch_size: int = 8,
                             progress_cb: Optional[Callable[[int, int], None]] = None) -> List[Paper]:
    """Moneyball predictor: hybrid author data + LLM narrative."""
    if not target_papers:
        return target_papers
    weights = DEFAULT_MONEYBALL_WEIGHTS
    if os.path.exists("moneyball_weights.json"):
        try:
            weights = json.load(open("moneyball_weights.json"))
        except Exception:
            pass
    s2_key = get_secret("S2_API_KEY")

    total = len(target_papers)
    for i, p in enumerate(target_papers):
        max_auth_cites = get_s2_citation_stats(p, s2_key)
        is_fresh = False
        try:
            if (datetime.now().date() - p.submitted_date.date()).days <= 5:
                is_fresh = True
        except Exception:
            pass

        if max_auth_cites > 0:
            h1_fame = min(math.log(max_auth_cites + 1) * 8, 95)
            fame_label = "real"
        elif is_fresh:
            fame_label, h1_fame = "too_new", 0.0
        else:
            h1_fame, fame_label = 0.0, "none"

        if not s2_key:
            time.sleep(0.3)

        t_lower = p.title.lower()
        h2_hype = 0
        if "benchmark" in t_lower or "dataset" in t_lower:
            h2_hype += 50
        if "survey" in t_lower:
            h2_hype += 40
        if "llm" in t_lower:
            h2_hype += 10
        h3_sniper = 0
        if "benchmark" in t_lower:
            h3_sniper += 50
        if any(n in t_lower for n in ["lidar", "3d", "audio", "wireless", "agriculture", "traffic", "physics"]):
            h3_sniper -= 20

        prompt = textwrap.dedent(f"""
            Analyze this abstract.
            1. Rate 'Citation Potential' (0-10) based on market fit (Broad/Hot = High, Niche = Low).
            2. Write 2 short, plain English sentences explaining the score.
               - Sentence 1 (Market Fit): Why is this topic hot or niche? (Do NOT start with "Market Fit:")
               - Sentence 2 (Contribution): What is the specific value? (Do NOT start with "Contribution:")

            Title: {p.title}
            Abstract: {p.abstract[:800]}...

            Return JSON: {{ "score": <int>, "bullets": ["string", "string"] }}
        """)
        h4_utility = 50.0
        content_bullets = [
            "The topic appears relevant to current research trends.",
            "The paper proposes a specific contribution to the field.",
        ]
        if llm_config.provider in ("openai", "gemini", "groq"):
            try:
                raw = call_llm(prompt, llm_config, label="moneyball_narrative")
                if raw:
                    if "```" in raw:
                        parts = raw.split("```json")
                        raw = parts[1].split("```")[0] if len(parts) > 1 else raw.split("```")[1].split("```")[0]
                    parsed = json.loads(raw.strip())
                    h4_utility = float(parsed.get("score", 5) * 10)
                    if isinstance(parsed.get("bullets"), list):
                        content_bullets = [b.replace("Market Fit:", "").replace("Contribution:", "").strip()
                                           for b in parsed["bullets"]][:2]
            except Exception:
                pass

        if fame_label == "too_new":
            p.predicted_citations = -1.0
        else:
            p.predicted_citations = (
                h1_fame * weights["weight_fame"] + h2_hype * weights["weight_hype"]
                + h3_sniper * weights["weight_sniper"] + h4_utility * weights["weight_utility"]
            )

        final_bullets = []
        if fame_label == "real":
            if max_auth_cites > 3000:
                final_bullets.append("🚀 **Distribution:** High influence author/lab.")
            elif max_auth_cites > 500:
                final_bullets.append("📢 **Reach:** Established track record.")
            elif max_auth_cites > 100:
                final_bullets.append("📈 **Momentum:** Authors have prior traction.")
            else:
                final_bullets.append("🌱 **Emerging:** Newer authors; relies on merit.")
        elif fame_label == "too_new":
            final_bullets.append("🆕 **Too new for impact score:** Citation data unavailable. Ranked by relevance only.")
            if p.semantic_reason:
                final_bullets.append(f"✨ **Relevance Insight:** Ranked highly because: {p.semantic_reason}")
        else:
            final_bullets.append("🌱 **Emerging:** Unknown authors.")
        if len(content_bullets) >= 1:
            final_bullets.append(f"🎯 **Market Fit:** {content_bullets[0]}")
        if len(content_bullets) >= 2:
            final_bullets.append(f"💡 **Contribution:** {content_bullets[1]}")
        p.prediction_explanations = final_bullets

        if progress_cb:
            progress_cb(i + 1, total)
    return target_papers


def assign_heuristic_citations_free(papers: List[Paper]) -> List[Paper]:
    if not papers:
        return papers
    scores = [(p.llm_relevance_score or 0.0) * 0.7 + (p.semantic_relevance or 0.0) * 0.3 for p in papers]
    min_s, max_s = min(scores), max(scores)
    for p, s in zip(papers, scores):
        norm = (s - min_s) / (max_s - min_s) if max_s > min_s else 0.5
        p.predicted_citations = float(int(10 + norm * 40))
    return papers


def summarize_paper_plain_english(paper: Paper, llm_config: LLMConfig) -> str:
    prompt = textwrap.dedent(f"""
    Explain this research paper to a non-expert.
    Title: {paper.title}
    Abstract: {paper.abstract}

    Provide 3-6 plain English bullet points covering main idea, problem solved, and takeaways.
    """).strip()
    return call_llm(prompt, llm_config, label="plain_english_summary")
