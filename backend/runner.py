"""Pipeline orchestrator: runs the extracted pipeline and yields SSE events.

`pipeline_events(req)` is a SYNCHRONOUS generator that yields plain dicts
(matching the StageEvent/DoneEvent/ErrorEvent schemas). The FastAPI endpoint
drives it one step at a time inside a worker thread, so each heavy stage runs
off the event loop while progress streams to the browser.

Stage mapping (to the frontend's 6-step loader):
  0 BM25 Recall   -> fetch + filters + BM25 lexical recall
  1 Vector Rerank -> MiniLM precomputed-embedding rerank
  2 CrossEncoder  -> cross-encoder precision rerank
  3 LLM Classify  -> relevance classification + build scoring set
  4 Impact Score  -> Moneyball citation scoring (or free heuristic)
  5 Summarize     -> plain-English summaries for the top N
"""

import time
import threading
from typing import List, Dict, Any

from . import pipeline_core as pc
from .models import PaperOut

# Guard so the ~760 MB corpus is only synced once per process.
_corpus_lock = threading.Lock()
_corpus_ready = False


def ensure_corpus(log=print) -> bool:
    global _corpus_ready
    with _corpus_lock:
        if _corpus_ready:
            return True
        ok = pc.download_corpus_artifacts(log=log)
        _corpus_ready = ok
        return ok


def _provider_to_internal(provider: str) -> str:
    return "free_local" if provider == "free" else provider


def _make_llm_config(req) -> pc.LLMConfig:
    provider = _provider_to_internal(req.provider)
    api_base = "https://api.openai.com/v1" if provider == "openai" else ""
    return pc.LLMConfig(
        api_key=req.api_key or "",
        model=req.model or "",
        api_base=api_base,
        provider=provider,
    )


def _clean_bullets(bullets) -> List[str]:
    # The Moneyball explanations contain **markdown** bold; strip it for the
    # plain-text React rendering (emojis kept).
    return [b.replace("**", "").strip() for b in (bullets or [])]


def _to_paper_out(p: pc.Paper, rank: int, summary: str | None) -> PaperOut:
    pred = p.predicted_citations
    too_new = (pred == -1.0)
    relevance = p.llm_relevance_score if p.llm_relevance_score is not None else (p.semantic_relevance or 0.0)
    relevance = max(0.0, min(1.0, float(relevance)))
    return PaperOut(
        rank=rank,
        arxiv_id=p.arxiv_id,
        title=p.title,
        authors=p.authors or [],
        venue=p.venue,
        abstract=p.abstract,
        arxiv_url=p.arxiv_url,
        pdf_url=p.pdf_url,
        score=(-1.0 if too_new else float(pred or 0.0)),
        too_new=too_new,
        focus=(p.focus_label if p.focus_label in ("primary", "secondary", "off-topic") else "off-topic"),
        relevance=relevance,
        why=_clean_bullets(p.prediction_explanations),
        summary=summary,
    )


def _sort_group(group: List[pc.Paper]) -> List[pc.Paper]:
    scored = [p for p in group if p.predicted_citations is not None and p.predicted_citations >= 0]
    unscored = [p for p in group if p.predicted_citations == -1.0]
    scored.sort(key=lambda p: p.predicted_citations, reverse=True)
    unscored.sort(key=lambda p: (p.llm_relevance_score or 0.0, p.semantic_relevance or 0.0), reverse=True)
    return scored + unscored


def _stage(index: int, status: str, name: str, detail: str = "") -> Dict[str, Any]:
    return {"type": "stage", "index": index, "status": status, "name": name, "detail": detail}


def pipeline_events(req):
    """Synchronous generator yielding SSE event dicts."""
    if not ensure_corpus():
        yield {"type": "error", "message": "Corpus is not available (R2 sync failed). Check backend logs."}
        return
    t0 = time.perf_counter()
    is_llm = req.provider in ("openai", "gemini", "groq")
    llm_config = _make_llm_config(req)

    brief = (req.query or "").strip()
    not_text = (req.exclude or "").strip()
    query_brief = pc.build_query_brief(brief, not_text)
    mode = "global" if (not brief and not not_text) else "targeted"

    start_date, end_date = pc.get_date_range(req.date_range)

    # ---- Stage 0: fetch + filters + BM25 recall ----
    yield _stage(0, "start", "BM25 Recall")
    current = pc.fetch_papers_from_db(start_date, end_date, category_filter=None, subcats=None)
    if not_text:
        current, _ = pc.filter_papers_by_not_terms(current, not_text)
    if not current:
        yield {"type": "error", "message": "No papers found in the corpus for that date range."}
        return

    if mode == "global":
        current.sort(key=lambda p: p.submitted_date, reverse=True)
        stage1 = current[:150]
    else:
        retriever, arxiv_to_pos = pc.load_bm25_index()
        stage1 = pc.bm25_recall(current, query_brief, retriever, arxiv_to_pos, n1=600) if retriever else current
    yield _stage(0, "done", "BM25 Recall", f"{len(current):,} → {len(stage1):,} papers")

    # ---- Stage 1: MiniLM vector rerank ----
    yield _stage(1, "start", "Vector Rerank")
    if mode == "global":
        stage2 = stage1
    else:
        embeddings = pc.load_precomputed_embeddings()
        _, arxiv_to_pos = pc.load_bm25_index()
        stage2 = pc.minilm_vector_rerank(stage1, query_brief, embeddings, arxiv_to_pos or {}, n2=300)
    yield _stage(1, "done", "Vector Rerank", f"{len(stage1):,} → {len(stage2):,} papers")

    # ---- Stage 2: cross-encoder rerank ----
    yield _stage(2, "start", "CrossEncoder")
    candidates = stage2 if mode == "global" else pc.cross_encoder_rerank(stage2, query_brief, n3=150)
    yield _stage(2, "done", "CrossEncoder", f"{len(stage2):,} → {len(candidates):,} papers")

    # ---- Stage 3: classify + build scoring set ----
    yield _stage(3, "start", "LLM Classify")
    if mode == "global":
        for p in candidates:
            p.focus_label = "primary"
            p.semantic_reason = p.semantic_reason or "Global mode: treated as primary."
    elif is_llm:
        candidates = pc.classify_papers_with_llm(candidates, query_brief, llm_config, batch_size=15)
    else:
        candidates = pc.heuristic_classify_papers_free(candidates)

    primary = [p for p in candidates if p.focus_label == "primary"]
    secondary = [p for p in candidates if p.focus_label == "secondary"]
    for group in (primary, secondary):
        group.sort(key=lambda p: (p.llm_relevance_score or 0.0, p.semantic_relevance or 0.0), reverse=True)

    if mode == "global":
        used = candidates[:]
    elif primary:
        used = primary[:]
        if len(primary) < pc.MIN_FOR_PREDICTION and secondary:
            used.extend(secondary[:pc.MIN_FOR_PREDICTION - len(primary)])
    elif secondary:
        used = secondary[:]
    else:
        yield {"type": "error", "message": "No relevant papers found for that brief."}
        return
    used.sort(key=lambda p: (p.llm_relevance_score or 0.0, p.semantic_relevance or 0.0), reverse=True)
    yield _stage(3, "done", "LLM Classify", f"{len(primary)} primary, {len(secondary)} secondary")

    # ---- Stage 4: Moneyball impact scoring ----
    yield _stage(4, "start", "Impact Score")
    if is_llm:
        used = pc.predict_citations_direct(used, llm_config)
    else:
        used = pc.assign_heuristic_citations_free(used)
    primaries = [p for p in used if p.focus_label == "primary"]
    secondaries = [p for p in used if p.focus_label == "secondary"]
    others = [p for p in used if p.focus_label not in ("primary", "secondary")]
    ranked = _sort_group(primaries) + _sort_group(secondaries) + _sort_group(others)
    yield _stage(4, "done", "Impact Score", f"{len(ranked)} papers scored")

    # ---- Stage 5: plain-English summaries for top N ----
    yield _stage(5, "start", "Summarize")
    top_n = min(req.top_n, len(ranked))
    summaries: Dict[str, str] = {}
    if is_llm:
        for p in ranked[:top_n]:
            try:
                summaries[p.arxiv_id] = pc.summarize_paper_plain_english(p, llm_config)
            except Exception:
                summaries[p.arxiv_id] = None
    yield _stage(5, "done", "Summarize", f"Top {top_n} summarized" if is_llm else "Heuristic mode — no summaries")

    papers_out = [_to_paper_out(p, i + 1, summaries.get(p.arxiv_id)) for i, p in enumerate(ranked)]
    yield {
        "type": "done",
        "papers": [po.model_dump() for po in papers_out],
        "primary_count": len(primaries),
        "secondary_count": len(secondaries),
        "total_seconds": round(time.perf_counter() - t0, 1),
        "provider": req.provider,
    }
