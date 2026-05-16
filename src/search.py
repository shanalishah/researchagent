from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import feedparser


ARXIV_CATEGORY_QUERY = (
    "cat:cs.AI OR "
    "cat:cs.LG OR "
    "cat:cs.CL OR "
    "cat:cs.CV OR "
    "cat:cs.HC OR "
    "cat:cs.IR OR "
    "cat:cs.RO OR "
    "cat:cs.SE OR "
    "cat:cs.DB OR "
    "cat:cs.MA"
)


def get_date_range(date_range: str) -> tuple[date, date]:
    today = date.today()

    if date_range == "Last 3 Days":
        return today - timedelta(days=3), today

    if date_range == "Last Week":
        return today - timedelta(days=7), today

    if date_range == "Last Month":
        return today - timedelta(days=30), today

    if date_range == "Last 6 Months":
        return today - timedelta(days=180), today

    return date(2000, 1, 1), today


def clean_text(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def parse_arxiv_date(value: str) -> Optional[datetime]:
    if not value:
        return None

    try:
        return datetime.strptime(value[:10], "%Y-%m-%d")
    except Exception:
        return None


def simple_relevance_score(query: str, title: str, abstract: str) -> float:
    """
    Lightweight relevance scoring.
    This is not your final Moneyball model.
    It gives the public UI a working search/ranking behavior.
    """
    query_terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9]+", query)
        if len(term) > 2
    }

    if not query_terms:
        return 0.0

    title_terms = set(re.findall(r"[A-Za-z0-9]+", title.lower()))
    abstract_terms = set(re.findall(r"[A-Za-z0-9]+", abstract.lower()))

    title_hits = len(query_terms.intersection(title_terms))
    abstract_hits = len(query_terms.intersection(abstract_terms))

    raw_score = (title_hits * 2.5) + abstract_hits
    max_possible = max(len(query_terms) * 3.5, 1)

    return min(raw_score / max_possible, 1.0)


def simple_impact_score(relevance: float, submitted_date: Optional[datetime]) -> Optional[int]:
    """
    Temporary public-demo impact score.
    Later, replace this with your actual Moneyball/citation engine.
    """
    if submitted_date is None:
        return None

    age_days = max((datetime.utcnow() - submitted_date).days, 0)

    if age_days < 5:
        return None

    recency_bonus = max(0, 1 - (age_days / 365))
    score = 45 + (relevance * 40) + (recency_bonus * 15)

    return int(round(min(score, 99)))


def should_exclude_paper(title: str, abstract: str, exclude_topics: str) -> bool:
    if not exclude_topics.strip():
        return False

    text = f"{title} {abstract}".lower()
    terms = [
        item.strip().lower()
        for item in re.split(r"[,;\n]+", exclude_topics)
        if item.strip()
    ]

    return any(term in text for term in terms)


def build_arxiv_url(query: str, max_results: int) -> str:
    user_query = clean_text(query)

    if user_query:
        search_query = f"({ARXIV_CATEGORY_QUERY}) AND all:{user_query}"
    else:
        search_query = f"({ARXIV_CATEGORY_QUERY})"

    return (
        "https://export.arxiv.org/api/query?"
        f"search_query={quote_plus(search_query)}"
        "&sortBy=submittedDate"
        "&sortOrder=descending"
        f"&max_results={max_results}"
    )


def run_research_search(
    query: str,
    date_range: str,
    provider: str = "Free Local",
    exclude_topics: str = "",
    max_results: int = 25,
) -> List[Dict]:
    """
    Public Streamlit search function.

    Current version:
    - Fetches recent AI/CS papers from arXiv.
    - Filters by date range.
    - Filters excluded topics.
    - Applies lightweight relevance and demo impact scoring.
    - Returns result dictionaries for the UI.

    Later replacement:
    - Swap this function with your full Research Agent pipeline.
    """
    start_date, end_date = get_date_range(date_range)

    feed_url = build_arxiv_url(query=query, max_results=max_results * 4)
    feed = feedparser.parse(feed_url)

    results: List[Dict] = []

    for entry in feed.entries:
        title = clean_text(getattr(entry, "title", ""))
        abstract = clean_text(getattr(entry, "summary", ""))

        submitted = parse_arxiv_date(getattr(entry, "published", ""))

        if submitted:
            submitted_day = submitted.date()
            if submitted_day < start_date or submitted_day > end_date:
                continue

        if should_exclude_paper(title, abstract, exclude_topics):
            continue

        authors = []
        for author in getattr(entry, "authors", []):
            name = clean_text(author.get("name", ""))
            if name:
                authors.append(name)

        arxiv_url = getattr(entry, "link", "")

        pdf_url = ""
        for link in getattr(entry, "links", []):
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")

        relevance = simple_relevance_score(query, title, abstract)
        impact_score = simple_impact_score(relevance, submitted)

        if relevance <= 0 and query.strip():
            continue

        why = [
            "Matches the research brief based on title and abstract keywords.",
            "Recent arXiv paper from a relevant AI or computer science category.",
        ]

        if impact_score is None:
            why.append("Too new or missing enough signal for an impact score.")
        else:
            why.append("Assigned a lightweight demo impact score for ranking.")

        results.append(
            {
                "rank": 0,
                "title": title,
                "authors": authors,
                "venue": "arXiv",
                "score": impact_score,
                "focus": "primary" if relevance >= 0.35 else "secondary",
                "relevance": relevance,
                "abstract": abstract,
                "why": why,
                "arxiv_url": arxiv_url,
                "pdf_url": pdf_url,
                "submitted_date": submitted.strftime("%Y-%m-%d") if submitted else "N/A",
                "provider": provider,
            }
        )

    results.sort(
        key=lambda paper: (
            paper["score"] if paper["score"] is not None else -1,
            paper["relevance"],
        ),
        reverse=True,
    )

    for index, paper in enumerate(results[:max_results], start=1):
        paper["rank"] = index

    return results[:max_results]
