from __future__ import annotations

import io
from typing import Dict, List

import pandas as pd
import streamlit as st


BRAND_CSS = """
<style>
:root {
    --cream: #FAF7F2;
    --card: #FFFFFF;
    --fg: #1C3041;
    --dim: #7A8A95;
    --line: #E8E4DD;
    --teal: #2B6B60;
    --teal-dark: #1E4F47;
    --teal-soft: #E2EEEC;
    --warm: #B8860B;
}

.stApp {
    background: var(--cream);
    color: var(--fg);
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
}

.ra-hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
}

.ra-logo {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: var(--teal);
    color: white;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-family: Georgia, serif;
    margin-bottom: 0.75rem;
}

.ra-title {
    font-family: Georgia, serif;
    font-size: 2.7rem;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}

.ra-subtitle {
    color: var(--dim);
    font-size: 1.05rem;
    max-width: 620px;
    margin: 0 auto;
}

.ra-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 2px 12px rgba(28,48,65,0.04);
}

.ra-rank {
    background: var(--teal);
    color: white;
    border-radius: 8px;
    padding: 0.35rem 0.55rem;
    font-weight: 700;
    font-size: 0.8rem;
}

.ra-badge-primary {
    color: var(--teal);
    background: var(--teal-soft);
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
}

.ra-badge-secondary {
    color: var(--dim);
    background: var(--line);
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
}

.ra-meta {
    color: var(--dim);
    font-size: 0.85rem;
}

.ra-small {
    color: var(--dim);
    font-size: 0.8rem;
}

.ra-footer {
    text-align: center;
    color: var(--dim);
    font-size: 0.8rem;
    padding: 2rem 0 1rem 0;
}

button[kind="primary"] {
    background-color: var(--teal) !important;
}
</style>
"""


def apply_branding() -> None:
    st.markdown(BRAND_CSS, unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown(
        """
        <div class="ra-hero">
            <div class="ra-logo">b²</div>
            <div class="ra-title">Find the research that matters</div>
            <div class="ra-subtitle">
                Search recent AI papers, rank them by relevance, and get plain-English explanations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_to_markdown(paper: Dict) -> str:
    authors = ", ".join(paper.get("authors", [])) or "N/A"
    score = paper.get("score")
    score_text = "Too new to score" if score is None else str(score)

    lines = [
        f"## #{paper.get('rank')} — {paper.get('title')}",
        "",
        f"- **Authors:** {authors}",
        f"- **Venue:** {paper.get('venue', 'N/A')}",
        f"- **Submitted:** {paper.get('submitted_date', 'N/A')}",
        f"- **Focus:** {paper.get('focus', 'N/A')}",
        f"- **Impact score:** {score_text}",
        f"- **Relevance:** {paper.get('relevance', 0):.0%}",
        f"- **arXiv:** {paper.get('arxiv_url', '')}",
        f"- **PDF:** {paper.get('pdf_url', '')}",
        "",
        "**Abstract**",
        "",
        paper.get("abstract", ""),
        "",
        "**Why this ranking**",
        "",
    ]

    for reason in paper.get("why", []):
        lines.append(f"- {reason}")

    lines.append("\n---\n")
    return "\n".join(lines)


def results_to_markdown(query: str, date_range: str, provider: str, papers: List[Dict]) -> str:
    lines = [
        "# Research Agent Results",
        "",
        f"**Query:** {query}",
        f"**Date range:** {date_range}",
        f"**Provider:** {provider}",
        "",
        "---",
        "",
    ]

    for paper in papers:
        lines.append(result_to_markdown(paper))

    return "\n".join(lines)


def results_to_dataframe(papers: List[Dict]) -> pd.DataFrame:
    rows = []

    for paper in papers:
        rows.append(
            {
                "Rank": paper.get("rank"),
                "Title": paper.get("title"),
                "Authors": ", ".join(paper.get("authors", [])),
                "Venue": paper.get("venue"),
                "Submitted": paper.get("submitted_date"),
                "Focus": paper.get("focus"),
                "Impact Score": "Too new" if paper.get("score") is None else paper.get("score"),
                "Relevance": round(float(paper.get("relevance", 0)), 3),
                "arXiv URL": paper.get("arxiv_url"),
                "PDF URL": paper.get("pdf_url"),
            }
        )

    return pd.DataFrame(rows)


def render_result_card(paper: Dict) -> None:
    rank = paper.get("rank", "")
    title = paper.get("title", "Untitled")
    authors = ", ".join(paper.get("authors", [])) or "N/A"
    focus = paper.get("focus", "secondary")
    badge_class = "ra-badge-primary" if focus == "primary" else "ra-badge-secondary"

    score = paper.get("score")
    score_text = "—" if score is None else str(score)
    score_label = "Too new" if score is None else "Impact"

    arxiv_url = paper.get("arxiv_url", "")
    pdf_url = paper.get("pdf_url", "")

    with st.container(border=False):
        st.markdown('<div class="ra-card">', unsafe_allow_html=True)

        top_left, top_right = st.columns([0.82, 0.18])

        with top_left:
            st.markdown(
                f"""
                <div style="display:flex; gap:0.65rem; align-items:center; margin-bottom:0.4rem;">
                    <span class="ra-rank">#{rank}</span>
                    <span class="{badge_class}">{focus}</span>
                    <span class="ra-meta">{paper.get("venue", "N/A")}</span>
                </div>
                <h3 style="font-family:Georgia,serif; margin-bottom:0.2rem;">{title}</h3>
                <div class="ra-meta">{authors}</div>
                """,
                unsafe_allow_html=True,
            )

        with top_right:
            st.markdown(
                f"""
                <div style="text-align:right;">
                    <div style="font-size:1.6rem; font-weight:800; color:#2B6B60;">{score_text}</div>
                    <div class="ra-small">{score_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write(paper.get("abstract", ""))

        with st.expander("Why this paper ranked here"):
            for reason in paper.get("why", []):
                st.markdown(f"- {reason}")

            st.caption(
                f"Submitted: {paper.get('submitted_date', 'N/A')} | "
                f"Relevance: {paper.get('relevance', 0):.0%}"
            )

        link_cols = st.columns([1, 1, 4])

        with link_cols[0]:
            if arxiv_url:
                st.link_button("arXiv", arxiv_url)

        with link_cols[1]:
            if pdf_url:
                st.link_button("PDF", pdf_url)

        st.markdown("</div>", unsafe_allow_html=True)


def render_downloads(query: str, date_range: str, provider: str, papers: List[Dict]) -> None:
    markdown_text = results_to_markdown(query, date_range, provider, papers)
    df = results_to_dataframe(papers)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download Markdown Report",
            data=markdown_text,
            file_name="research_agent_results.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name="research_agent_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_footer() -> None:
    st.markdown(
        """
        <div class="ra-footer">
            © The Benevolent Bandwidth Foundation, Inc. · Massachusetts Nonprofit Corporation.
            All rights reserved. Built with ❤️ for humanity.
        </div>
        """,
        unsafe_allow_html=True,
    )
