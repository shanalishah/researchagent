import streamlit as st

from src.ui import apply_branding, render_footer


st.set_page_config(
    page_title="About Research Agent",
    page_icon="ℹ️",
    layout="centered",
)

apply_branding()

st.title("About Research Agent")

st.write(
    """
    Research Agent helps users find, rank, and understand recent AI research papers.
    The goal is to reduce the time researchers, builders, students, and public-interest
    teams spend searching for useful papers.
    """
)

st.subheader("What it does")

st.markdown(
    """
    - Searches recent AI and computer science papers from arXiv.
    - Ranks results using relevance and lightweight impact signals.
    - Explains why each paper appears in the results.
    - Lets users download results as Markdown or CSV.
    """
)

st.subheader("Current public version")

st.write(
    """
    This Streamlit version is a clean public interface. It currently connects to arXiv
    and uses a lightweight local ranking method. The deeper Research Agent pipeline
    can be connected later through the same `src/search.py` function.
    """
)

st.subheader("Long-term direction")

st.markdown(
    """
    The full version can include:

    - Local research corpus search
    - BM25 recall
    - Vector reranking
    - Cross-encoder reranking
    - Moneyball-style citation impact prediction
    - OpenAI, Gemini, Groq, and local model support
    - Cloudflare R2 corpus sync
    """
)

st.subheader("Principles")

st.markdown(
    """
    - Useful tools
    - Public benefit
    - Open by default
    - Privacy first
    - Humility
    """
)

render_footer()
