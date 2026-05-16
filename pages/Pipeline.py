import pandas as pd
import streamlit as st

from src.ui import apply_branding, render_footer


st.set_page_config(
    page_title="Search Pipeline",
    page_icon="⚙️",
    layout="centered",
)

apply_branding()

st.title("Search Pipeline")

st.write(
    """
    This page explains how the public Streamlit version currently works and where the
    full Research Agent pipeline can be connected later.
    """
)

st.subheader("Current Streamlit pipeline")

current_steps = [
    {
        "Step": 1,
        "Stage": "User Query",
        "Description": "User describes what kind of research they are looking for.",
    },
    {
        "Step": 2,
        "Stage": "arXiv Fetch",
        "Description": "The app searches recent AI and computer science papers from arXiv.",
    },
    {
        "Step": 3,
        "Stage": "Date Filtering",
        "Description": "Results are filtered based on the selected date range.",
    },
    {
        "Step": 4,
        "Stage": "Exclude Topics",
        "Description": "Papers containing excluded terms are removed.",
    },
    {
        "Step": 5,
        "Stage": "Local Ranking",
        "Description": "Papers are ranked using lightweight title and abstract relevance.",
    },
    {
        "Step": 6,
        "Stage": "Results + Export",
        "Description": "The app displays cards and lets users download Markdown or CSV results.",
    },
]

st.dataframe(pd.DataFrame(current_steps), use_container_width=True, hide_index=True)

st.subheader("Future full Research Agent pipeline")

future_steps = [
    {
        "Step": 1,
        "Stage": "BM25 Recall",
        "Description": "Retrieve a larger pool of candidate papers from the local corpus.",
    },
    {
        "Step": 2,
        "Stage": "Vector Rerank",
        "Description": "Use embeddings to rerank candidates by semantic relevance.",
    },
    {
        "Step": 3,
        "Stage": "CrossEncoder",
        "Description": "Apply a deeper relevance model to the top candidates.",
    },
    {
        "Step": 4,
        "Stage": "LLM Classification",
        "Description": "Classify papers as primary, secondary, or off-topic.",
    },
    {
        "Step": 5,
        "Stage": "Moneyball Impact Score",
        "Description": "Predict likely citation impact using citation and author signals.",
    },
    {
        "Step": 6,
        "Stage": "Plain-English Summary",
        "Description": "Generate human-readable summaries and explanations.",
    },
]

st.dataframe(pd.DataFrame(future_steps), use_container_width=True, hide_index=True)

render_footer()
