import streamlit as st

from src.search import run_research_search
from src.ui import (
    apply_branding,
    render_downloads,
    render_footer,
    render_hero,
    render_result_card,
)


st.set_page_config(
    page_title="Research Agent",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="collapsed",
)

apply_branding()


if "results" not in st.session_state:
    st.session_state.results = []

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "last_date_range" not in st.session_state:
    st.session_state.last_date_range = "Last Month"

if "last_provider" not in st.session_state:
    st.session_state.last_provider = "Free Local"


render_hero()

with st.container(border=True):
    query = st.text_area(
        "Describe what you're looking for",
        placeholder="Example: recent papers on multi-agent systems for healthcare workflows",
        height=120,
    )

    col1, col2 = st.columns(2)

    with col1:
        date_range = st.selectbox(
            "Date range",
            ["Last 3 Days", "Last Week", "Last Month", "Last 6 Months", "All Time"],
            index=2,
        )

    with col2:
        provider = st.selectbox(
            "Reasoning provider",
            ["Free Local", "OpenAI", "Gemini", "Groq"],
            index=0,
            help="This starter version uses Free Local search. Provider is saved for future backend connection.",
        )

    exclude_topics = st.text_input(
        "Exclude topics",
        placeholder="Example: surveys, generic LLM papers, robotics-only papers",
    )

    max_results = st.slider(
        "Number of results",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
    )

    search_clicked = st.button(
        "Search",
        type="primary",
        use_container_width=True,
    )


if search_clicked:
    if not query.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.last_query = query
        st.session_state.last_date_range = date_range
        st.session_state.last_provider = provider

        with st.spinner("Searching arXiv and ranking papers..."):
            st.session_state.results = run_research_search(
                query=query,
                date_range=date_range,
                provider=provider,
                exclude_topics=exclude_topics,
                max_results=max_results,
            )

        if not st.session_state.results:
            st.info(
                "No matching papers were found. Try a broader query, a longer date range, "
                "or remove excluded topics."
            )


if st.session_state.results:
    st.divider()

    st.subheader(f"Results for: {st.session_state.last_query}")

    primary_count = sum(
        1 for paper in st.session_state.results if paper.get("focus") == "primary"
    )
    secondary_count = len(st.session_state.results) - primary_count

    st.caption(
        f"{len(st.session_state.results)} results · "
        f"{primary_count} primary · "
        f"{secondary_count} secondary · "
        f"{st.session_state.last_date_range}"
    )

    render_downloads(
        query=st.session_state.last_query,
        date_range=st.session_state.last_date_range,
        provider=st.session_state.last_provider,
        papers=st.session_state.results,
    )

    st.write("")

    for paper in st.session_state.results:
        render_result_card(paper)


render_footer()
