🔎 Research Agent v5.0

A lightweight, intelligent research assistant that fetches, ranks, and explains recent AI papers from arXiv.

It runs fully in Streamlit and acts as a "Universal Brain" for research—you can plug in OpenAI, Google Gemini, or Groq to handle the reasoning, or run it entirely offline with a Free Local Model.

It produces ranked tables, citation impact scores, and "Plain English" summaries for the top papers.

🌐 Try the Live App: https://research-aiagent.streamlit.app/



🚀 What's New

⚾ The "Moneyball" Ranking Engine

We have completely overhauled how papers are ranked. We stopped asking LLMs to guess "Will this paper be famous?" (which turned out to be unreliable) and built a Moneyball Algorithm.

Hard Data (84% Weight): The agent now queries Semantic Scholar to analyze the "Author Velocity" (recent citation momentum) of the paper's authors.

Soft Data (16% Weight): The LLM analyzes the abstract for "Market Fit" (Is this a trending topic?) and "Novelty."

The Result: A 6x increase in precision@10, meaning the papers at the top of your list are statistically much more likely to be influential.

⚡ Groq Integration (Llama 3.3)

You can now use Groq as your intelligence provider. This allows you to run high-performance open-source models (like Llama 3.3 70B) with blazing fast inference speeds and free API access.

Note: In Groq mode, the agent uses local embeddings for search and the Groq LLM for reasoning.

🏷️ Smart Venue Filtering

Filter papers by publication venue without losing semantic quality:

Filter Types: "All Conferences", "All Journals", or "Specific Venue"

Specific Selection: Pick individual top-tier venues like NeurIPS, CVPR, ICML, Nature, or Science

Smart Pipeline: The agent performs semantic search on all papers first to understand the landscape, and then applies your venue filter. This ensures the AI sees the full context of related work before narrowing down.

🛡️ Robust arXiv Fetching

Improved rate-limiting logic to strictly adhere to arXiv's API policies

Prevents IP bans during large data fetches by using smart backoff (3-second delays)

🔌 Four Intelligence Modes

You can choose your "Brain" from the sidebar:

1. OpenAI Mode (Best Quality)

Requires: OpenAI API Key

Models: GPT-5.2, GPT-5, GPT-5-mini, GPT-5-nano, GPT-4o, GPT-4.1, GPT-4.1-mini, GPT-4.0-mini, GPT-o1

Best For: Highest quality reasoning, summaries, and narrative analysis. Uses OpenAI embeddings for search.

2. Gemini Mode (Best Context/Price)

Requires: Google Gemini API Key

Models: Gemini 3 Pro (Preview), Gemini 2.5 Flash, Gemini 2.5 Pro, Gemini 2.0 Flash

Best For: Processing large batches of papers quickly. Uses Gemini embeddings.

3. Groq Mode (Fastest / Open Source)

Requires: Groq API Key (Currently Free)

Models: Llama 3.3 70B, Llama 3.1 8B

Best For: Fast speed and using open-weight models.

Architecture: Uses Local Embeddings (MiniLM) for search + Groq for classification/scoring.

4. Free Local Mode (Offline / Privacy)

Requires: No API Key. Runs on CPU.

Models: SentenceTransformers (MiniLM-L6-v2) + Heuristics

Best For: Offline use, zero cost, and quick browsing without LLM summaries.

🧠 How It Works (The Pipeline)

The agent follows a 9-step pipeline to distill hundreds of papers into the top 3:

Briefing: You provide a natural language "Research Brief" (e.g., "Agents that use tool use") and optional constraints.

Fetching: It downloads recent metadata from cs.AI, cs.LG, and cs.HC via the arXiv API.

Semantic Search:

OpenAI/Gemini: Uses cloud embeddings to find papers semantically similar to your brief.

Groq/Local: Uses local sentence-transformers (MiniLM) on your CPU to find relevant papers.

Venue Filtering: Filters candidates (e.g., "NeurIPS only") after the semantic search.

LLM Classification: The chosen LLM reads the abstracts and classifies papers as Primary (direct match), Secondary (partial match), or Off-topic.

Scoring Set: It builds a final list of ~20 papers to score, prioritizing Primary matches.

Moneyball Scoring: Instead of asking the LLM to guess citations, the agent fetches live Semantic Scholar data (Author Momentum, Citation Velocity) and combines it with an LLM analysis of "Market Fit" and "Novelty" to calculate a weighted Impact Score.

Ranking: Papers are ranked by this hybrid score.

Reporting: The UI displays metadata, PDF links, and generates a Narrative Analysis explains why the paper is influential (e.g. "Backed by a powerhouse lab").

💻 Usage Guide

Run Locally

This is a pure Python application. No Docker or complex databases required.

1. Clone the repository:

git clone [https://github.com/benevolentbandwidth/researchagent.git](https://github.com/benevolentbandwidth/researchagent.git)
cd researchagent

2. Create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

3. Install dependencies:

pip install -r requirements.txt
# Ensure you install groq if it wasn't automatically picked up
pip install groq

4. Run the app:

streamlit run app.py

Your browser will open automatically at http://localhost:8501.

🛠️ For Developers

app.py: The entire application logic resides here. It is a single-file Streamlit app designed for portability.

Modifying Prompt Logic: Look for the Moneyball scoring functions and classify_papers_with_llm in app.py to change how the AI judges papers.

Adding Providers: The code uses a LLMConfig dataclass. To add a new provider (e.g., Anthropic), add the client initialization in call_llm and the specific embedding logic in select_embedding_candidates.

📊 Comparison Table

| Option | Requirements | Best For |
|--------|-------------|----------|
| OpenAI | API Key | Highest quality summaries and narrative analysis. (GPT-4o, GPT-4.1) |
| Gemini | API Key (Google AI Studio) | Speed and large context windows. (Gemini 3 Pro, Flash 2.5) |
| Groq | API Key (Free) | Fastest inference with open-source models. (Llama 3.1, Mixtral) |
| Free Local | CPU | Privacy, offline use, and zero cost |
