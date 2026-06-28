"""Mock pipeline output, mirroring the frontend prototype's sample data.

This lets us prove the full frontend<->backend contract (request shape, SSE
stage streaming, result rendering) before wiring in the real ML pipeline in
`pipeline.py`. Phase 3 replaces `run_mock_pipeline` with the real thing behind
the same API.
"""

from .models import PaperOut, Stage


STAGES = [
    Stage(n="1", name="BM25 Recall", detail="71,803 → 600 papers", seconds=0.8),
    Stage(n="2", name="Vector Rerank", detail="600 → 300 papers", seconds=1.2),
    Stage(n="3", name="CrossEncoder", detail="300 → 150 papers", seconds=3.1),
    Stage(n="4", name="LLM Classify", detail="12 primary, 28 secondary", seconds=4.5),
    Stage(n="5", name="Impact Score", detail="Moneyball citation engine", seconds=6.2),
    Stage(n="6", name="Summarize", detail="Plain English for top 5", seconds=3.8),
]


MOCK_PAPERS = [
    PaperOut(
        rank=1, arxiv_id="2506.01234",
        title="Adaptive Preference Optimization for Multi-Turn Conversational Recommenders",
        authors=["Y. Zhang", "L. Chen", "M. Wang"], venue="NeurIPS",
        abstract=("We propose APO, a framework that fine-tunes recommender models using "
                  "multi-turn dialogue feedback. Unlike prior RLHF approaches that treat "
                  "recommendations as single-shot predictions, APO captures how user "
                  "preferences evolve across conversation turns."),
        arxiv_url="https://arxiv.org/abs/2506.01234",
        pdf_url="https://arxiv.org/pdf/2506.01234",
        score=87, too_new=False, focus="primary", relevance=0.94,
        why=["High-influence authors with strong citation velocity",
             "Directly addresses dynamic user modeling in recommenders",
             "Novel multi-turn optimization — strong results on 3 benchmarks"],
    ),
    PaperOut(
        rank=2, arxiv_id="2506.02345",
        title="Scaling Laws for Collaborative Filtering: When More Data Hurts",
        authors=["A. Patel", "R. Gupta"], venue="ICML",
        abstract=("We empirically demonstrate that collaborative filtering models exhibit "
                  "non-monotonic scaling — performance degrades beyond a critical data "
                  "threshold due to preference noise amplification."),
        arxiv_url="https://arxiv.org/abs/2506.02345",
        pdf_url="https://arxiv.org/pdf/2506.02345",
        score=72, too_new=False, focus="primary", relevance=0.91,
        why=["Established researchers with prior traction",
             "Challenges a fundamental assumption about data scaling",
             "Provides both theoretical analysis and practical recommendations"],
    ),
    PaperOut(
        rank=3, arxiv_id="2506.03456",
        title="Graph-Augmented Cross-Domain Recommendation via Unified User Embeddings",
        authors=["S. Kim", "J. Park", "H. Lee", "T. Nakamura"], venue=None,
        abstract=("We present GACDR, a graph neural network approach that learns unified "
                  "user representations across multiple domains via cross-domain interaction "
                  "graphs and message passing."),
        arxiv_url="https://arxiv.org/abs/2506.03456",
        pdf_url="https://arxiv.org/pdf/2506.03456",
        score=-1, too_new=True, focus="primary", relevance=0.88,
        why=["Too new for impact score — ranked by relevance only",
             "Highly relevant: novel graph-based cross-domain recommendation approach"],
    ),
    PaperOut(
        rank=4, arxiv_id="2506.04567",
        title="Debiasing Sequential Recommendation with Counterfactual Augmentation",
        authors=["M. Rodriguez", "C. Li"], venue="AAAI",
        abstract=("Sequential recommenders amplify popularity bias from biased click "
                  "sequences. We propose counterfactual data augmentation that generates "
                  "debiased training sequences, improving fairness by 34%."),
        arxiv_url="https://arxiv.org/abs/2506.04567",
        pdf_url="https://arxiv.org/pdf/2506.04567",
        score=65, too_new=False, focus="primary", relevance=0.85,
        why=["Prior traction in fairness-aware recommendation",
             "Addresses well-known bias problem with practical method",
             "Strong fairness results while maintaining quality"],
    ),
    PaperOut(
        rank=5, arxiv_id="2506.05678",
        title="Efficient Retrieval-Augmented Recommendation with Sparse Mixture-of-Experts",
        authors=["D. Wu", "F. Zhao", "X. Huang"], venue=None,
        abstract=("RAR-SMoE combines retrieval-augmented generation with sparse "
                  "mixture-of-experts routing. The system retrieves relevant item "
                  "descriptions and routes them through specialized expert networks."),
        arxiv_url="https://arxiv.org/abs/2506.05678",
        pdf_url="https://arxiv.org/pdf/2506.05678",
        score=58, too_new=False, focus="secondary", relevance=0.79,
        why=["Emerging authors — relies on content merit",
             "Bridges RAG and recommendation, a trending intersection",
             "Sparse MoE routing reduces compute cost"],
    ),
]
