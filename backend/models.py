"""Pydantic request/response schemas for the Research Agent API.

These are the wire contract between the React frontend and the FastAPI
backend. The field names are chosen to map cleanly onto the frontend's
existing data shapes (see frontend mockup `PAPERS` / `STEPS`).
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


Provider = Literal["openai", "gemini", "groq", "free"]
DateRange = Literal["Last 3 Days", "Last Week", "Last Month", "All Time"]


class SearchRequest(BaseModel):
    """A single search submitted from the home screen."""

    query: str = Field(..., description="The user's research brief / what they're looking for")
    exclude: str = Field("", description="Topics to exclude (the 'NOT looking for' box)")
    date_range: DateRange = Field("Last Month", description="Recency window")
    provider: Provider = Field("free", description="Reasoning provider")
    top_n: int = Field(5, ge=1, le=10, description="How many top papers to highlight")
    api_key: Optional[str] = Field(None, description="Provider API key (in-memory only, never persisted)")
    model: Optional[str] = Field(None, description="Chat model id for the chosen provider")


class PaperOut(BaseModel):
    """One ranked paper as rendered by a result card."""

    rank: int
    arxiv_id: str
    title: str
    authors: List[str]
    venue: Optional[str] = None
    abstract: str
    arxiv_url: str
    pdf_url: Optional[str] = None
    # score == -1 means "too new to rate" (mirrors the Streamlit app's sentinel)
    score: float
    too_new: bool
    focus: Literal["primary", "secondary", "off-topic"]
    relevance: float  # 0..1
    why: List[str] = Field(default_factory=list)
    summary: Optional[str] = None  # plain-English summary (LLM providers only)


class Stage(BaseModel):
    """A pipeline stage, used both for the loader and the 'how it works' panel."""

    n: str
    name: str
    detail: str = ""
    seconds: Optional[float] = None


# ---- Server-Sent Event payloads (serialized to JSON in the `data:` line) ----

class StageEvent(BaseModel):
    type: Literal["stage"] = "stage"
    index: int            # 0-based stage index
    status: Literal["start", "done"]
    name: str
    detail: str = ""


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    papers: List[PaperOut]
    primary_count: int
    secondary_count: int
    total_seconds: float
    provider: Provider


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
