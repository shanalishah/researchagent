"""Research Agent — FastAPI backend.

Serves the React frontend (when built) and the streaming search API. Uses the
real extracted pipeline when R2 credentials are configured (backend/.env),
otherwise falls back to the mock pipeline so the UI is always demoable.

Run (dev):  uvicorn backend.main:app --reload --port 8000
"""

import asyncio
import json
import pathlib
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .models import SearchRequest, StageEvent, DoneEvent, ErrorEvent, Stage
from .config import load_env, has_r2_credentials
from . import mockdata

load_env()
USE_REAL = has_r2_credentials()

app = FastAPI(title="Research Agent API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MOCK_STAGE_DELAY_S = 0.4
_SENTINEL = object()


def _sse(payload) -> str:
    if hasattr(payload, "model_dump_json"):
        return f"data: {payload.model_dump_json()}\n\n"
    return f"data: {json.dumps(payload)}\n\n"


@app.on_event("startup")
def _warm_corpus():
    """Kick off the corpus sync in the background so the first search is fast."""
    if USE_REAL:
        from .runner import ensure_corpus
        threading.Thread(target=ensure_corpus, daemon=True).start()


@app.get("/api/health")
def health():
    mode = "real" if USE_REAL else "mock"
    info = {"status": "ok", "mode": mode}
    if USE_REAL:
        from .runner import _corpus_ready
        info["corpus_ready"] = _corpus_ready
    return info


@app.get("/api/stages", response_model=list[Stage])
def stages():
    return mockdata.STAGES


@app.post("/api/search")
async def search(req: SearchRequest):
    if USE_REAL:
        from .runner import pipeline_events

        async def event_stream():
            gen = pipeline_events(req)
            try:
                while True:
                    # Advance the synchronous pipeline one stage at a time in a
                    # worker thread so heavy model inference never blocks the loop.
                    ev = await asyncio.to_thread(next, gen, _SENTINEL)
                    if ev is _SENTINEL:
                        break
                    yield _sse(ev)
            except Exception as exc:
                yield _sse({"type": "error", "message": str(exc)})

        return StreamingResponse(
            event_stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---- Mock fallback (no R2 creds) ----
    async def mock_stream():
        try:
            total = 0.0
            for index, stage in enumerate(mockdata.STAGES):
                yield _sse(StageEvent(index=index, status="start", name=stage.name, detail=stage.detail))
                await asyncio.sleep(MOCK_STAGE_DELAY_S)
                total += stage.seconds or 0.0
                yield _sse(StageEvent(index=index, status="done", name=stage.name, detail=stage.detail))
            papers = list(mockdata.MOCK_PAPERS)
            yield _sse(DoneEvent(
                papers=papers,
                primary_count=sum(1 for p in papers if p.focus == "primary"),
                secondary_count=sum(1 for p in papers if p.focus == "secondary"),
                total_seconds=round(total, 1),
                provider=req.provider,
            ))
        except Exception as exc:
            yield _sse(ErrorEvent(message=str(exc)))

    return StreamingResponse(
        mock_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---- Static frontend (production) ----
_DIST = pathlib.Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
else:
    @app.get("/")
    def _root_placeholder():
        return JSONResponse({"status": f"backend up ({'real' if USE_REAL else 'mock'})",
                             "frontend": "not built — run the Vite dev server"})
