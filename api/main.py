"""FastAPI service for MemAssist (spec §11 P4).

    uvicorn api.main:app --reload

Endpoints
    POST   /chat                          SSE: run a turn
    POST   /sessions/{sid}/approve        SSE: resume a suspended turn
    GET    /sessions/{sid}                memory-inspector snapshot
    GET    /sessions/{sid}/pending        the gated action awaiting a human
    POST   /sessions/{sid}/reset          clear the in-context window
    GET    /providers                     failover-chain status
    GET    /healthz                       liveness for docker-compose

**On "token streaming".** The user-facing reply is the argument of a
``send_message`` TOOL CALL, not the model's content field — content is internal
monologue the architecture says the user must never see (spec §2). Streaming
provider tokens would therefore stream the wrong text. What streams instead is
the turn itself: every tool call, tool result, eviction and security decision
arrives live as it happens (sourced from the memory audit stream), and the reply
is chunked into ``token`` events so the UI still types out. That keeps the
4-provider flat tool-calling contract, which is load-bearing for the whole
project, completely untouched.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
from api.sessions import DEFAULT_SESSION, Session, registry, run_turn

_log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Start the shared MCP subprocesses and router once per process."""
    registry.startup()
    yield
    registry.shutdown()


app = FastAPI(title="MemAssist API", version="0.4.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- request models -------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = DEFAULT_SESSION


class ApprovalRequest(BaseModel):
    approved: bool


# --- helpers --------------------------------------------------------------
def sse(events: Iterator[dict]) -> StreamingResponse:
    def encode() -> Iterator[str]:
        for event in events:
            yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(
        encode(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx would otherwise buffer the stream
        },
    )


def session_or_404(session_id: str) -> Session:
    if session_id not in registry.ids():
        raise HTTPException(status_code=404, detail=f"Unknown session '{session_id}'.")
    return registry.get(session_id)


# --- chat -----------------------------------------------------------------
@app.post("/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    session = registry.get(request.session_id)
    # One turn at a time per session: concurrent turns would interleave writes
    # into the same checkpointer thread.
    if not session.lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="This session already has a turn running.")
    if session.loop.pending_approval:
        session.lock.release()
        raise HTTPException(
            status_code=409,
            detail="This session is waiting on an approval. Answer it before sending more.",
        )

    def events() -> Iterator[dict]:
        try:
            yield {"type": "start", "session_id": session.session_id}
            yield from run_turn(session, lambda: session.loop.step(request.message))
        finally:
            session.lock.release()

    return sse(events())


# --- the interrupt surface ------------------------------------------------
@app.get("/sessions/{session_id}/pending")
def pending(session_id: str) -> dict:
    session = session_or_404(session_id)
    return {"session_id": session_id, "pending_approval": session.loop.pending_approval}


@app.post("/sessions/{session_id}/approve")
def approve(session_id: str, request: ApprovalRequest) -> StreamingResponse:
    session = session_or_404(session_id)
    if not session.loop.pending_approval:
        raise HTTPException(status_code=409, detail="Nothing is awaiting approval.")
    if not session.lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="This session already has a turn running.")

    def events() -> Iterator[dict]:
        try:
            yield {
                "type": "start",
                "session_id": session.session_id,
                "resumed": True,
                "approved": request.approved,
            }
            yield from run_turn(session, lambda: session.loop.resume(approved=request.approved))
        finally:
            session.lock.release()

    return sse(events())


# --- memory inspector -----------------------------------------------------
@app.get("/sessions/{session_id}")
def session_state(session_id: str) -> dict:
    return registry.get(session_id).snapshot()


@app.get("/sessions/{session_id}/messages")
def session_messages(session_id: str, limit: int = 50) -> dict:
    """The in-context FIFO — what the model is actually looking at right now."""
    session = session_or_404(session_id)
    messages = session.loop.messages[-limit:]
    return {
        "session_id": session_id,
        "count": len(session.loop.messages),
        "messages": [
            {
                "role": m.get("role"),
                "content": m.get("content"),
                "tool_calls": [
                    tc.get("function", {}).get("name") for tc in m.get("tool_calls", []) or []
                ],
            }
            for m in messages
        ],
    }


@app.post("/sessions/{session_id}/reset")
def reset(session_id: str) -> dict:
    """Clear the in-context window. Saved memory is deliberately untouched."""
    session = registry.get(session_id)
    session.loop.reset()
    return session.snapshot()


@app.get("/sessions")
def list_sessions() -> dict:
    return {"sessions": registry.ids()}


# --- provider panel -------------------------------------------------------
@app.get("/providers")
def providers() -> dict:
    router = registry._router or registry.get(DEFAULT_SESSION).loop.router
    return {"providers": router.provider_status()}


@app.get("/tools")
def tools() -> dict:
    """Which external MCP tools loaded, and their trust zone (spec §6.1)."""
    external = registry._external
    if external is None:
        return {"tools": [], "errors": {}}
    return {
        "tools": [
            {
                "name": name,
                "server": external.server_of(name),
                "trust": external.trust_of(name),
                "gated": external.is_gated(name),
            }
            for name in sorted(external.names())
        ],
        "errors": external.errors,
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "backend": config.STORAGE_BACKEND}
