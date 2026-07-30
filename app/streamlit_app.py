"""Streamlit chat UI for MemAssist.

The sidebar shows live core memory, context usage vs. the active provider's
window, memory counts, and the free-tier provider chain (which provider is
available / cooling down). Each reply is badged with the provider that served
it.

Run with:  streamlit run app/streamlit_app.py   (or:  make dev)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the repo root importable regardless of how Streamlit is launched.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

import assembly  # noqa: E402
from agent.token_budget import usage_fraction  # noqa: E402

st.set_page_config(page_title="MemAssist", page_icon="🧠", layout="wide")

PROVIDER_ENVS = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


def any_key_present() -> bool:
    return any(os.getenv(env) for env in PROVIDER_ENVS.values())


def onboarding_gate() -> None:
    """Block until at least one provider key is available."""
    if any_key_present():
        return
    st.title("🧠 MemAssist")
    st.info(
        "No LLM provider keys found. Add at least one to a `.env` file (see "
        "`.env.example`) — Gemini + Groq is the recommended free-tier pair — or "
        "paste a Gemini key below to get started for this session."
    )
    key = st.text_input("GEMINI_API_KEY (free at aistudio.google.com/apikey)", type="password")
    if key:
        os.environ["GEMINI_API_KEY"] = key
        st.rerun()
    st.stop()


def get_loop():
    if "loop" not in st.session_state:
        st.session_state.loop = assembly.build_loop(
            memory=assembly.build_memory(session_id="streamlit"),
        )
        st.session_state.chat = []  # list of {role, text, served_by}
    return st.session_state.loop


def render_sidebar(loop) -> None:
    with st.sidebar:
        st.header("🧠 Memory")

        blocks = loop.memory.core_blocks()
        st.subheader("Core memory")
        st.caption("persona")
        st.code(blocks.get("persona", "").strip() or "(empty)", language=None)
        st.caption("human")
        st.code(blocks.get("human", "").strip() or "(empty)", language=None)

        st.subheader("Context usage")
        frac = min(1.0, usage_fraction(loop.last_input_tokens, loop.last_limit))
        st.progress(frac, text=loop.usage_string)
        if loop.under_pressure():
            st.warning("Memory pressure — the agent should offload to archival memory.")

        st.subheader("Memory tiers")
        stats = loop.memory.memory_stats()
        c1, c2 = st.columns(2)
        c1.metric("Recall messages", stats["recall_messages"])
        c2.metric("Archival passages", stats["archival_passages"])
        if loop.memory.archival is None:
            st.caption("⚠️ Archival memory unavailable (Chroma failed to init).")

        st.subheader("Providers")
        if loop.served_by:
            st.caption(f"Last reply served by **{loop.served_by}**")
        for s in loop.router.provider_status():
            mark = "✅" if s["available"] else "⛔"
            note = "" if s["available"] else f" — {s['reason']}"
            cooling = s["cooldown_remaining"]
            if cooling > 0:
                note += f" ({int(cooling)}s)"
            st.markdown(f"{mark} **{s['name']}** · {s['requests']} req{note}")

        st.divider()
        if st.button("Reset conversation", help="Clears the in-context window; keeps saved memory."):
            loop.messages.clear()
            loop.last_input_tokens = 0
            loop.served_by = None
            st.session_state.chat = []
            st.rerun()


def main() -> None:
    onboarding_gate()
    loop = get_loop()

    st.title("🧠 MemAssist")
    st.caption(
        "A MemGPT-style assistant that edits its own memory, on free-tier LLMs "
        "behind a failover router. Tell it about yourself, then restart — it remembers."
    )
    render_sidebar(loop)

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            if msg["role"] == "assistant" and msg.get("served_by"):
                st.caption(f"⚡ served by {msg['served_by']}")

    prompt = st.chat_input("Message MemAssist…")
    if not prompt:
        return

    st.session_state.chat.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking & managing memory…"):
            try:
                outputs = loop.step(prompt)
            except Exception as exc:  # surface API/router errors without crashing
                st.error(f"Something went wrong: {exc}")
                return
        served_by = loop.served_by
        if not outputs:
            outputs = ["(The assistant finished without sending a message.)"]
        for text in outputs:
            st.markdown(text)
            if served_by:
                st.caption(f"⚡ served by {served_by}")
            st.session_state.chat.append(
                {"role": "assistant", "text": text, "served_by": served_by}
            )

    st.rerun()


main()
