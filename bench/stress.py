"""Stress tier — UNSCORED. Run with ``python -m bench --stress``.

The 115-point suite checks that each mechanic is correct. It says nothing about
what happens when you keep going: whether context stays bounded over a hundred
turns, whether retrieval still discriminates at fifty facts instead of four,
whether the router survives a bad afternoon on the free tier, whether something
buried in a long document is still findable much later.

Deliberately not scored. These measure *behaviour under load*, and load-shaped
numbers move with the machine, the model download, and the free tier's mood. A
number that drifts for reasons unrelated to the source would corrupt the one
guarantee the scored suite has — that a delta is attributable to a change. So
the stress tier reports, and a human reads it.

Findings live in docs/benchmarks.md.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from agent.token_budget import approx_tokens
from llm.router import ChatResult, ToolCall, Usage


# --- shared scaffolding ---------------------------------------------------
def _result(content=None, tool_calls=None, served_by="gemini", prompt_tokens=100):
    return ChatResult(
        served_by=served_by,
        model=f"{served_by}-model",
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=5),
    )


class ScriptedByTurn:
    """A router whose reply is a function of the turn, not a fixed queue.

    A 100-turn run cannot be scripted as a list without the list becoming the
    test. This decides per call, so the *loop* drives the length.
    """

    def __init__(self, plan, window=32_000):
        self.plan = plan
        self.window = window
        self.calls = 0
        self.turn = 0
        self.served = {}

    def chat(self, messages, tools=None, *, tool_choice="auto", lane="interactive"):
        self.calls += 1
        # The prompt the loop actually built — measured, not assumed, so the
        # pressure mechanic is exercised by real growth.
        tokens = approx_tokens(json.dumps(list(messages), default=str))
        res = self.plan(self.turn, self.calls, tokens)
        self.served[res.served_by] = self.served.get(res.served_by, 0) + 1
        return res

    def context_window(self, provider):
        return self.window

    def min_context_window(self):
        return self.window


def _send(text, cid="s1"):
    return ToolCall(cid, "send_message", json.dumps({"text": text}))


def _archive(text, cid="a1"):
    return ToolCall(
        cid,
        "archival_memory_insert",
        json.dumps({"content": text, "source": "stated", "request_heartbeat": True}),
    )


# --- S1: 100-turn session coherence --------------------------------------
FACTS_100 = [
    "The user's name is Ada Okonkwo.",
    "The user works as a paediatric nurse at Ruby Hall Clinic.",
    "The user's daughter Mira was born in March 2019.",
    "The user is allergic to penicillin.",
    "The user drives a red Toyota Corolla.",
    "The user's favourite meal is grilled salmon with lemon.",
    "The user's mortgage renews in November 2027.",
    "The user speaks Igbo, English and Marathi.",
    "The user's sister Nkiru lives in Lagos.",
    "The user runs on Tuesday and Thursday mornings.",
]


def s1_hundred_turn_coherence(tmp, make_memory, make_loop) -> list[str]:
    """Does a long session stay bounded, and do early facts survive to the end?"""
    mem = make_memory(tmp / "s1")

    def plan(turn, calls, tokens):
        # Turns 0-9 state a fact, which the agent archives then acknowledges.
        if turn < len(FACTS_100):
            if calls % 2 == 1:
                return _result(tool_calls=[_archive(FACTS_100[turn])], prompt_tokens=tokens)
            return _result(tool_calls=[_send("Noted.")], prompt_tokens=tokens)
        # Under pressure the agent offloads; otherwise it just replies.
        return _result(tool_calls=[_send("Mm-hmm, go on.")], prompt_tokens=tokens)

    router = ScriptedByTurn(plan)
    loop = make_loop(mem, router, planning_context_limit=8_000, pressure_threshold=0.7)

    peak_pct = 0.0
    evictions = 0
    queue_high_water = 0
    started = time.time()
    for turn in range(100):
        router.turn = turn
        loop.step(
            FACTS_100[turn] if turn < len(FACTS_100)
            else f"Turn {turn}: telling you a long, unremarkable story about my commute "
                 f"and the weather and the state of the roads, at some length."
        )
        pct = loop.last_input_tokens / max(loop.last_limit, 1)
        peak_pct = max(peak_pct, pct)
        queue_high_water = max(queue_high_water, len(loop.messages))
    elapsed = time.time() - started

    evictions = mem.store.search_messages("", page=0, page_size=1000, event_types=("eviction",))[1]

    # The point of the whole architecture: facts from turn 0-9 must still be
    # findable after 90 more turns have pushed them out of the window.
    found = 0
    for fact in FACTS_100:
        probe = fact.split(" is ")[-1].split(" at ")[-1].rstrip(".")
        items, _ = mem.archival.search(probe, top_k=3)
        if any(fact[:40] in item["content"] for item in items):
            found += 1

    return [
        f"  turns                 100 ({elapsed:.1f}s, {router.calls} model calls)",
        f"  in-context queue      {len(loop.messages)} messages (high water {queue_high_water})",
        f"  peak context usage    {peak_pct:.0%} of the {loop.last_limit}-token limit",
        f"  evictions fired       {evictions}",
        f"  recall log            {mem.memory_stats()['recall_events']} events",
        f"  early facts retrieved {found}/{len(FACTS_100)} after 90 intervening turns",
    ]


# --- S2: 50-fact retrieval precision -------------------------------------
# (fact, paraphrase probe sharing as few content words as possible)
FACTS_50 = [
    ("The user is allergic to penicillin.", "which medication makes them unwell"),
    ("The user drives a red Toyota Corolla.", "what vehicle do they own"),
    ("The user's daughter Mira was born in Pune.", "where was their child delivered"),
    ("The user's favourite meal is grilled salmon.", "what dish do they enjoy most"),
    ("The user works as a paediatric nurse.", "what is their profession"),
    ("The user's mortgage renews in November 2027.", "when does the home loan come up"),
    ("The user speaks Igbo and Marathi.", "which languages can they converse in"),
    ("The user's sister Nkiru lives in Lagos.", "where is their sibling based"),
    ("The user runs on Tuesday mornings.", "when do they exercise"),
    ("The user's laptop is a ThinkPad X1.", "what computer do they use"),
    ("The user dislikes herbal tea.", "which drink do they avoid"),
    ("The user's landlord is called Mr Deshmukh.", "who owns the property they rent"),
    ("The user studied at the University of Nsukka.", "where did they take their degree"),
    ("The user's father was a railway engineer.", "what did their dad do for a living"),
    ("The user broke their left wrist in 2015.", "what injury have they had"),
    ("The user's cat is named Biscuit.", "what is their pet called"),
    ("The user prefers window seats on flights.", "what do they choose when they travel"),
    ("The user's blood group is O negative.", "what type is their blood"),
    ("The user cannot swim.", "what physical skill do they lack"),
    ("The user's wedding anniversary is 12 June.", "when do they celebrate their marriage"),
    ("The user is saving for a trip to Patagonia.", "where do they want to go"),
    ("The user's manager is called Priya.", "who do they report to at work"),
    ("The user takes vitamin D in winter.", "what supplement do they use"),
    ("The user's first job was at a bakery.", "where did they start their career"),
    ("The user watches cricket, not football.", "which sport do they follow"),
    ("The user's flat is on the fourth floor.", "which storey do they live on"),
    ("The user grinds coffee beans by hand.", "how do they prepare their morning drink"),
    ("The user's bicycle was stolen last year.", "what possession did they lose"),
    ("The user is afraid of heights.", "what are they frightened of"),
    ("The user's grandmother taught them to sew.", "who showed them needlework"),
    ("The user plays the kora.", "which instrument do they perform on"),
    ("The user's passport expires in 2029.", "when does their travel document run out"),
    ("The user is lactose intolerant.", "which food group upsets their stomach"),
    ("The user's best friend is called Tunde.", "who are they closest to"),
    ("The user gardens on the balcony.", "what hobby do they do outdoors"),
    ("The user's car insurance is with Bajaj.", "who covers their vehicle"),
    ("The user reads before sleeping.", "what do they do at bedtime"),
    ("The user's nephew is starting school.", "which relative begins education"),
    ("The user donates blood twice a year.", "what charitable act do they repeat"),
    ("The user's phone is an old Pixel.", "what handset do they carry"),
    ("The user hates driving at night.", "what do they dislike about the dark"),
    ("The user's colleague retired in April.", "who left the workplace"),
    ("The user grew up near the sea.", "what was their childhood landscape"),
    ("The user's dentist appointment is overdue.", "which health check have they skipped"),
    ("The user bakes sourdough on weekends.", "what do they cook on days off"),
    ("The user's uncle owns a print shop.", "what business is in the family"),
    ("The user commutes by train.", "how do they get to work"),
    ("The user's favourite colour is ochre.", "which shade do they like best"),
    ("The user volunteers at a shelter.", "where do they give their time"),
    ("The user's alarm goes off at 5:30.", "when do they wake up"),
]


def s2_fifty_fact_precision(tmp, make_memory, _make_loop) -> list[str]:
    """Does semantic retrieval still discriminate at 50 facts, not 4?

    T5 scores four probes against a four-passage store, where almost anything
    ranks first. This is the same mechanic with twelve times the haystack.
    """
    mem = make_memory(tmp / "s2")
    started = time.time()
    for fact, _ in FACTS_50:
        mem.archival.insert(fact, source="stated")
    insert_s = time.time() - started

    top1 = top3 = 0
    misses = []
    started = time.time()
    for fact, probe in FACTS_50:
        items, _ = mem.archival.search(probe, top_k=3)
        contents = [i["content"] for i in items]
        if contents and contents[0] == fact:
            top1 += 1
        elif fact in contents:
            top3 += 1
            misses.append(f"{probe!r} -> top1 was {contents[0][:44]!r}")
        else:
            misses.append(f"MISS {probe!r} -> {contents[0][:44]!r}" if contents else f"MISS {probe!r}")
    query_s = time.time() - started

    n = len(FACTS_50)
    lines = [
        f"  passages              {mem.archival.count()} ({insert_s:.1f}s to embed and insert)",
        f"  precision@1           {top1}/{n} ({top1 / n:.0%})",
        f"  precision@3           {top1 + top3}/{n} ({(top1 + top3) / n:.0%})",
        f"  query latency         {query_s / n * 1000:.0f} ms/probe",
    ]
    lines += [f"    - {m}" for m in misses[:6]]
    if len(misses) > 6:
        lines.append(f"    - … and {len(misses) - 6} more")
    return lines


# --- S3: 20 messages under provider cooldowns ----------------------------
def s3_rapid_fire_under_cooldowns(tmp, make_memory, make_loop) -> list[str]:
    """Twenty turns while the chain is falling over. Does the user still get replies?"""
    from llm import errors

    mem = make_memory(tmp / "s3")
    state = {"gemini_dead_until": 6, "groq_dead_between": (3, 12)}
    exhausted_turns = []

    def plan(turn, calls, tokens):
        # Priority 1 is rate-limited for the first stretch; priority 2 dies in
        # the middle; the two of them overlap for turns 3-5, so the run has a
        # window where only the lower-priority lanes can serve.
        if turn < state["gemini_dead_until"]:
            lo, hi = state["groq_dead_between"]
            if lo <= turn < hi:
                served = "openrouter"
            else:
                served = "groq"
        else:
            served = "gemini"
        return _result(tool_calls=[_send("ok")], served_by=served, prompt_tokens=tokens)

    router = ScriptedByTurn(plan)
    loop = make_loop(mem, router)

    delivered = 0
    started = time.time()
    for turn in range(20):
        router.turn = turn
        try:
            out = loop.step(f"rapid message {turn}")
            if out:
                delivered += 1
        except errors.AllProvidersExhausted:
            exhausted_turns.append(turn)
    elapsed = time.time() - started

    order = sorted(router.served.items(), key=lambda kv: -kv[1])
    return [
        f"  turns                 20 in {elapsed:.1f}s ({elapsed / 20 * 1000:.0f} ms/turn)",
        f"  replies delivered     {delivered}/20",
        f"  served_by             {', '.join(f'{k}={v}' for k, v in order)}",
        f"  turns left unanswered {len(exhausted_turns)}"
        + (f" (turns {exhausted_turns})" if exhausted_turns else ""),
        f"  recall continuity     {mem.memory_stats()['recall_messages']} messages logged",
    ]


# --- S4: a 10-page document, recalled 20 turns later ---------------------
_NEEDLE = (
    "The building's emergency water shut-off valve is behind the third panel "
    "in the basement corridor, marked B-14."
)


def _ten_page_document() -> list[str]:
    """~10 pages of filler with one specific fact buried in the middle.

    Filler is varied rather than repeated: identical paragraphs would collapse
    to identical embeddings and make retrieval look better than it is.
    """
    topics = [
        "quarterly maintenance of the ventilation system",
        "the revised parking allocation for residents",
        "waste segregation and collection timings",
        "the lift service contract and its renewal terms",
        "landscaping around the eastern boundary wall",
        "fire drill procedures and assembly points",
        "visitor registration at the gate",
        "the society's annual audit and its findings",
        "roof waterproofing carried out before the monsoon",
        "the proposed solar installation on block C",
    ]
    pages = []
    for i, topic in enumerate(topics):
        body = (
            f"Section {i + 1}. This section concerns {topic}. "
            f"The committee reviewed {topic} and recorded that arrangements remain "
            f"broadly satisfactory, subject to the usual seasonal variation. Residents "
            f"raising questions about {topic} should address them to the secretary in "
            f"writing. Costs relating to {topic} are apportioned by floor area."
        )
        if i == 4:  # buried in the middle, not at an edge
            body += " " + _NEEDLE
        pages.append(body)
    return pages


def s4_long_document_delayed_recall(tmp, make_memory, make_loop) -> list[str]:
    """Ingest ten pages, talk about something else for twenty turns, then ask."""
    mem = make_memory(tmp / "s4")
    pages = _ten_page_document()
    words = sum(len(p.split()) for p in pages)
    for page in pages:
        mem.archival.insert(page, source="external")

    def plan(turn, calls, tokens):
        return _result(tool_calls=[_send("Right.")], prompt_tokens=tokens)

    router = ScriptedByTurn(plan)
    loop = make_loop(mem, router, planning_context_limit=8_000)
    for turn in range(20):
        router.turn = turn
        loop.step(f"Unrelated turn {turn}: chatting about lunch and the cricket score.")

    probes = [
        "where do I turn off the water in an emergency",
        "which panel hides the shut-off valve",
        "B-14",
    ]
    lines = [
        f"  document              {len(pages)} pages, ~{words} words, "
        f"{mem.archival.count()} passages",
        f"  intervening turns     20 (in-context queue now {len(loop.messages)})",
    ]
    for probe in probes:
        items, _ = mem.archival.search(probe, top_k=3)
        rank = next(
            (i + 1 for i, item in enumerate(items) if _NEEDLE[:40] in item["content"]), None
        )
        lines.append(
            f"  {probe[:38]:<40} " + (f"needle at rank {rank}" if rank else "NOT in top 3")
        )
    return lines


# --- runner ---------------------------------------------------------------
SCENARIOS = [
    ("S1", "100-turn session coherence", s1_hundred_turn_coherence),
    ("S2", "50-fact retrieval precision", s2_fifty_fact_precision),
    ("S3", "20 messages under provider cooldowns", s3_rapid_fire_under_cooldowns),
    ("S4", "10-page document, recalled 20 turns later", s4_long_document_delayed_recall),
]


def run_stress(tmp: Path, make_memory, make_loop) -> list[str]:
    lines = ["", "Stress tier (NOT scored — findings recorded in docs/benchmarks.md)", "-" * 62]
    for sid, title, fn in SCENARIOS:
        lines.append(f"\n{sid} {title}")
        try:
            lines.extend(fn(tmp, make_memory, make_loop))
        except Exception as exc:  # a stress failure is a finding, not a crash
            lines.append(f"  FAILED: {type(exc).__name__}: {str(exc)[:160]}")
    return lines
