"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import ApprovalDialog from "@/components/ApprovalDialog";
import MemoryPanel from "@/components/MemoryPanel";
import {
  type ApprovalRequest,
  type ProviderStatus,
  type SessionState,
  type StreamEvent,
  type ToolInfo,
  getProviders,
  getSession,
  getTools,
  resetSession,
  streamApproval,
  streamChat,
} from "@/lib/api";

const SESSION_ID = "web";

type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "error";
  text: string;
  servedBy?: string | null;
  streaming?: boolean;
};

type Activity = { id: string; label: string };

let counter = 0;
const nextId = () => `m${++counter}`;

/** Tool traffic is noise unless it is readable — name the operation plainly. */
function describe(event: Extract<StreamEvent, { type: "event" }>): string | null {
  const first = event.content.split("(")[0].split(" ->")[0];
  switch (event.event) {
    case "tool_call":
      return `calling ${first}`;
    case "tool_result":
      return `${first} returned`;
    case "pressure_warning":
      return "memory pressure — offloading to archival";
    case "eviction":
      return "evicted older messages from context";
    case "security":
      return `security: ${event.content.slice(0, 90)}`;
    default:
      return null;
  }
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [activity, setActivity] = useState<Activity[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [state, setState] = useState<SessionState | null>(null);
  const [providers, setProviders] = useState<ProviderStatus[]>([]);
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [pending, setPending] = useState<ApprovalRequest | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Fetch first, then apply — and drop the result if the component went away
  // mid-flight, so a slow API cannot write into an unmounted tree.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [session, providerList, toolList] = await Promise.all([
          getSession(SESSION_ID),
          getProviders(),
          getTools(),
        ]);
        if (cancelled) return;
        setState(session);
        setProviders(providerList.providers);
        setTools(toolList.tools);
        setPending(session.pending_approval);
      } catch {
        /* the API may not be up yet; the panels stay empty */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, activity, pending]);

  /** Drive one SSE stream to completion, applying every event to the UI. */
  const consume = useCallback(
    async (stream: AsyncGenerator<StreamEvent>) => {
      const assistantId = nextId();
      let opened = false;
      setActivity([]);

      for await (const event of stream) {
        switch (event.type) {
          case "event": {
            const label = describe(event);
            if (label) {
              setActivity((prev) => [...prev, { id: nextId(), label }]);
            }
            break;
          }
          case "token": {
            if (!opened) {
              opened = true;
              setMessages((prev) => [
                ...prev,
                { id: assistantId, role: "assistant", text: "", streaming: true },
              ]);
            }
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, text: m.text + event.text } : m,
              ),
            );
            break;
          }
          case "message": {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      text: event.text,
                      servedBy: event.served_by,
                      streaming: false,
                    }
                  : m,
              ),
            );
            break;
          }
          case "approval_required":
            setPending(event.request);
            break;
          case "state":
            setState(event as SessionState);
            setPending(event.pending_approval);
            break;
          case "error":
            setMessages((prev) => [
              ...prev,
              { id: nextId(), role: "error", text: event.message },
            ]);
            break;
          case "done":
            setActivity([]);
            break;
        }
      }
      void getProviders().then((p) => setProviders(p.providers)).catch(() => {});
    },
    [],
  );

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || busy || pending) return;
    setInput("");
    setMessages((prev) => [...prev, { id: nextId(), role: "user", text }]);
    setBusy(true);
    try {
      await consume(streamChat(SESSION_ID, text));
    } finally {
      setBusy(false);
    }
  }, [busy, consume, input, pending]);

  const decide = useCallback(
    async (approved: boolean) => {
      setPending(null);
      setBusy(true);
      try {
        await consume(streamApproval(SESSION_ID, approved));
      } finally {
        setBusy(false);
      }
    },
    [consume],
  );

  const reset = useCallback(async () => {
    const snapshot = await resetSession(SESSION_ID);
    setState(snapshot);
    setMessages([]);
    setActivity([]);
    setPending(null);
  }, []);

  return (
    <main className="flex h-screen flex-col md:flex-row">
      <MemoryPanel
        state={state}
        providers={providers}
        tools={tools}
        onReset={reset}
        busy={busy}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex-1 space-y-4 overflow-y-auto px-4 py-6 md:px-8">
          {messages.length === 0 && !busy && (
            <div className="mx-auto max-w-xl pt-16 text-center text-sm text-slate-500">
              <p className="mb-2 text-base text-slate-300">
                Tell it something about yourself.
              </p>
              <p>
                It writes durable facts into core memory, so it still knows them
                after a restart. Watch the sidebar change as it does.
              </p>
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={
                message.role === "user" ? "flex justify-end" : "flex justify-start"
              }
            >
              <div
                className={`max-w-[80ch] rounded-lg px-4 py-2.5 text-sm leading-relaxed ${
                  message.role === "user"
                    ? "bg-sky-500/15 text-slate-100"
                    : message.role === "error"
                      ? "border border-rose-500/40 bg-rose-500/10 text-rose-200"
                      : "bg-[var(--color-panel)] text-slate-200"
                }`}
              >
                <p className="whitespace-pre-wrap break-words">
                  {message.text}
                  {message.streaming && (
                    <span className="ml-0.5 inline-block animate-pulse">▍</span>
                  )}
                </p>
                {message.servedBy && (
                  <p className="mt-1.5 text-[10px] text-slate-500">
                    ⚡ served by {message.servedBy}
                  </p>
                )}
              </div>
            </div>
          ))}

          {activity.length > 0 && (
            <div className="flex justify-start">
              <ul className="space-y-1 rounded-lg bg-black/20 px-4 py-2.5 font-mono text-[11px] text-slate-500">
                {activity.map((item) => (
                  <li key={item.id}>· {item.label}</li>
                ))}
              </ul>
            </div>
          )}

          {pending && (
            <div className="max-w-[80ch]">
              <ApprovalDialog request={pending} onDecide={decide} busy={busy} />
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        <div className="border-t border-[var(--color-edge)] bg-[var(--color-panel)] px-4 py-3 md:px-8">
          <div className="flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void send();
                }
              }}
              rows={1}
              disabled={busy || !!pending}
              placeholder={
                pending
                  ? "Answer the approval request above to continue…"
                  : "Message MemAssist…"
              }
              className="max-h-40 flex-1 resize-y rounded-md border border-[var(--color-edge)] bg-black/30 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-sky-500/60 focus:outline-none disabled:opacity-50"
            />
            <button
              onClick={() => void send()}
              disabled={busy || !!pending || !input.trim()}
              className="rounded-md bg-sky-500 px-4 py-2 text-sm font-semibold text-sky-950 transition hover:bg-sky-400 disabled:opacity-40"
            >
              {busy ? "…" : "Send"}
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}
