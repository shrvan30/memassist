/**
 * Typed client for the MemAssist API (see api/main.py).
 *
 * The chat endpoints stream Server-Sent Events. EventSource is not used because
 * it cannot issue a POST — the turn's input has to go in a body — so the stream
 * is read straight off the fetch response and parsed here.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type CoreBlocks = { persona: string; human: string };

export type Tiers = {
  recall_messages: number;
  recall_events: number;
  archival_passages: number;
  context_messages: number;
};

export type ContextUsage = {
  input_tokens: number;
  limit: number;
  usage: string;
  pct: number;
  under_pressure: boolean;
};

export type GatedAction = {
  tool_call_id: string;
  name: string;
  arguments: Record<string, unknown>;
};

export type ApprovalRequest = { kind: string; actions: GatedAction[] };

export type SessionState = {
  session_id: string;
  core: CoreBlocks;
  tiers: Tiers;
  context: ContextUsage;
  served_by: string | null;
  pending_approval: ApprovalRequest | null;
  archival_available: boolean;
};

export type ProviderStatus = {
  name: string;
  priority: number;
  model: string;
  available: boolean;
  reason: string;
  requests: number;
  tokens: number;
  cooldown_remaining: number;
};

export type ToolInfo = {
  name: string;
  server: string | null;
  trust: string;
  gated: boolean;
};

/** One decoded SSE frame. `type` mirrors the server's event name. */
export type StreamEvent =
  | { type: "start"; session_id: string; resumed?: boolean; approved?: boolean }
  | {
      type: "event";
      role: string;
      event: string;
      content: string;
      served_by: string | null;
    }
  | { type: "token"; text: string }
  | { type: "message"; text: string; served_by: string | null }
  | { type: "approval_required"; request: ApprovalRequest }
  | ({ type: "state" } & SessionState)
  | { type: "done" }
  | { type: "error"; message: string };

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status}: ${detail || response.statusText}`);
  }
  return (await response.json()) as T;
}

export const getSession = (id: string) => json<SessionState>(`/sessions/${id}`);
export const resetSession = (id: string) =>
  json<SessionState>(`/sessions/${id}/reset`, { method: "POST" });
export const getProviders = () =>
  json<{ providers: ProviderStatus[] }>("/providers");
export const getTools = () =>
  json<{ tools: ToolInfo[]; errors: Record<string, string> }>("/tools");

/**
 * POST to an SSE endpoint and yield decoded events as they arrive.
 *
 * Frames are separated by a blank line and may split across chunk boundaries,
 * so the tail of the buffer is kept until its terminator shows up.
 */
async function* streamSse(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = ((await response.json()) as { detail?: string }).detail ?? detail;
    } catch {
      /* non-JSON error body; the status text will do */
    }
    yield { type: "error", message: detail };
    return;
  }
  if (!response.body) {
    yield { type: "error", message: "The server returned an empty stream." };
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let split = buffer.indexOf("\n\n");
    while (split !== -1) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      const line = frame.split("\n").find((l) => l.startsWith("data: "));
      if (line) {
        try {
          yield JSON.parse(line.slice(6)) as StreamEvent;
        } catch {
          /* a malformed frame must not kill the whole turn */
        }
      }
      split = buffer.indexOf("\n\n");
    }
  }
}

export const streamChat = (
  sessionId: string,
  message: string,
  signal?: AbortSignal,
) => streamSse("/chat", { session_id: sessionId, message }, signal);

export const streamApproval = (
  sessionId: string,
  approved: boolean,
  signal?: AbortSignal,
) => streamSse(`/sessions/${sessionId}/approve`, { approved }, signal);
