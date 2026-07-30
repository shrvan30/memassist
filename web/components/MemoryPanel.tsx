"use client";

/**
 * The memory inspector — parity with the Streamlit sidebar, plus the tool panel.
 *
 * Everything here is a live view of the agent's own state: the core blocks it
 * rewrites, how full its context is, and which provider answered last.
 */

import type {
  ProviderStatus,
  SessionState,
  ToolInfo,
} from "@/lib/api";

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="border-b border-[var(--color-edge)] px-4 py-4 last:border-b-0">
      <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
        {title}
      </h2>
      {children}
    </section>
  );
}

function Block({ label, text }: { label: string; text: string }) {
  return (
    <div className="mb-3 last:mb-0">
      <div className="mb-1 text-[11px] text-slate-500">{label}</div>
      <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words rounded-md bg-black/30 p-2 font-mono text-[11px] leading-relaxed text-slate-300">
        {text.trim() || "(empty)"}
      </pre>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-md bg-black/20 px-2 py-1.5">
      <div className="text-lg font-semibold text-slate-100">{value}</div>
      <div className="text-[10px] uppercase tracking-wide text-slate-500">
        {label}
      </div>
    </div>
  );
}

export default function MemoryPanel({
  state,
  providers,
  tools,
  onReset,
  busy,
}: {
  state: SessionState | null;
  providers: ProviderStatus[];
  tools: ToolInfo[];
  onReset: () => void;
  busy: boolean;
}) {
  const pct = Math.min(1, state?.context.pct ?? 0);
  const pressure = state?.context.under_pressure ?? false;

  return (
    <aside className="flex h-full w-full flex-col overflow-y-auto border-r border-[var(--color-edge)] bg-[var(--color-panel)] md:w-96">
      <div className="px-4 py-4">
        <h1 className="text-lg font-semibold text-slate-100">🧠 MemAssist</h1>
        <p className="mt-1 text-[11px] leading-snug text-slate-500">
          Edits its own memory with tool calls, on free-tier LLMs behind a
          failover router.
        </p>
      </div>

      <Section title="Core memory">
        <Block label="persona" text={state?.core.persona ?? ""} />
        <Block label="human" text={state?.core.human ?? ""} />
      </Section>

      <Section title="Context usage">
        <div className="mb-1.5 h-2 w-full overflow-hidden rounded-full bg-black/40">
          <div
            className={`h-full rounded-full transition-[width] duration-500 ${
              pressure ? "bg-amber-400" : "bg-emerald-400"
            }`}
            style={{ width: `${Math.round(pct * 100)}%` }}
          />
        </div>
        <div className="text-[11px] text-slate-400">
          {state?.context.usage ?? "0 / 0 tokens (0%)"}
        </div>
        {pressure && (
          <p className="mt-2 rounded-md bg-amber-500/10 px-2 py-1.5 text-[11px] text-amber-300">
            Memory pressure — the agent should offload to archival memory.
          </p>
        )}
      </Section>

      <Section title="Memory tiers">
        <div className="grid grid-cols-3 gap-2">
          <Metric label="Recall" value={state?.tiers.recall_messages ?? 0} />
          <Metric label="Archival" value={state?.tiers.archival_passages ?? 0} />
          <Metric label="In context" value={state?.tiers.context_messages ?? 0} />
        </div>
        {state && !state.archival_available && (
          <p className="mt-2 text-[11px] text-amber-400">
            ⚠️ Archival memory unavailable.
          </p>
        )}
      </Section>

      <Section title="Providers">
        {state?.served_by && (
          <p className="mb-2 text-[11px] text-slate-400">
            Last reply served by{" "}
            <span className="font-semibold text-slate-200">
              {state.served_by}
            </span>
          </p>
        )}
        <ul className="space-y-1">
          {providers.map((p) => (
            <li key={p.name} className="flex items-baseline gap-2 text-[11px]">
              <span>{p.available ? "✅" : "⛔"}</span>
              <span className="font-medium text-slate-200">{p.name}</span>
              <span className="text-slate-500">{p.requests} req</span>
              {!p.available && (
                <span className="truncate text-slate-500">— {p.reason}</span>
              )}
            </li>
          ))}
          {providers.length === 0 && (
            <li className="text-[11px] text-slate-500">No providers loaded.</li>
          )}
        </ul>
      </Section>

      {tools.length > 0 && (
        <Section title="External tools">
          <ul className="space-y-1">
            {tools.map((t) => (
              <li key={t.name} className="flex items-baseline gap-2 text-[11px]">
                <span
                  className={
                    t.trust === "untrusted" ? "text-amber-400" : "text-emerald-400"
                  }
                  title={`trust: ${t.trust}`}
                >
                  ●
                </span>
                <span className="font-mono text-slate-300">{t.name}</span>
                {t.gated && (
                  <span
                    className="rounded bg-rose-500/15 px-1 text-[10px] text-rose-300"
                    title="Requires your approval before it runs"
                  >
                    gated
                  </span>
                )}
              </li>
            ))}
          </ul>
        </Section>
      )}

      <div className="mt-auto px-4 py-4">
        <button
          onClick={onReset}
          disabled={busy}
          title="Clears the in-context window; saved memory is untouched."
          className="w-full rounded-md border border-[var(--color-edge)] px-3 py-2 text-xs text-slate-300 transition hover:bg-white/5 disabled:opacity-40"
        >
          Reset conversation
        </button>
      </div>
    </aside>
  );
}
