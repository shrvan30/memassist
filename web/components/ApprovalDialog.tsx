"use client";

/**
 * Approve/deny for a suspended turn (spec §6.3).
 *
 * The graph is paused inside `security_gate` — nothing has run. Answering
 * resumes it, so this is a real gate rather than a notification after the fact.
 */

import type { ApprovalRequest } from "@/lib/api";

export default function ApprovalDialog({
  request,
  onDecide,
  busy,
}: {
  request: ApprovalRequest;
  onDecide: (approved: boolean) => void;
  busy: boolean;
}) {
  return (
    <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 p-4">
      <h3 className="mb-1 text-sm font-semibold text-amber-300">
        ⛔ Approval required
      </h3>
      <p className="mb-3 text-xs text-slate-400">
        The assistant wants to perform a gated action. Filesystem writes are
        confined to <code className="text-slate-300">./workspace</code> and always
        require approval. Deny if you did not ask for this — a web page or a file
        can try to talk the assistant into it.
      </p>

      {request.actions.map((action) => (
        <div
          key={action.tool_call_id}
          className="mb-3 rounded-md bg-black/30 p-3"
        >
          <div className="mb-1.5 font-mono text-xs font-semibold text-slate-100">
            {action.name}
          </div>
          <pre className="overflow-x-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-slate-400">
            {JSON.stringify(action.arguments, null, 2)}
          </pre>
        </div>
      ))}

      <div className="flex gap-2">
        <button
          onClick={() => onDecide(true)}
          disabled={busy}
          className="flex-1 rounded-md bg-emerald-500/90 px-3 py-2 text-xs font-semibold text-emerald-950 transition hover:bg-emerald-400 disabled:opacity-40"
        >
          ✅ Approve
        </button>
        <button
          onClick={() => onDecide(false)}
          disabled={busy}
          className="flex-1 rounded-md bg-rose-500/90 px-3 py-2 text-xs font-semibold text-rose-950 transition hover:bg-rose-400 disabled:opacity-40"
        >
          🚫 Deny
        </button>
      </div>
    </div>
  );
}
