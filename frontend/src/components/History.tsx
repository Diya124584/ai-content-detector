import type { HistoryRecord } from "../types";

export default function History({ items, total }: { items: HistoryRecord[]; total: number }) {
  return (
    <section className="rounded-lg bg-white p-5 shadow">
      <h2 className="mb-1 text-lg font-semibold">Detection history</h2>
      <p className="mb-3 text-xs text-slate-500">
        Only a hash of each text is stored, not the text itself ({total} total).
      </p>
      {items.length === 0 ? (
        <p className="text-sm text-slate-500">No detections yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-slate-500">
              <tr>
                <th className="py-1 pr-4">When</th>
                <th className="pr-4">Prediction</th>
                <th className="pr-4">AI prob.</th>
                <th className="pr-4">Trust (prototype)</th>
                <th>Hash</th>
              </tr>
            </thead>
            <tbody>
              {items.map((h) => (
                <tr key={h.id} className="border-t border-slate-100">
                  <td className="py-1 pr-4 whitespace-nowrap">{new Date(h.created_at).toLocaleString()}</td>
                  <td className="pr-4">{h.prediction}</td>
                  <td className="pr-4 tabular-nums">{(h.ai_probability * 100).toFixed(0)}%</td>
                  <td className="pr-4 tabular-nums">{h.trust_score === null ? "n/a" : h.trust_score.toFixed(0)}</td>
                  <td className="font-mono text-xs text-slate-400">{h.text_hash.slice(0, 10)}…</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
