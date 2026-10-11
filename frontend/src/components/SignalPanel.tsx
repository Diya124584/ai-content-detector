import type { DetectResponse } from "../types";
import Bar from "./Bar";

const fmt = (v: number | null | undefined, d = 2) => (v === null || v === undefined ? "n/a" : v.toFixed(d));

export default function SignalPanel({ r }: { r: DetectResponse }) {
  const s = r.signals;
  const sc = r.signal_scores;
  return (
    <section className="rounded-lg bg-white p-5 shadow">
      <h2 className="mb-1 text-lg font-semibold">Signal breakdown</h2>
      <p className="mb-4 text-xs text-slate-500">
        Bars show how strongly each signal points toward AI (0-100%). Each signal is only weak evidence.
        Missing signals (n/a) are excluded and the remaining weights are renormalised.
      </p>
      <div className="space-y-3">
        <Bar label="RoBERTa baseline P(AI) (GPT-2-era detector)" value={s.roberta ?? null} />
        <Bar
          label={`Perplexity (raw: ${fmt(s.perplexity, 1)})`}
          value={sc.perplexity ?? null}
        />
        <Bar label={`Burstiness (raw: ${fmt(s.burstiness)})`} value={null} right={fmt(s.burstiness)} />
        <Bar label={`TTR (raw: ${fmt(s.ttr)})`} value={null} right={fmt(s.ttr)} />
        <Bar label={`Repetition (raw: ${fmt(s.repetition)})`} value={null} right={fmt(s.repetition)} />
        <Bar label="Linguistic composite (burstiness + TTR + repetition)" value={sc.linguistic ?? null} />
        <Bar label="TF-IDF model P(AI)" value={s.feature_model ?? null} color="bg-emerald-500" />
      </div>
      <p className="mt-3 text-xs text-slate-500">
        Active: {r.components_active.join(", ") || "none"} · model {r.model_version}
      </p>
    </section>
  );
}
