import { useCallback, useEffect, useState } from "react";
import { detect, listHistory } from "./api";
import Bar from "./components/Bar";
import Explanation from "./components/Explanation";
import History from "./components/History";
import SignalPanel from "./components/SignalPanel";
import type { DetectResponse, HistoryList, Prediction } from "./types";

const badge: Record<Prediction, string> = {
  "AI-generated": "bg-red-600",
  "Human-written": "bg-green-600",
  Uncertain: "bg-amber-500",
};

export default function App() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DetectResponse | null>(null);
  const [history, setHistory] = useState<HistoryList>({ total: 0, items: [] });

  const refreshHistory = useCallback(async () => {
    try {
      setHistory(await listHistory(10, 0));
    } catch {
      /* history is optional; ignore */
    }
  }, []);

  useEffect(() => {
    void refreshHistory();
  }, [refreshHistory]);

  async function analyze() {
    setLoading(true);
    setError(null);
    try {
      setResult(await detect(text));
      void refreshHistory();
    } catch (e) {
      setResult(null);
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  return (
    <main className="mx-auto max-w-4xl space-y-5 p-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">AI CONTENT DETECTOR</h1>
        <p className="text-sm text-slate-500">Text module · research prototype</p>
      </header>

      <div className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
        This tool gives probabilistic signals, not proof of authorship. It can be wrong in both directions,
        especially on short, edited, paraphrased, non-English or unseen-model text. Do not use it as sole evidence.
      </div>

      <section className="rounded-lg bg-white p-5 shadow">
        <textarea
          className="h-48 w-full resize-y rounded border border-slate-300 p-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
          placeholder="Paste text here (at least 10 words; 50+ words gives more reliable signals)…"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <div className="mt-3 flex items-center justify-between">
          <span className="text-xs text-slate-500">{wordCount} words</span>
          <button
            onClick={analyze}
            disabled={loading || wordCount === 0}
            className="rounded bg-indigo-600 px-5 py-2 text-sm font-semibold text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Analyzing…" : "ANALYZE"}
          </button>
        </div>
        {error && <p className="mt-3 rounded bg-red-50 p-2 text-sm text-red-700">{error}</p>}
      </section>

      {result && (
        <>
          <section className="rounded-lg bg-white p-5 shadow">
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-sm text-slate-500">Prediction:</span>
              <span className={`rounded px-3 py-1 text-sm font-bold uppercase text-white ${badge[result.prediction]}`}>
                {result.prediction}
              </span>
            </div>
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <div className="space-y-3">
                <Bar label="AI probability" value={result.ai_probability} color="bg-red-500" />
                <Bar label="Human probability" value={result.human_probability} color="bg-green-500" />
                <p className="text-xs text-slate-500">
                  Detection score (fused, before any calibration):{" "}
                  <span className="font-mono">{result.detection_score.toFixed(2)}</span>. Not a calibrated probability
                  unless a calibrator is loaded.
                </p>
              </div>
              <div className="rounded border border-slate-200 p-3">
                <p className="text-sm text-slate-500">Trust score (reliability, not P(AI))</p>
                <p className="text-2xl font-bold">
                  <span className="text-base font-medium text-amber-700">
                    {result.trust_score.status.startsWith("prototype") ? "Prototype / " : ""}
                  </span>
                  {result.trust_score.value.toFixed(0)}
                </p>
                <p className="text-xs text-slate-500">{result.trust_score.status}</p>
                {result.trust_score.ood_flags.length > 0 && (
                  <ul className="mt-2 list-disc pl-4 text-xs text-amber-800">
                    {result.trust_score.ood_flags.map((f) => (
                      <li key={f}>{f}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
            {result.prediction === "Uncertain" && (
              <p className="mt-3 text-sm text-amber-700">
                The evidence is mixed, so the detector abstains rather than guessing.
              </p>
            )}
            {result.warnings.length > 0 && (
              <ul className="mt-3 list-disc pl-5 text-xs text-slate-500">
                {result.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            )}
          </section>
          <SignalPanel r={result} />
          <Explanation items={result.explanation} />
        </>
      )}

      <History items={history.items} total={history.total} />
      <footer className="pb-6 text-center text-xs text-slate-400">
        RoBERTa signal comes from a GPT-2-era baseline detector. The fusion weights are prototype priors, not fitted values.
      </footer>
    </main>
  );
}
