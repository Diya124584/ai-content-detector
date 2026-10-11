import type { ExplanationItem } from "../types";

const style: Record<ExplanationItem["direction"], string> = {
  toward_ai: "bg-red-100 text-red-800",
  toward_human: "bg-green-100 text-green-800",
  neutral: "bg-slate-100 text-slate-700",
  caveat: "bg-amber-100 text-amber-800",
};
const text: Record<ExplanationItem["direction"], string> = {
  toward_ai: "toward AI",
  toward_human: "toward human",
  neutral: "inconclusive",
  caveat: "caveat",
};

export default function Explanation({ items }: { items: ExplanationItem[] }) {
  return (
    <section className="rounded-lg bg-white p-5 shadow">
      <h2 className="mb-3 text-lg font-semibold">Explanation</h2>
      <ul className="space-y-2">
        {items.map((e, i) => (
          <li key={i} className="flex items-start gap-2 text-sm">
            <span className={`mt-0.5 shrink-0 rounded px-2 py-0.5 text-xs font-medium ${style[e.direction]}`}>
              {text[e.direction]}
            </span>
            <span className="text-slate-700">{e.detail}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
