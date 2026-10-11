interface Props {
  label: string;
  value: number | null; // 0..1
  color?: string;
  right?: string;
}

export default function Bar({ label, value, color = "bg-indigo-500", right }: Props) {
  const pct = value === null ? 0 : Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <div>
      <div className="flex justify-between text-sm text-slate-600">
        <span>{label}</span>
        <span className="tabular-nums">{value === null ? "n/a" : right ?? `${pct}%`}</span>
      </div>
      <div className="mt-1 h-2 w-full rounded bg-slate-200">
        <div className={`h-2 rounded ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
