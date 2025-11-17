import React from "react";

export default function MetricsStrip({ m }) {
  if (!m) return null;
  const Cell = ({ label, value }) => (
    <div className="px-3 py-2 rounded-xl bg-slate-50 border border-slate-200">
      <div className="text-[11px] text-slate-500">{label}</div>
      <div className="text-sm font-semibold text-slate-900">{value}</div>
    </div>
  );
  const fmt = (x, pct=false) => (x==null || x==="") ? "—" : (pct ? `${(Number(x)*100).toFixed(1)}%` : Number(x).toFixed(3));

  return (
    <div className="grid grid-cols-2 sm:grid-cols-6 gap-3">
      <Cell label="Threshold" value={Number(m.threshold).toFixed(2)} />
      <Cell label="Precision" value={fmt(m.precision, true)} />
      <Cell label="Recall" value={fmt(m.recall, true)} />
      <Cell label="F1" value={fmt(m.f1)} />
      <Cell label="Brier" value={fmt(m.brier)} />
      <Cell label="Support (1/0)" value={`${m.support_pos ?? "—"}/${m.support_neg ?? "—"}`} />
    </div>
  );
}
