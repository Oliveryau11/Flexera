import React from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine } from "recharts";

export default function ReliabilityChart({ bins }) {
  if (!bins?.length) return null;
  const data = bins.map(b => ({
    x: ((Number(b.bin_low)+Number(b.bin_high))/2)*100,
    mean_prob: Number(b.mean_prob)*100,
    frac_pos: (b.frac_pos==="" || b.frac_pos==null) ? NaN : Number(b.frac_pos)*100,
    count: Number(b.count||0),
  }));
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="x" tickFormatter={(v)=>`${v.toFixed(0)}%`} />
          <YAxis domain={[0,100]} tickFormatter={(v)=>`${v}%`} />
          <Tooltip formatter={(v)=> Number.isFinite(v) ? `${Number(v).toFixed(1)}%` : "—"} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]} stroke="#94A3B8" />
          <Line type="monotone" dataKey="mean_prob" name="Mean p(Win)" stroke="#2563EB" dot />
          <Line type="monotone" dataKey="frac_pos" name="Observed Win%" stroke="#10B981" dot />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
