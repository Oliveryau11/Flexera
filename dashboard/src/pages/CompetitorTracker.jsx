import { useEffect, useMemo, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";

async function fetchCSVRows(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) return [];
  const text = await res.text();
  const lines = text.trim().split(/\r?\n/);
  if (lines.length <= 1) return [];
  const headers = lines[0].split(",").map(h => h.trim());
  return lines.slice(1).map((line) => {
    const cols = [];
    let current = "";
    let inQuotes = false;
    for (const char of line) {
      if (char === '"') inQuotes = !inQuotes;
      else if (char === ',' && !inQuotes) { cols.push(current.trim()); current = ""; }
      else current += char;
    }
    cols.push(current.trim());
    const row = {};
    headers.forEach((h, i) => (row[h] = cols[i] ?? ""));
    return row;
  });
}

function toNum(v) {
  if (v === "" || v == null) return NaN;
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

export default function CompetitorTracker() {
  const [deals, setDeals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("All");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const csvRows = await fetchCSVRows("/api/win_probabilities.csv");
        const seen = new Set();
        const transformed = [];
        for (let i = 0; i < csvRows.length; i++) {
          const row = csvRows[i];
          const id = row["Opportunity ID"];
          if (id && !seen.has(id)) {
            seen.add(id);
            const name = row["Opportunity Name"] || "";
            const stage = row["Stage"] || "";
            const owner = row["Owner"] || "Unassigned";
            const pWin = toNum(row["win_prob"]);
            const isWon = /closed won/i.test(stage);
            const isLost = /closed lost/i.test(stage);
            
            let oppType = "Other";
            if (/renewal/i.test(name)) oppType = "Renewal";
            else if (/new|expansion/i.test(name)) oppType = "New Business";
            
            let confidence = "Low";
            if (Number.isFinite(pWin)) {
              if (pWin >= 0.9) confidence = "High";
              else if (pWin >= 0.7) confidence = "Medium";
            }
            
            transformed.push({
              id, name, stage, owner, oppType, confidence,
              p_win: Number.isFinite(pWin) ? pWin : null,
              status: isWon ? "Won" : (isLost ? "Lost" : "Open"),
            });
          }
        }
        if (!cancelled) { setDeals(transformed); setLoading(false); }
      } catch (e) {
        if (!cancelled) { setDeals([]); setLoading(false); }
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const filtered = useMemo(() => {
    if (filter === "All") return deals;
    return deals.filter(d => d.confidence === filter);
  }, [deals, filter]);

  const byOwner = useMemo(() => {
    const g = {};
    for (const d of filtered) {
      if (!d.owner || d.owner === "Unassigned") continue;
      g[d.owner] ??= { owner: d.owner, total: 0, won: 0, high: 0, totalP: 0 };
      g[d.owner].total++;
      if (d.status === "Won") g[d.owner].won++;
      if (d.confidence === "High") g[d.owner].high++;
      if (d.p_win !== null) g[d.owner].totalP += d.p_win;
    }
    return Object.values(g)
      .map(o => ({ ...o, avgP: o.total > 0 ? (o.totalP / o.total) * 100 : 0, winRate: o.total > 0 ? (o.won / o.total) * 100 : 0 }))
      .sort((a, b) => b.total - a.total)
      .slice(0, 15);
  }, [filtered]);

  const byType = useMemo(() => {
    const g = {};
    for (const d of filtered) {
      g[d.oppType] ??= { type: d.oppType, total: 0, won: 0 };
      g[d.oppType].total++;
      if (d.status === "Won") g[d.oppType].won++;
    }
    return Object.values(g).map(t => ({ ...t, winRate: t.total > 0 ? (t.won / t.total) * 100 : 0 }));
  }, [filtered]);

  const byConfidence = useMemo(() => {
    const g = { High: 0, Medium: 0, Low: 0 };
    for (const d of filtered) if (d.p_win !== null) g[d.confidence]++;
    return Object.entries(g).map(([name, value]) => ({ name, value }));
  }, [filtered]);

  const avgProbability = useMemo(() => {
    const withProb = filtered.filter(d => d.p_win !== null);
    return withProb.length === 0 ? 0 : withProb.reduce((s, d) => s + d.p_win, 0) / withProb.length;
  }, [filtered]);

  const winRate = useMemo(() => {
    const closed = filtered.filter(d => d.status === "Won" || d.status === "Lost");
    return closed.length === 0 ? 0 : closed.filter(d => d.status === "Won").length / closed.length;
  }, [filtered]);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0f0f0f] flex items-center justify-center">
        <div className="w-6 h-6 border-2 border-amber-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0f0f0f] text-neutral-100">
      <div className="max-w-[1400px] mx-auto px-8 py-8">
        <div className="flex items-end justify-between mb-8">
          <div>
            <div className="text-xs text-neutral-500 uppercase tracking-[0.2em] mb-1">Performance</div>
            <h1 className="text-2xl font-light">Team Analytics</h1>
          </div>
          <select value={filter} onChange={(e) => setFilter(e.target.value)}
            className="bg-transparent border border-neutral-700 px-4 py-2 text-sm focus:outline-none">
            <option value="All" className="bg-neutral-900">All confidence</option>
            <option value="High" className="bg-neutral-900">High</option>
            <option value="Medium" className="bg-neutral-900">Medium</option>
            <option value="Low" className="bg-neutral-900">Low</option>
          </select>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-4 gap-6 mb-8">
          <div className="bg-neutral-900/50 border border-neutral-800 p-5">
            <div className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Total Deals</div>
            <div className="text-3xl font-light">{filtered.length.toLocaleString()}</div>
          </div>
          <div className="bg-neutral-900/50 border border-neutral-800 p-5">
            <div className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Avg Probability</div>
            <div className="text-3xl font-light text-amber-500">{(avgProbability * 100).toFixed(1)}%</div>
          </div>
          <div className="bg-neutral-900/50 border border-neutral-800 p-5">
            <div className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Win Rate</div>
            <div className="text-3xl font-light text-emerald-500">{(winRate * 100).toFixed(1)}%</div>
          </div>
          <div className="bg-neutral-900/50 border border-neutral-800 p-5">
            <div className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Top Performer</div>
            <div className="text-xl font-light">{byOwner[0]?.owner?.split(" ")[0] || "—"}</div>
            <div className="text-xs text-neutral-600 mt-1">{byOwner[0]?.total || 0} deals</div>
          </div>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-2 gap-6 mb-8">
          <div className="bg-neutral-900/30 border border-neutral-800 p-5">
            <div className="text-xs text-neutral-500 uppercase tracking-wider mb-4">By Opportunity Type</div>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={byType}>
                  <CartesianGrid stroke="#262626" strokeDasharray="3 3" />
                  <XAxis dataKey="type" tick={{ fill: '#737373', fontSize: 11 }} axisLine={{ stroke: '#404040' }} />
                  <YAxis tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#404040' }} />
                  <Tooltip contentStyle={{ background: '#171717', border: '1px solid #404040', borderRadius: 0 }} />
                  <Bar dataKey="total" fill="#525252" name="Total" />
                  <Bar dataKey="won" fill="#10b981" name="Won" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-neutral-900/30 border border-neutral-800 p-5">
            <div className="text-xs text-neutral-500 uppercase tracking-wider mb-4">Confidence Distribution</div>
            <div className="h-56 flex items-center">
              <ResponsiveContainer width="60%" height="100%">
                <PieChart>
                  <Pie data={byConfidence} cx="50%" cy="50%" innerRadius={40} outerRadius={70} dataKey="value">
                    <Cell fill="#10b981" />
                    <Cell fill="#f59e0b" />
                    <Cell fill="#ef4444" />
                  </Pie>
                  <Tooltip contentStyle={{ background: '#171717', border: '1px solid #404040', borderRadius: 0 }} />
                </PieChart>
              </ResponsiveContainer>
              <div className="text-xs space-y-2">
                <div className="flex items-center gap-2"><div className="w-2 h-2 bg-emerald-500" /> High</div>
                <div className="flex items-center gap-2"><div className="w-2 h-2 bg-amber-500" /> Medium</div>
                <div className="flex items-center gap-2"><div className="w-2 h-2 bg-red-500" /> Low</div>
              </div>
            </div>
          </div>
        </div>

        {/* Leaderboard */}
        <div className="bg-neutral-900/30 border border-neutral-800">
          <div className="px-5 py-4 border-b border-neutral-800">
            <div className="text-xs text-neutral-500 uppercase tracking-wider">Sales Leaderboard</div>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-neutral-500 text-xs uppercase tracking-wider border-b border-neutral-800">
                <th className="px-5 py-3 font-medium w-8">#</th>
                <th className="px-5 py-3 font-medium">Account Executive</th>
                <th className="px-5 py-3 font-medium text-right">Total</th>
                <th className="px-5 py-3 font-medium text-right">Won</th>
                <th className="px-5 py-3 font-medium text-right">High Conf</th>
                <th className="px-5 py-3 font-medium text-right">Avg p(Win)</th>
                <th className="px-5 py-3 font-medium text-right">Win Rate</th>
              </tr>
            </thead>
            <tbody>
              {byOwner.map((o, i) => (
                <tr key={o.owner} className={`border-b border-neutral-800/50 hover:bg-neutral-800/30 ${i % 2 === 0 ? 'bg-neutral-900/20' : ''}`}>
                  <td className="px-5 py-3 text-neutral-600">{i + 1}</td>
                  <td className="px-5 py-3 font-medium">{o.owner}</td>
                  <td className="px-5 py-3 text-right">{o.total}</td>
                  <td className="px-5 py-3 text-right text-emerald-500">{o.won}</td>
                  <td className="px-5 py-3 text-right">{o.high}</td>
                  <td className="px-5 py-3 text-right">
                    <span className={o.avgP >= 90 ? 'text-emerald-500' : o.avgP >= 70 ? 'text-amber-500' : 'text-neutral-400'}>
                      {o.avgP.toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-5 py-3 text-right">
                    <span className={`text-xs px-2 py-0.5 ${
                      o.winRate >= 80 ? 'bg-emerald-500/20 text-emerald-400' :
                      o.winRate >= 50 ? 'bg-amber-500/20 text-amber-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>{o.winRate.toFixed(0)}%</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
