import React, { useEffect, useMemo, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, AreaChart, Area, Legend,
} from "recharts";
import { Link } from "react-router-dom";

/* ------------ Data Fetching ------------ */
async function fetchModelRegistry() {
  try {
    const res = await fetch("/api/model_registry.json", { cache: "no-store" });
    if (!res.ok) return null;
    return res.json();
  } catch { return null; }
}

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

/* ------------ Transform ------------ */
function transformPrediction(row, idx) {
  const id = row["Opportunity ID"] || `OPP-${idx + 1}`;
  const name = row["Opportunity Name"] || "Unnamed Opportunity";
  const stage = row["Stage"] || "Unknown";
  const owner = row["Owner"] || "Unassigned";
  const pWin = toNum(row["win_prob"]);
  const predAtThr = row["pred_at_prec_thr"] === "1";
  
  const isWon = /closed won/i.test(stage);
  const isLost = /closed lost/i.test(stage);
  const isMerged = /merged/i.test(stage);
  const isClosed = isWon || isLost || isMerged;
  
  let oppType = "Other";
  if (/renewal/i.test(name)) oppType = "Renewal";
  else if (/new|expansion/i.test(name)) oppType = "New Business";
  else if (/upsell/i.test(name)) oppType = "Upsell";
  
  let confidence = "Low";
  if (Number.isFinite(pWin)) {
    if (pWin >= 0.9) confidence = "High";
    else if (pWin >= 0.7) confidence = "Medium";
    // Low includes everything below 70%
  }

  return {
    id, name: name.length > 55 ? name.substring(0, 55) + "…" : name,
    fullName: name, stage, owner,
    p_win: Number.isFinite(pWin) ? pWin : null,
    pred_win: predAtThr, oppType, confidence, isClosed,
    status: isWon ? "Won" : (isLost ? "Lost" : (isMerged ? "Merged" : "Open")),
  };
}

/* ------------ Main Component ------------ */
export default function App() {
  const [deals, setDeals] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [lossReasons, setLossReasons] = useState([]);
  const [lossRegional, setLossRegional] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("overview");
  const [filters, setFilters] = useState({ oppType: "All", confidence: "All", status: "Open" });
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [registry, csvRows, lossData, lossRegionData] = await Promise.all([
          fetchModelRegistry(),
          fetchCSVRows("/api/win_probabilities.csv"),
          fetchCSVRows("/api/feedback/loss_reasons_overall.csv"),
          fetchCSVRows("/api/feedback/loss_reasons_by_region.csv"),
        ]);
        
        if (!cancelled) {
          setModelInfo(registry);
          setLossReasons(lossData);
          setLossRegional(lossRegionData);
          
          const seen = new Set();
          const transformed = [];
          for (let i = 0; i < csvRows.length; i++) {
            const row = csvRows[i];
            const id = row["Opportunity ID"];
            if (id && !seen.has(id)) {
              seen.add(id);
              transformed.push(transformPrediction(row, i));
            }
          }
          transformed.sort((a, b) => (b.p_win ?? 0) - (a.p_win ?? 0));
          setDeals(transformed);
          setLoading(false);
        }
      } catch (e) {
        console.error("Failed to load data:", e);
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Filter options & apply
  const oppTypes = useMemo(() => ["All", ...Array.from(new Set(deals.map(d => d.oppType)))], [deals]);
  const statuses = useMemo(() => ["All", ...Array.from(new Set(deals.map(d => d.status)))], [deals]);
  
  const filtered = useMemo(() => {
    return deals.filter(d => {
      if (filters.oppType !== "All" && d.oppType !== filters.oppType) return false;
      if (filters.confidence !== "All" && d.confidence !== filters.confidence) return false;
      if (filters.status !== "All" && d.status !== filters.status) return false;
      if (searchTerm && !d.name.toLowerCase().includes(searchTerm.toLowerCase()) && 
          !d.id.toLowerCase().includes(searchTerm.toLowerCase())) return false;
      return true;
    });
  }, [deals, filters, searchTerm]);

  // KPIs - Context-aware based on current filter
  const dealsWithProb = useMemo(() => filtered.filter(d => d.p_win !== null), [filtered]);
  const avgProbability = useMemo(() => {
    if (dealsWithProb.length === 0) return 0;
    return dealsWithProb.reduce((s, d) => s + d.p_win, 0) / dealsWithProb.length;
  }, [dealsWithProb]);
  
  
  // Confidence distribution for current view
  const confidenceCounts = useMemo(() => ({
    high: dealsWithProb.filter(d => d.p_win >= 0.9).length,
    medium: dealsWithProb.filter(d => d.p_win >= 0.7 && d.p_win < 0.9).length,
    low: dealsWithProb.filter(d => d.p_win < 0.7).length,
  }), [dealsWithProb]);
  
  // Historical win rate from ALL closed deals (not filtered)
  const historicalWinRate = useMemo(() => {
    const allClosed = deals.filter(d => d.status === "Won" || d.status === "Lost");
    return allClosed.length === 0 ? 0 : allClosed.filter(d => d.status === "Won").length / allClosed.length;
  }, [deals]);
  
  // Total lost deals count
  const totalLostDeals = useMemo(() => deals.filter(d => d.status === "Lost").length, [deals]);

  // Loss Analysis
  const topLossReasons = useMemo(() => {
    return lossReasons
      .filter(r => {
        const val = (r.reason_value || "").toLowerCase();
        return !val.includes("unknown") && !val.includes("duplicate") && !val.includes("merged") && r.reason_field !== "Opportunity Reason of Churn";
      })
      .map(r => ({
        reason: r.reason_value?.length > 30 ? r.reason_value.substring(0, 30) + "…" : r.reason_value,
        fullReason: r.reason_value,
        count: parseInt(r.n_lost) || 0,
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 8);
  }, [lossReasons]);

  const lossCategories = useMemo(() => {
    const cat = { "Timing": 0, "Competition": 0, "Budget": 0, "Pricing": 0, "Product": 0, "Other": 0 };
    for (const r of lossReasons) {
      const val = (r.reason_value || "").toLowerCase();
      const count = parseInt(r.n_lost) || 0;
      if (val.includes("not now") || val.includes("priorities") || val.includes("stopped")) cat["Timing"] += count;
      else if (val.includes("not flexera") || val.includes("competitor")) cat["Competition"] += count;
      else if (val.includes("budget")) cat["Budget"] += count;
      else if (val.includes("pricing") || val.includes("price")) cat["Pricing"] += count;
      else if (val.includes("functionality") || val.includes("product")) cat["Product"] += count;
      else if (!val.includes("unknown") && !val.includes("duplicate")) cat["Other"] += count;
    }
    return Object.entries(cat).filter(([_, v]) => v > 0).map(([name, value]) => ({ name, value })).sort((a, b) => b.value - a.value);
  }, [lossReasons]);

  const lossByRegion = useMemo(() => {
    const data = {};
    for (const r of lossRegional) {
      if (r.segment_field === "Account Region" && r.segment_value && r.segment_value !== "Unknown") {
        const region = r.segment_value;
        const val = (r.reason_value || "").toLowerCase();
        if (!data[region]) data[region] = { region, timing: 0, competition: 0, budget: 0, pricing: 0 };
        const count = parseInt(r.n_lost) || 0;
        if (val.includes("not now") || val.includes("priorities") || val.includes("stopped")) data[region].timing += count;
        else if (val.includes("not flexera") || val.includes("competitor")) data[region].competition += count;
        else if (val.includes("budget")) data[region].budget += count;
        else if (val.includes("pricing") || val.includes("price")) data[region].pricing += count;
      }
    }
    return Object.values(data);
  }, [lossRegional]);

  // Charts
  const byOppType = useMemo(() => {
    const g = {};
    for (const d of filtered) {
      g[d.oppType] ??= { type: d.oppType, count: 0, total: 0 };
      g[d.oppType].count++;
      if (d.p_win !== null) g[d.oppType].total += d.p_win;
    }
    return Object.values(g).map(v => ({ ...v, avgP: v.count > 0 ? Math.round((v.total / v.count) * 100) : 0 }));
  }, [filtered]);

  // Charts use current filtered data
  const byConfidence = useMemo(() => {
    const g = { High: 0, Medium: 0, Low: 0 };
    for (const d of dealsWithProb) g[d.confidence] = (g[d.confidence] || 0) + 1;
    return Object.entries(g).map(([name, value]) => ({ name, value }));
  }, [dealsWithProb]);

  const probDist = useMemo(() => {
    const bins = Array(10).fill(0).map((_, i) => ({ range: `${i * 10}–${(i + 1) * 10}`, count: 0 }));
    for (const d of dealsWithProb) bins[Math.min(9, Math.floor(d.p_win * 10))].count++;
    return bins;
  }, [dealsWithProb]);

  const modelMetrics = useMemo(() => {
    if (!modelInfo?.metrics) return null;
    return modelInfo.metrics.find(m => m.model === modelInfo.best_model) || modelInfo.metrics[0];
  }, [modelInfo]);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0f0f0f] flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-amber-500 border-t-transparent rounded-full animate-spin mx-auto" />
          <div className="mt-4 text-neutral-400 text-sm tracking-wide">Loading data…</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0f0f0f] text-neutral-100">
      {/* Header */}
      <header className="border-b border-neutral-800">
        <div className="max-w-[1400px] mx-auto px-8 py-5">
          <div className="flex items-end justify-between">
            <div>
              <div className="text-xs text-neutral-500 uppercase tracking-[0.2em] mb-1">Flexera Intelligence</div>
              <h1 className="text-2xl font-light tracking-tight">Win/Loss Analytics</h1>
            </div>
            <div className="flex items-center gap-10 text-sm">
              <div>
                <div className="text-neutral-500 text-xs mb-0.5">Model</div>
                <div className="text-amber-500 font-medium">{modelInfo?.best_model || "XGBoost"}</div>
              </div>
              <div>
                <div className="text-neutral-500 text-xs mb-0.5">AUC Score</div>
                <div className="font-medium">{modelMetrics ? (parseFloat(modelMetrics.val_auc) * 100).toFixed(1) : "—"}%</div>
              </div>
              <div>
                <div className="text-neutral-500 text-xs mb-0.5">Total Deals</div>
                <div className="font-medium">{deals.length.toLocaleString()}</div>
              </div>
              <Link to="/model-insights" className="text-amber-500 hover:text-amber-400 text-xs uppercase tracking-wider">
                Model Details →
              </Link>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-[1400px] mx-auto px-8 py-6">
        {/* Tab Navigation */}
        <div className="flex gap-1 mb-8 border-b border-neutral-800">
          <button onClick={() => setActiveTab("overview")}
            className={`px-5 py-3 text-sm tracking-wide transition-colors relative ${
              activeTab === "overview" ? "text-neutral-100" : "text-neutral-500 hover:text-neutral-300"}`}>
            Pipeline
            {activeTab === "overview" && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-amber-500" />}
          </button>
          <button onClick={() => setActiveTab("loss")}
            className={`px-5 py-3 text-sm tracking-wide transition-colors relative ${
              activeTab === "loss" ? "text-neutral-100" : "text-neutral-500 hover:text-neutral-300"}`}>
            Loss Analysis
            {activeTab === "loss" && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-red-500" />}
          </button>
        </div>

        {activeTab === "overview" ? (
          <>
            {/* Metrics Row */}
            <div className="grid grid-cols-3 gap-6 mb-8">
              <div className="bg-amber-950/30 border border-amber-900/50 p-5">
                <div className="text-amber-400/70 text-xs uppercase tracking-wider mb-2">
                  {filters.status === "Open" ? "Avg Predicted p(Win)" : 
                   filters.status === "Won" ? "Avg p(Win) — Won" :
                   filters.status === "Lost" ? "Avg p(Win) — Lost" : "Avg p(Win)"}
                </div>
                <div className="text-3xl font-light text-amber-400">{(avgProbability * 100).toFixed(1)}%</div>
                <div className="text-xs text-neutral-600 mt-1">{dealsWithProb.length.toLocaleString()} deals</div>
              </div>
              <div className="bg-neutral-900/50 border border-neutral-800 p-5">
                <div className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Confidence Split</div>
                <div className="flex items-baseline gap-3">
                  <div className="text-center">
                    <div className="text-xl text-emerald-500">{confidenceCounts.high.toLocaleString()}</div>
                    <div className="text-xs text-neutral-600">High</div>
                  </div>
                  <div className="text-center">
                    <div className="text-xl text-amber-500">{confidenceCounts.medium.toLocaleString()}</div>
                    <div className="text-xs text-neutral-600">Med</div>
                  </div>
                  <div className="text-center">
                    <div className="text-xl text-red-400">{confidenceCounts.low.toLocaleString()}</div>
                    <div className="text-xs text-neutral-600">Low</div>
                  </div>
                </div>
              </div>
              <div className="bg-neutral-900/50 border border-neutral-800 p-5">
                <div className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Historical Win Rate</div>
                <div className="text-3xl font-light">{(historicalWinRate * 100).toFixed(1)}%</div>
                <div className="text-xs text-neutral-600 mt-1">All time ({deals.filter(d => d.isClosed).length.toLocaleString()} closed)</div>
              </div>
            </div>

            {/* Filters */}
            <div className="bg-neutral-900/30 border border-neutral-800 p-4 mb-6">
              <div className="flex flex-wrap items-center gap-4">
                <input type="text" placeholder="Search opportunities…" value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="flex-1 min-w-[200px] bg-transparent border border-neutral-700 px-4 py-2 text-sm focus:outline-none focus:border-amber-500/50 placeholder:text-neutral-600" />
                <select value={filters.oppType} onChange={(e) => setFilters(f => ({ ...f, oppType: e.target.value }))}
                  className="bg-transparent border border-neutral-700 px-3 py-2 text-sm focus:outline-none">
                  {oppTypes.map(t => <option key={t} value={t} className="bg-neutral-900">{t === "All" ? "All types" : t}</option>)}
                </select>
                <select value={filters.confidence} onChange={(e) => setFilters(f => ({ ...f, confidence: e.target.value }))}
                  className="bg-transparent border border-neutral-700 px-3 py-2 text-sm focus:outline-none">
                  <option value="All" className="bg-neutral-900">All confidence</option>
                  <option value="High" className="bg-neutral-900">High (≥90%)</option>
                  <option value="Medium" className="bg-neutral-900">Medium (70-90%)</option>
                  <option value="Low" className="bg-neutral-900">Low (&lt;70%)</option>
                </select>
                <select value={filters.status} onChange={(e) => setFilters(f => ({ ...f, status: e.target.value }))}
                  className="bg-transparent border border-neutral-700 px-3 py-2 text-sm focus:outline-none">
                  {statuses.map(s => <option key={s} value={s} className="bg-neutral-900">{s === "All" ? "All statuses" : s}</option>)}
                </select>
                <button onClick={() => { setFilters({ oppType: "All", confidence: "All", status: "Open" }); setSearchTerm(""); }}
                  className="text-xs text-neutral-500 hover:text-neutral-300">Reset</button>
                <div className="text-xs text-neutral-500 ml-auto">{filtered.length.toLocaleString()} {filters.status === "All" ? "deals" : filters.status.toLowerCase() + " deals"}</div>
              </div>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-2 gap-6 mb-8">
              <div className="bg-neutral-900/30 border border-neutral-800 p-5">
                <div className="text-xs text-neutral-500 uppercase tracking-wider mb-4">Probability Distribution</div>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={probDist}>
                      <defs>
                        <linearGradient id="probGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.3}/>
                          <stop offset="100%" stopColor="#f59e0b" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="#262626" strokeDasharray="3 3" />
                      <XAxis dataKey="range" tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#404040' }} />
                      <YAxis tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#404040' }} />
                      <Tooltip contentStyle={{ background: '#171717', border: '1px solid #404040', borderRadius: 0 }} />
                      <Area type="monotone" dataKey="count" stroke="#f59e0b" strokeWidth={1.5} fill="url(#probGrad)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-neutral-900/30 border border-neutral-800 p-5">
                <div className="text-xs text-neutral-500 uppercase tracking-wider mb-4">By Opportunity Type</div>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={byOppType}>
                      <CartesianGrid stroke="#262626" strokeDasharray="3 3" />
                      <XAxis dataKey="type" tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#404040' }} />
                      <YAxis yAxisId="l" tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#404040' }} />
                      <YAxis yAxisId="r" orientation="right" tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#404040' }} />
                      <Tooltip contentStyle={{ background: '#171717', border: '1px solid #404040', borderRadius: 0 }} />
                      <Bar yAxisId="l" dataKey="count" fill="#525252" name="Count" />
                      <Bar yAxisId="r" dataKey="avgP" fill="#f59e0b" name="Avg %" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-neutral-900/30 border border-neutral-800 p-5">
                <div className="text-xs text-neutral-500 uppercase tracking-wider mb-4">Confidence Breakdown</div>
                <div className="h-56 flex items-center justify-center">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={byConfidence} cx="50%" cy="50%" innerRadius={50} outerRadius={80} dataKey="value" nameKey="name">
                        <Cell fill="#10b981" />
                        <Cell fill="#f59e0b" />
                        <Cell fill="#ef4444" />
                      </Pie>
                      <Tooltip contentStyle={{ background: '#171717', border: '1px solid #404040', borderRadius: 0 }} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="text-xs space-y-2">
                    <div className="flex items-center gap-2"><div className="w-2 h-2 bg-emerald-500" /> High (≥90%)</div>
                    <div className="flex items-center gap-2"><div className="w-2 h-2 bg-amber-500" /> Medium (70-90%)</div>
                    <div className="flex items-center gap-2"><div className="w-2 h-2 bg-red-500" /> Low (&lt;70%)</div>
                  </div>
                </div>
              </div>

              <div className="bg-neutral-900/30 border border-neutral-800 p-5">
                <div className="text-xs text-neutral-500 uppercase tracking-wider mb-4">Quick Actions</div>
                <div className="space-y-3">
                  <Link to="/deal" className="block p-3 border border-neutral-700 hover:border-amber-500/50 transition-colors">
                    <div className="text-sm font-medium">View All Deals</div>
                    <div className="text-xs text-neutral-500 mt-0.5">Browse {deals.length.toLocaleString()} opportunities</div>
                  </Link>
                  <button onClick={() => setActiveTab("loss")} className="w-full text-left p-3 border border-neutral-700 hover:border-red-500/50 transition-colors">
                    <div className="text-sm font-medium">Analyze Losses</div>
                    <div className="text-xs text-neutral-500 mt-0.5">Understand why deals are lost</div>
                  </button>
                  <Link to="/competitors" className="block p-3 border border-neutral-700 hover:border-amber-500/50 transition-colors">
                    <div className="text-sm font-medium">Team Performance</div>
                    <div className="text-xs text-neutral-500 mt-0.5">Sales analytics by owner</div>
                  </Link>
                </div>
              </div>
            </div>

            {/* Table */}
            <div className="bg-neutral-900/30 border border-neutral-800">
              <div className="px-5 py-4 border-b border-neutral-800">
                <div className="text-xs text-neutral-500 uppercase tracking-wider">Opportunities</div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-neutral-500 text-xs uppercase tracking-wider border-b border-neutral-800">
                      <th className="px-5 py-3 font-medium">Opportunity</th>
                      <th className="px-5 py-3 font-medium">Type</th>
                      <th className="px-5 py-3 font-medium">Owner</th>
                      <th className="px-5 py-3 font-medium">Stage</th>
                      <th className="px-5 py-3 font-medium text-right">Probability</th>
                      <th className="px-5 py-3 font-medium">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.slice(0, 50).map((d, i) => (
                      <tr key={d.id} className={`border-b border-neutral-800/50 hover:bg-neutral-800/30 ${i % 2 === 0 ? 'bg-neutral-900/20' : ''}`}>
                        <td className="px-5 py-3">
                          <Link to={`/deal/${encodeURIComponent(d.id)}`} state={{ deal: d }} className="text-amber-500 hover:underline">
                            {d.name}
                          </Link>
                          <div className="text-xs text-neutral-600 mt-0.5">{d.id}</div>
                        </td>
                        <td className="px-5 py-3 text-neutral-400">{d.oppType}</td>
                        <td className="px-5 py-3 text-neutral-400">{d.owner}</td>
                        <td className="px-5 py-3 text-neutral-500 text-xs">{d.stage}</td>
                        <td className="px-5 py-3 text-right">
                          {d.p_win !== null ? (
                            <span className={
                              d.p_win >= 0.9 ? 'text-emerald-500' : 
                              d.p_win >= 0.7 ? 'text-amber-500' : 
                              'text-red-400'
                            }>
                              {(d.p_win * 100).toFixed(1)}%
                            </span>
                          ) : '—'}
                        </td>
                        <td className="px-5 py-3">
                          <span className={`text-xs px-2 py-0.5 ${
                            d.status === 'Won' ? 'bg-emerald-500/20 text-emerald-400' :
                            d.status === 'Lost' ? 'bg-red-500/20 text-red-400' :
                            'bg-neutral-700/50 text-neutral-400'
                          }`}>{d.status}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {filtered.length > 50 && (
                <div className="px-5 py-3 text-xs text-neutral-500 border-t border-neutral-800">
                  Showing 50 of {filtered.length.toLocaleString()} — <Link to="/deal" className="text-amber-500 hover:underline">View all</Link>
                </div>
              )}
            </div>
          </>
        ) : (
          /* Loss Analysis */
          <>
            {/* Loss Metrics */}
            <div className="grid grid-cols-4 gap-6 mb-8">
              <div className="bg-red-950/30 border border-red-900/50 p-5">
                <div className="text-red-400/70 text-xs uppercase tracking-wider mb-2">Total Lost</div>
                <div className="text-3xl font-light text-red-400">{totalLostDeals.toLocaleString()}</div>
              </div>
              <div className="bg-neutral-900/50 border border-neutral-800 p-5">
                <div className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Top Reason</div>
                <div className="text-xl font-light">Timing Issues</div>
                <div className="text-xs text-neutral-600 mt-1">{lossCategories[0]?.value?.toLocaleString() || 0} deals</div>
              </div>
              <div className="bg-neutral-900/50 border border-neutral-800 p-5">
                <div className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Competitive Loss</div>
                <div className="text-xl font-light text-orange-400">{(lossCategories.find(c => c.name === "Competition")?.value || 0).toLocaleString()}</div>
              </div>
              <div className="bg-neutral-900/50 border border-neutral-800 p-5">
                <div className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Pricing Issues</div>
                <div className="text-xl font-light text-amber-400">{(lossCategories.find(c => c.name === "Pricing")?.value || 0).toLocaleString()}</div>
              </div>
            </div>

            {/* Key Insights */}
            <div className="bg-red-950/20 border border-red-900/30 p-6 mb-8">
              <div className="text-xs text-red-400/70 uppercase tracking-wider mb-4">Key Findings</div>
              <div className="grid grid-cols-3 gap-6">
                <div>
                  <div className="text-2xl font-light text-red-400 mb-1">#1</div>
                  <div className="font-medium">Timing & Readiness</div>
                  <div className="text-sm text-neutral-400 mt-1">Most deals lost due to customers not being ready. Consider better qualification earlier in the cycle.</div>
                </div>
                <div>
                  <div className="text-2xl font-light text-orange-400 mb-1">#2</div>
                  <div className="font-medium">Competitive Pressure</div>
                  <div className="text-sm text-neutral-400 mt-1">Significant losses to competitors. Review battle cards and differentiation messaging.</div>
                </div>
                <div>
                  <div className="text-2xl font-light text-amber-400 mb-1">#3</div>
                  <div className="font-medium">Budget & Pricing</div>
                  <div className="text-sm text-neutral-400 mt-1">Combined budget and pricing issues represent notable loss driver. Consider value-based selling.</div>
                </div>
              </div>
            </div>

            {/* Loss Charts */}
            <div className="grid grid-cols-2 gap-6 mb-8">
              <div className="bg-neutral-900/30 border border-neutral-800 p-5">
                <div className="text-xs text-neutral-500 uppercase tracking-wider mb-4">Loss Reasons</div>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={topLossReasons} layout="vertical">
                      <CartesianGrid stroke="#262626" strokeDasharray="3 3" />
                      <XAxis type="number" tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#404040' }} />
                      <YAxis type="category" dataKey="reason" tick={{ fill: '#a3a3a3', fontSize: 10 }} width={100} axisLine={{ stroke: '#404040' }} />
                      <Tooltip contentStyle={{ background: '#171717', border: '1px solid #404040', borderRadius: 0 }}
                        formatter={(v) => [v.toLocaleString(), "Lost"]}
                        labelFormatter={(_, p) => p?.[0]?.payload?.fullReason || ""} />
                      <Bar dataKey="count" fill="#ef4444" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-neutral-900/30 border border-neutral-800 p-5">
                <div className="text-xs text-neutral-500 uppercase tracking-wider mb-4">Loss Categories</div>
                <div className="h-64 flex items-center">
                  <ResponsiveContainer width="60%" height="100%">
                    <PieChart>
                      <Pie data={lossCategories} cx="50%" cy="50%" innerRadius={45} outerRadius={75} dataKey="value">
                        {lossCategories.map((_, i) => (
                          <Cell key={i} fill={['#ef4444', '#f97316', '#eab308', '#a3e635', '#22d3ee', '#818cf8'][i % 6]} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={{ background: '#171717', border: '1px solid #404040', borderRadius: 0 }} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="text-xs space-y-1.5">
                    {lossCategories.slice(0, 5).map((c, i) => (
                      <div key={c.name} className="flex items-center gap-2">
                        <div className="w-2 h-2" style={{ background: ['#ef4444', '#f97316', '#eab308', '#a3e635', '#22d3ee'][i] }} />
                        <span className="text-neutral-400">{c.name}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-neutral-900/30 border border-neutral-800 p-5 col-span-2">
                <div className="text-xs text-neutral-500 uppercase tracking-wider mb-4">Loss Patterns by Region</div>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={lossByRegion}>
                      <CartesianGrid stroke="#262626" strokeDasharray="3 3" />
                      <XAxis dataKey="region" tick={{ fill: '#737373', fontSize: 11 }} axisLine={{ stroke: '#404040' }} />
                      <YAxis tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#404040' }} />
                      <Tooltip contentStyle={{ background: '#171717', border: '1px solid #404040', borderRadius: 0 }} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Bar dataKey="timing" name="Timing" stackId="a" fill="#f59e0b" />
                      <Bar dataKey="competition" name="Competition" stackId="a" fill="#ef4444" />
                      <Bar dataKey="budget" name="Budget" stackId="a" fill="#8b5cf6" />
                      <Bar dataKey="pricing" name="Pricing" stackId="a" fill="#06b6d4" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Loss Reasons Table */}
            <div className="bg-neutral-900/30 border border-neutral-800">
              <div className="px-5 py-4 border-b border-neutral-800">
                <div className="text-xs text-neutral-500 uppercase tracking-wider">All Loss Reasons</div>
              </div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-neutral-500 text-xs uppercase tracking-wider border-b border-neutral-800">
                    <th className="px-5 py-3 font-medium w-8">#</th>
                    <th className="px-5 py-3 font-medium">Reason</th>
                    <th className="px-5 py-3 font-medium text-right">Lost Deals</th>
                    <th className="px-5 py-3 font-medium w-32">Impact</th>
                  </tr>
                </thead>
                <tbody>
                  {topLossReasons.map((r, i) => (
                    <tr key={i} className="border-b border-neutral-800/50 hover:bg-neutral-800/30">
                      <td className="px-5 py-3 text-neutral-600">{i + 1}</td>
                      <td className="px-5 py-3">{r.fullReason}</td>
                      <td className="px-5 py-3 text-right text-red-400 font-medium">{r.count.toLocaleString()}</td>
                      <td className="px-5 py-3">
                        <div className="h-1.5 bg-neutral-800 w-full">
                          <div className="h-full bg-red-500" style={{ width: `${(r.count / (topLossReasons[0]?.count || 1)) * 100}%` }} />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
