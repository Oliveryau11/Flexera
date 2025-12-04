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
  const region = row["Region"] || "Unknown";
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
    fullName: name, stage, owner, region,
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
  const [filters, setFilters] = useState({ oppType: "All", confidence: "All", status: "Open", region: "All" });
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
  const regions = useMemo(() => ["All", ...Array.from(new Set(deals.map(d => d.region).filter(r => r && r !== "Unknown")))], [deals]);
  
  const filtered = useMemo(() => {
    return deals.filter(d => {
      if (filters.oppType !== "All" && d.oppType !== filters.oppType) return false;
      if (filters.confidence !== "All" && d.confidence !== filters.confidence) return false;
      if (filters.status !== "All" && d.status !== filters.status) return false;
      if (filters.region !== "All" && d.region !== filters.region) return false;
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
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto" />
          <div className="mt-4 text-slate-500 text-sm">Loading data…</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-[1400px] mx-auto px-8 py-5">
          <div className="flex items-end justify-between">
            <div>
              <div className="text-xs text-slate-400 uppercase tracking-[0.2em] mb-1">Flexera Intelligence</div>
              <h1 className="text-2xl font-semibold text-slate-800">Win/Loss Analytics</h1>
            </div>
            <div className="flex items-center gap-10 text-sm">
              <div>
                <div className="text-slate-400 text-xs mb-0.5">Model</div>
                <div className="text-blue-600 font-semibold">{modelInfo?.best_model || "XGBoost"}</div>
              </div>
              <div>
                <div className="text-slate-400 text-xs mb-0.5">AUC Score</div>
                <div className="font-semibold text-slate-700">{modelMetrics ? (parseFloat(modelMetrics.val_auc) * 100).toFixed(1) : "—"}%</div>
              </div>
              <div>
                <div className="text-slate-400 text-xs mb-0.5">Total Deals</div>
                <div className="font-semibold text-slate-700">{deals.length.toLocaleString()}</div>
              </div>
              <Link to="/model-insights" className="text-blue-600 hover:text-blue-700 text-xs font-medium uppercase tracking-wider">
                Model Details →
              </Link>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-[1400px] mx-auto px-8 py-6">
        {/* Tab Navigation */}
        <div className="flex gap-1 mb-8 border-b border-slate-200">
          <button onClick={() => setActiveTab("overview")}
            className={`px-5 py-3 text-sm font-medium transition-colors relative ${
              activeTab === "overview" ? "text-slate-800" : "text-slate-400 hover:text-slate-600"}`}>
            Pipeline
            {activeTab === "overview" && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500 rounded-full" />}
          </button>
          <button onClick={() => setActiveTab("loss")}
            className={`px-5 py-3 text-sm font-medium transition-colors relative ${
              activeTab === "loss" ? "text-slate-800" : "text-slate-400 hover:text-slate-600"}`}>
            Loss Analysis
            {activeTab === "loss" && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-rose-500 rounded-full" />}
          </button>
        </div>

        {activeTab === "overview" ? (
          <>
            {/* Metrics Row */}
            <div className="grid grid-cols-3 gap-6 mb-8">
              <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl p-5 text-white shadow-lg shadow-blue-500/20">
                <div className="text-blue-100 text-xs uppercase tracking-wider mb-2">
                  {filters.status === "Open" ? "Avg Predicted p(Win)" : 
                   filters.status === "Won" ? "Avg p(Win) — Won" :
                   filters.status === "Lost" ? "Avg p(Win) — Lost" : "Avg p(Win)"}
                </div>
                <div className="text-3xl font-bold">{(avgProbability * 100).toFixed(1)}%</div>
                <div className="text-xs text-blue-100 mt-1">{dealsWithProb.length.toLocaleString()} deals</div>
              </div>
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <div className="text-slate-400 text-xs uppercase tracking-wider mb-2">Confidence Split</div>
                <div className="flex items-baseline gap-4">
                  <div className="text-center">
                    <div className="text-xl font-bold text-emerald-500">{confidenceCounts.high.toLocaleString()}</div>
                    <div className="text-xs text-slate-400">High</div>
                  </div>
                  <div className="text-center">
                    <div className="text-xl font-bold text-amber-500">{confidenceCounts.medium.toLocaleString()}</div>
                    <div className="text-xs text-slate-400">Med</div>
                  </div>
                  <div className="text-center">
                    <div className="text-xl font-bold text-rose-500">{confidenceCounts.low.toLocaleString()}</div>
                    <div className="text-xs text-slate-400">Low</div>
                  </div>
                </div>
              </div>
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <div className="text-slate-400 text-xs uppercase tracking-wider mb-2">Historical Win Rate</div>
                <div className="text-3xl font-bold text-slate-700">{(historicalWinRate * 100).toFixed(1)}%</div>
                <div className="text-xs text-slate-400 mt-1">All time ({deals.filter(d => d.isClosed).length.toLocaleString()} closed)</div>
              </div>
            </div>

            {/* Filters */}
            <div className="bg-white rounded-xl border border-slate-200 p-4 mb-6 shadow-sm">
              <div className="flex flex-wrap items-center gap-3">
                <input type="text" placeholder="Search opportunities…" value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="flex-1 min-w-[180px] bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100 placeholder:text-slate-400" />
                <select value={filters.region} onChange={(e) => setFilters(f => ({ ...f, region: e.target.value }))}
                  className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-400">
                  {regions.map(r => <option key={r} value={r}>{r === "All" ? "All regions" : r}</option>)}
                </select>
                <select value={filters.oppType} onChange={(e) => setFilters(f => ({ ...f, oppType: e.target.value }))}
                  className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-400">
                  {oppTypes.map(t => <option key={t} value={t}>{t === "All" ? "All types" : t}</option>)}
                </select>
                <select value={filters.confidence} onChange={(e) => setFilters(f => ({ ...f, confidence: e.target.value }))}
                  className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-400">
                  <option value="All">All confidence</option>
                  <option value="High">High (≥90%)</option>
                  <option value="Medium">Medium (70-90%)</option>
                  <option value="Low">Low (&lt;70%)</option>
                </select>
                <select value={filters.status} onChange={(e) => setFilters(f => ({ ...f, status: e.target.value }))}
                  className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-400">
                  {statuses.map(s => <option key={s} value={s}>{s === "All" ? "All statuses" : s}</option>)}
                </select>
                <button onClick={() => { setFilters({ oppType: "All", confidence: "All", status: "Open", region: "All" }); setSearchTerm(""); }}
                  className="text-xs text-slate-400 hover:text-slate-600 font-medium">Reset</button>
                <div className="text-xs text-slate-500 ml-auto font-medium">{filtered.length.toLocaleString()} {filters.status === "All" ? "deals" : filters.status.toLowerCase() + " deals"}</div>
              </div>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-2 gap-6 mb-8">
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <div className="text-xs text-slate-400 uppercase tracking-wider mb-4 font-medium">Probability Distribution</div>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={probDist}>
                      <defs>
                        <linearGradient id="probGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3}/>
                          <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.05}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
                      <XAxis dataKey="range" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={{ stroke: '#cbd5e1' }} />
                      <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={{ stroke: '#cbd5e1' }} />
                      <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8, boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                      <Area type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} fill="url(#probGrad)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <div className="text-xs text-slate-400 uppercase tracking-wider mb-4 font-medium">By Opportunity Type</div>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={byOppType}>
                      <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
                      <XAxis dataKey="type" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={{ stroke: '#cbd5e1' }} />
                      <YAxis yAxisId="l" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={{ stroke: '#cbd5e1' }} />
                      <YAxis yAxisId="r" orientation="right" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={{ stroke: '#cbd5e1' }} />
                      <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8 }} />
                      <Bar yAxisId="l" dataKey="count" fill="#94a3b8" name="Count" radius={[4, 4, 0, 0]} />
                      <Bar yAxisId="r" dataKey="avgP" fill="#3b82f6" name="Avg %" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <div className="text-xs text-slate-400 uppercase tracking-wider mb-4 font-medium">Confidence Breakdown</div>
                <div className="h-56 flex items-center justify-center">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={byConfidence} cx="50%" cy="50%" innerRadius={50} outerRadius={80} dataKey="value" nameKey="name">
                        <Cell fill="#10b981" />
                        <Cell fill="#f59e0b" />
                        <Cell fill="#f43f5e" />
                      </Pie>
                      <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8 }} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="text-xs space-y-2 text-slate-600">
                    <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-emerald-500" /> High (≥90%)</div>
                    <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-amber-500" /> Medium (70-90%)</div>
                    <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-rose-500" /> Low (&lt;70%)</div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <div className="text-xs text-slate-400 uppercase tracking-wider mb-4 font-medium">Quick Actions</div>
                <div className="space-y-3">
                  <Link to="/deal" className="block p-3 bg-slate-50 rounded-lg border border-slate-200 hover:border-blue-300 hover:bg-blue-50 transition-colors">
                    <div className="text-sm font-semibold text-slate-700">View All Deals</div>
                    <div className="text-xs text-slate-400 mt-0.5">Browse {deals.length.toLocaleString()} opportunities</div>
                  </Link>
                  <button onClick={() => setActiveTab("loss")} className="w-full text-left p-3 bg-slate-50 rounded-lg border border-slate-200 hover:border-rose-300 hover:bg-rose-50 transition-colors">
                    <div className="text-sm font-semibold text-slate-700">Analyze Losses</div>
                    <div className="text-xs text-slate-400 mt-0.5">Understand why deals are lost</div>
                  </button>
                  <Link to="/competitors" className="block p-3 bg-slate-50 rounded-lg border border-slate-200 hover:border-blue-300 hover:bg-blue-50 transition-colors">
                    <div className="text-sm font-semibold text-slate-700">Team Performance</div>
                    <div className="text-xs text-slate-400 mt-0.5">Sales analytics by owner</div>
                  </Link>
                </div>
              </div>
            </div>

            {/* Table */}
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
              <div className="px-5 py-4 border-b border-slate-100 bg-slate-50">
                <div className="text-xs text-slate-500 uppercase tracking-wider font-medium">Opportunities</div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-slate-400 text-xs uppercase tracking-wider border-b border-slate-100 bg-slate-50/50">
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
                      <tr key={d.id} className={`border-b border-slate-100 hover:bg-blue-50/50 transition-colors ${i % 2 === 0 ? 'bg-slate-50/30' : ''}`}>
                        <td className="px-5 py-3">
                          <Link to={`/deal/${encodeURIComponent(d.id)}`} state={{ deal: d }} className="text-blue-600 hover:text-blue-700 font-medium hover:underline">
                            {d.name}
                          </Link>
                          <div className="text-xs text-slate-400 mt-0.5">{d.id}</div>
                        </td>
                        <td className="px-5 py-3 text-slate-600">{d.oppType}</td>
                        <td className="px-5 py-3 text-slate-600">{d.owner}</td>
                        <td className="px-5 py-3 text-slate-400 text-xs">{d.stage}</td>
                        <td className="px-5 py-3 text-right">
                          {d.p_win !== null ? (
                            <span className={`font-semibold ${
                              d.p_win >= 0.9 ? 'text-emerald-600' : 
                              d.p_win >= 0.7 ? 'text-amber-600' : 
                              'text-rose-600'
                            }`}>
                              {(d.p_win * 100).toFixed(1)}%
                            </span>
                          ) : '—'}
                        </td>
                        <td className="px-5 py-3">
                          <span className={`text-xs px-2.5 py-1 rounded-full font-medium ${
                            d.status === 'Won' ? 'bg-emerald-100 text-emerald-700' :
                            d.status === 'Lost' ? 'bg-rose-100 text-rose-700' :
                            'bg-slate-100 text-slate-600'
                          }`}>{d.status}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {filtered.length > 50 && (
                <div className="px-5 py-3 text-xs text-slate-500 border-t border-slate-100 bg-slate-50">
                  Showing 50 of {filtered.length.toLocaleString()} — <Link to="/deal" className="text-blue-600 hover:underline font-medium">View all</Link>
                </div>
              )}
            </div>
          </>
        ) : (
          /* Loss Analysis */
          <>
            {/* Loss Metrics */}
            <div className="grid grid-cols-4 gap-6 mb-8">
              <div className="bg-gradient-to-br from-rose-500 to-rose-600 rounded-xl p-5 text-white shadow-lg shadow-rose-500/20">
                <div className="text-rose-100 text-xs uppercase tracking-wider mb-2">Total Lost</div>
                <div className="text-3xl font-bold">{totalLostDeals.toLocaleString()}</div>
              </div>
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <div className="text-slate-400 text-xs uppercase tracking-wider mb-2">Top Reason</div>
                <div className="text-xl font-bold text-slate-700">Timing Issues</div>
                <div className="text-xs text-slate-400 mt-1">{lossCategories[0]?.value?.toLocaleString() || 0} deals</div>
              </div>
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <div className="text-slate-400 text-xs uppercase tracking-wider mb-2">Competitive Loss</div>
                <div className="text-xl font-bold text-orange-500">{(lossCategories.find(c => c.name === "Competition")?.value || 0).toLocaleString()}</div>
              </div>
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <div className="text-slate-400 text-xs uppercase tracking-wider mb-2">Pricing Issues</div>
                <div className="text-xl font-bold text-amber-500">{(lossCategories.find(c => c.name === "Pricing")?.value || 0).toLocaleString()}</div>
              </div>
            </div>

            {/* Key Insights */}
            <div className="bg-rose-50 border border-rose-200 rounded-xl p-6 mb-8">
              <div className="text-xs text-rose-600 uppercase tracking-wider mb-4 font-semibold">Key Findings</div>
              <div className="grid grid-cols-3 gap-6">
                <div>
                  <div className="text-2xl font-bold text-rose-500 mb-1">#1</div>
                  <div className="font-semibold text-slate-700">Timing & Readiness</div>
                  <div className="text-sm text-slate-500 mt-1">Most deals lost due to customers not being ready. Consider better qualification earlier in the cycle.</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-orange-500 mb-1">#2</div>
                  <div className="font-semibold text-slate-700">Competitive Pressure</div>
                  <div className="text-sm text-slate-500 mt-1">Significant losses to competitors. Review battle cards and differentiation messaging.</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-amber-500 mb-1">#3</div>
                  <div className="font-semibold text-slate-700">Budget & Pricing</div>
                  <div className="text-sm text-slate-500 mt-1">Combined budget and pricing issues represent notable loss driver. Consider value-based selling.</div>
                </div>
              </div>
            </div>

            {/* Loss Charts */}
            <div className="grid grid-cols-2 gap-6 mb-8">
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <div className="text-xs text-slate-400 uppercase tracking-wider mb-4 font-medium">Loss Reasons</div>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={topLossReasons} layout="vertical">
                      <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
                      <XAxis type="number" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={{ stroke: '#cbd5e1' }} />
                      <YAxis type="category" dataKey="reason" tick={{ fill: '#475569', fontSize: 10 }} width={100} axisLine={{ stroke: '#cbd5e1' }} />
                      <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8 }}
                        formatter={(v) => [v.toLocaleString(), "Lost"]}
                        labelFormatter={(_, p) => p?.[0]?.payload?.fullReason || ""} />
                      <Bar dataKey="count" fill="#f43f5e" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <div className="text-xs text-slate-400 uppercase tracking-wider mb-4 font-medium">Loss Categories</div>
                <div className="h-64 flex items-center">
                  <ResponsiveContainer width="60%" height="100%">
                    <PieChart>
                      <Pie data={lossCategories} cx="50%" cy="50%" innerRadius={45} outerRadius={75} dataKey="value">
                        {lossCategories.map((_, i) => (
                          <Cell key={i} fill={['#f43f5e', '#f97316', '#eab308', '#22c55e', '#06b6d4', '#8b5cf6'][i % 6]} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8 }} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="text-xs space-y-1.5">
                    {lossCategories.slice(0, 5).map((c, i) => (
                      <div key={c.name} className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full" style={{ background: ['#f43f5e', '#f97316', '#eab308', '#22c55e', '#06b6d4'][i] }} />
                        <span className="text-slate-600">{c.name}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm col-span-2">
                <div className="text-xs text-slate-400 uppercase tracking-wider mb-4 font-medium">Loss Patterns by Region</div>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={lossByRegion}>
                      <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
                      <XAxis dataKey="region" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={{ stroke: '#cbd5e1' }} />
                      <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={{ stroke: '#cbd5e1' }} />
                      <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8 }} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Bar dataKey="timing" name="Timing" stackId="a" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="competition" name="Competition" stackId="a" fill="#f43f5e" />
                      <Bar dataKey="budget" name="Budget" stackId="a" fill="#8b5cf6" />
                      <Bar dataKey="pricing" name="Pricing" stackId="a" fill="#06b6d4" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Loss Reasons Table */}
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
              <div className="px-5 py-4 border-b border-slate-100 bg-slate-50">
                <div className="text-xs text-slate-500 uppercase tracking-wider font-medium">All Loss Reasons</div>
              </div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-slate-400 text-xs uppercase tracking-wider border-b border-slate-100 bg-slate-50/50">
                    <th className="px-5 py-3 font-medium w-8">#</th>
                    <th className="px-5 py-3 font-medium">Reason</th>
                    <th className="px-5 py-3 font-medium text-right">Lost Deals</th>
                    <th className="px-5 py-3 font-medium w-32">Impact</th>
                  </tr>
                </thead>
                <tbody>
                  {topLossReasons.map((r, i) => (
                    <tr key={i} className="border-b border-slate-100 hover:bg-rose-50/50 transition-colors">
                      <td className="px-5 py-3 text-slate-400">{i + 1}</td>
                      <td className="px-5 py-3 text-slate-700">{r.fullReason}</td>
                      <td className="px-5 py-3 text-right text-rose-600 font-semibold">{r.count.toLocaleString()}</td>
                      <td className="px-5 py-3">
                        <div className="h-2 bg-slate-100 rounded-full w-full overflow-hidden">
                          <div className="h-full bg-rose-500 rounded-full" style={{ width: `${(r.count / (topLossReasons[0]?.count || 1)) * 100}%` }} />
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
