import React, { useEffect, useMemo, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  ReferenceLine,
} from "recharts";
import { Link } from "react-router-dom";



// --- Helper UI atoms (Tailwind only) ---
const Card = ({ title, subtitle, children, className = "" }) => (
  <div className={`bg-white rounded-2xl shadow-sm border border-slate-200 ${className}`}>
    {title && (
      <div className="px-4 pt-4 text-slate-900 font-semibold text-[15px]">{title}</div>
    )}
    {subtitle && (
      <div className="px-4 text-slate-500 text-xs">{subtitle}</div>
    )}
    <div className="p-4">{children}</div>
  </div>
);


const KPICard = ({ value, label, suffix = "", accent = "text-blue-600" }) => (
  <Card className="h-[120px] flex items-center justify-center">
    <div className="text-center">
      <div className={`text-3xl font-bold ${accent}`}>{value}{suffix}</div>
      <div className="text-xs text-slate-500 mt-1">{label}</div>
    </div>
  </Card>
);



// Build an API URL that respects the Vite base path (e.g., GitHub Pages)
const apiUrl = (path) => new URL(path.replace(/^\//, ""), import.meta.env.BASE_URL || "/");

// --- Mock fetch with graceful fallback ---
async function fetchJSON(path, fallback) {
  try {
    const r = await fetch(apiUrl(path));
    if (!r.ok) throw new Error("bad status");
    return await r.json();
  } catch {
    return fallback;
  }
}

// --- Color tokens ---
const COLORS = {
  blue: "#0056D2",
  blueLight: "#93C5FD",
  gray: "#E5E7EB",
  slate: "#6B7280",
  green: "#22C55E",
  orange: "#F59E0B",
  red: "#EF4444",
  purple: "#8B5CF6",
};

// Derived dimensions for enriching model rows into dashboard-friendly fields
const DERIVED_DIMENSIONS = {
  regions: ["North America", "Europe", "Asia Pacific", "Latin America"],
  products: ["Platform", "ITAM", "Cloud", "Security"],
  competitors: ["Competitor A", "Competitor B", "Competitor C", "Competitor D"],
};

function mapWinProbabilityRow(row, idx) {
  const region = DERIVED_DIMENSIONS.regions[idx % DERIVED_DIMENSIONS.regions.length];
  const product = DERIVED_DIMENSIONS.products[idx % DERIVED_DIMENSIONS.products.length];
  const competitor = DERIVED_DIMENSIONS.competitors[idx % DERIVED_DIMENSIONS.competitors.length];
  const probability = Math.max(0, Math.min(1, Number(row.winProb ?? row.win_prob ?? 0)));

  return {
    id: row.id || row["Opportunity ID"] || `OPP-${idx + 1}`,
    region,
    product,
    competitor,
    acv: Math.round(Number(row.amount ?? 0)),
    p_win: probability,
    outcome: row.predictedWin || row.pred_at_prec_thr === "1" ? "Won" : "Lost",
    loss_reason: row.predictedWin ? "Model: expected win" : "Model: below precision threshold",
    owner: row.owner || row.stage || "—",
  };
}

// --- Main App ---
export default function App() {
  const [filters, setFilters] = useState({ region: "All", product: "All", competitor: "All" });
  const [opps, setOpps] = useState([]);
  const [meta, setMeta] = useState({ updatedAt: null, source: "Fallback" });

  useEffect(() => {
    let cancelled = false;

    async function loadData() {
      const modelRows = await fetchJSON("api/win_probabilities.json", null);
      if (!cancelled && Array.isArray(modelRows) && modelRows.length) {
        setOpps(modelRows.map(mapWinProbabilityRow));
        setMeta((prev) => ({
          ...prev,
          updatedAt: new Date().toISOString(),
          source: "Model predictions",
        }));
        return;
      }

      const oppData = await fetchJSON("api/opportunities", []);
      if (!cancelled) setOpps(oppData);
    }

    loadData();

    fetchJSON("api/meta", { updatedAt: new Date().toISOString() }).then((metaData) => {
      if (!cancelled) setMeta((prev) => ({ ...prev, ...metaData }));
    });

    return () => {
      cancelled = true;
    };
  }, []);

  // Fallback mock data if backend not present
  useEffect(() => {
    if (opps.length === 0) {
      const regions = ["NA", "EMEA", "APAC", "LATAM"];
      const comps = ["CompetitorX", "CompetitorY", "CompetitorZ"];
      const products = ["FlexSecure", "CloudVis", "ITAM", "Optima"];
      const sample = Array.from({ length: 80 }).map((_, i) => {
        const p = Math.random() * 0.9 + 0.05;
        return {
          id: `OPP-${1000 + i}`,
          region: regions[i % regions.length],
          product: products[i % products.length],
          competitor: comps[i % comps.length],
          acv: Math.round(10000 + Math.random() * 150000),
          p_win: p,
          outcome: Math.random() < p ? "Won" : "Lost",
          loss_reason: ["Price", "Features", "Support", "Timeline", "Other"][i % 5],
          owner: ["Dana", "Arun", "Julia", "Morgan"][i % 4],
        };
      });
      setOpps(sample);
    }
  }, [opps]);

  const filtered = useMemo(() => {
    return opps.filter((o) =>
      (filters.region === "All" || o.region === filters.region) &&
      (filters.product === "All" || o.product === filters.product) &&
      (filters.competitor === "All" || o.competitor === filters.competitor)
    );
  }, [opps, filters]);

  // KPI calculations
  const winRate = useMemo(() => {
    if (filtered.length === 0) return 0;
    const won = filtered.filter((o) => o.outcome === "Won").length;
    return won / filtered.length;
  }, [filtered]);

  const avgP = useMemo(() => {
    if (filtered.length === 0) return 0;
    return filtered.reduce((s, o) => s + o.p_win, 0) / filtered.length;
  }, [filtered]);

  const highConfCount = useMemo(() => filtered.filter((o) => o.p_win >= 0.7).length, [filtered]);

  // Charts data
  const byRegion = useMemo(() => {
    const g = {};
    for (const o of filtered) {
      g[o.region] ??= { region: o.region, Won: 0, Lost: 0 };
      g[o.region][o.outcome] += 1;
    }
    return Object.values(g);
  }, [filtered]);

  const byCompetitor = useMemo(() => {
    const g = {};
    for (const o of filtered) {
      g[o.competitor] ??= { competitor: o.competitor, Won: 0, Lost: 0 };
      g[o.competitor][o.outcome] += 1;
    }
    return Object.values(g).sort((a,b)=> (b.Won+b.Lost)-(a.Won+a.Lost));
  }, [filtered]);

  const reasons = useMemo(() => {
    const g = {};
    for (const o of filtered) {
      if (o.outcome !== "Lost") continue;
      g[o.loss_reason] = (g[o.loss_reason] || 0) + 1;
    }
    return Object.entries(g).map(([name, value]) => ({ name, value }));
  }, [filtered]);

  const dist = useMemo(() => {
    const bins = [0,0,0,0,0,0,0,0,0,0];
    for (const o of filtered) {
      const idx = Math.min(9, Math.floor(o.p_win * 10));
      bins[idx]++;
    }
    return bins.map((v,i)=> ({ bucket: `${(i*10)}-${(i+1)*10}%`, count: v }));
  }, [filtered]);

  const regions = useMemo(() => Array.from(new Set(opps.map(o=>o.region))), [opps]);
  const products = useMemo(() => Array.from(new Set(opps.map(o=>o.product))), [opps]);
  const competitors = useMemo(() => Array.from(new Set(opps.map(o=>o.competitor))), [opps]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900" style={{ margin: 0, width: '100vw', minHeight: '100vh' }}>
      <div className="mx-auto max-w-none px-5 pt-4 pb-1 text-xs text-slate-500 flex flex-col sm:flex-row sm:items-center sm:gap-2">
        <span>Last updated {new Date(meta.updatedAt || Date.now()).toLocaleString()}</span>
        <span className="hidden sm:inline">·</span>
        <span>Data source: {meta.source || "Model fallback"}</span>
      </div>
      {/* Filter bar */}
      <div className="mx-auto max-w-none px-5">
        <Card className="p-0">
          <div className="p-3 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
            <select
              className="px-3 py-2 rounded-xl border border-slate-200 text-sm"
              value={filters.region}
              onChange={(e) => setFilters(f => ({ ...f, region: e.target.value }))}
            >
              <option value="All">All Regions</option>
              {regions.map(r => <option key={r} value={r}>{r}</option>)}
            </select>

            <select
              className="px-3 py-2 rounded-xl border border-slate-200 text-sm"
              value={filters.product}
              onChange={(e) => setFilters(f => ({ ...f, product: e.target.value }))}
            >
              <option value="All">All Products</option>
              {products.map(p => <option key={p} value={p}>{p}</option>)}
            </select>

            <select
              className="px-3 py-2 rounded-xl border border-slate-200 text-sm"
              value={filters.competitor}
              onChange={(e) => setFilters(f => ({ ...f, competitor: e.target.value }))}
            >
              <option value="All">All Competitors</option>
              {competitors.map(c => <option key={c} value={c}>{c}</option>)}
            </select>

            <button className="px-3 py-2 rounded-xl text-sm border border-slate-200 hover:bg-slate-50" onClick={()=>setFilters({region:"All", product:"All", competitor:"All"})}>Reset</button>
            <div className="hidden md:flex items-center text-xs text-slate-500 col-span-2">{filtered.length} opportunities</div>
          </div>
        </Card>
      </div>

      {/* KPI row */}
      <div className="mx-auto max-w-none px-5 mt-4 grid grid-cols-1 sm:grid-cols-3 gap-4">
        <KPICard value={(winRate*100).toFixed(1) + "%"} label="Win Rate" />
        <KPICard value={(avgP*100).toFixed(1) + "%"} label="Avg p(Win)" />
        <KPICard value={highConfCount} label="High-Confidence Deals (≥0.7)" />
      </div>

      {/* Charts grid */}
      <div className="mx-auto max-w-none px-5 mt-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card title="Win Rate by Region" subtitle="Won vs Lost by region">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={byRegion} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={COLORS.gray} />
                <XAxis dataKey="region" tick={{ fontSize: 12 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="Won" stackId="a" fill={COLORS.blue} radius={[6,6,0,0]} />
                <Bar dataKey="Lost" stackId="a" fill="#D1D5DB" radius={[6,6,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Top Competitors" subtitle="Deal outcomes by competitor">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={byCompetitor} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={COLORS.gray} />
                <XAxis dataKey="competitor" tick={{ fontSize: 12 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="Won" stackId="b" fill={COLORS.blue} radius={[6,6,0,0]} />
                <Bar dataKey="Lost" stackId="b" fill="#D1D5DB" radius={[6,6,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Win/Loss Reasons" subtitle="Top loss drivers (lost deals only)">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={reasons} dataKey="value" nameKey="name" innerRadius={55} outerRadius={90} paddingAngle={2}>
                  {reasons.map((entry, idx) => (
                    <Cell key={idx} fill={[COLORS.blue, COLORS.orange, COLORS.green, COLORS.purple, "#94A3B8"][idx % 5]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="p(Win) Distribution" subtitle="Model confidence buckets">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={dist} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={COLORS.gray} />
                <XAxis dataKey="bucket" tick={{ fontSize: 11 }} angle={-12} height={50} />
                <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                <Tooltip />
                <ReferenceLine x="40-50%" stroke={COLORS.orange} />
                <ReferenceLine x="70-80%" stroke={COLORS.green} />
                <Bar dataKey="count" fill={COLORS.blue} radius={[6,6,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Table */}
      <div className="mx-auto max-w-none px-5 mt-4 mb-10">
        <Card title="Deals" subtitle="Interactive table (filtered)">
          <div className="overflow-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left text-slate-500 border-b border-slate-200">
                  <th className="py-2 pr-4">Deal</th>
                  <th className="py-2 pr-4">Region</th>
                  <th className="py-2 pr-4">Product</th>
                  <th className="py-2 pr-4">Competitor</th>
                  <th className="py-2 pr-4">ACV</th>
                  <th className="py-2 pr-4">p(Win)</th>
                  <th className="py-2 pr-4">Outcome</th>
                </tr>
              </thead>
              <tbody>
                {filtered.slice(0, 50).map((o) => (
                  <tr key={o.id} className="border-b border-slate-100 hover:bg-slate-50">
                    <td className="py-2 pr-4">
                      <Link
                        to={`/deal/${encodeURIComponent(o.id)}`}
                        state={{ deal: o }}
                        className="text-indigo-600 hover:underline"
                      >
                        {o.id}
                      </Link>
                    </td>
                    <td className="py-2 pr-4">{o.region}</td>
                    <td className="py-2 pr-4">{o.product}</td>
                    <td className="py-2 pr-4">{o.competitor}</td>
                    <td className="py-2 pr-4">${o.acv.toLocaleString()}</td>
                    <td className="py-2 pr-4">
                      <span className={`px-2 py-1 rounded-lg text-xs ${o.p_win>=0.7?"bg-green-50 text-green-700": o.p_win>=0.4?"bg-amber-50 text-amber-700":"bg-red-50 text-red-700"}`}>
                        {(o.p_win*100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-2 pr-4">{o.outcome}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}