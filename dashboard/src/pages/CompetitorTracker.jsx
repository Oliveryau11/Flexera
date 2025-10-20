import { useEffect, useMemo, useState } from "react";

/** --- Tiny UI atoms (local to this page) --- */
function Card({ title, subtitle, children, className = "" }) {
  return (
    <section className={`rounded-2xl bg-white border border-slate-200 ${className}`}>
      {(title || subtitle) && (
        <header className="px-5 pt-4">
          {title && <h2 className="text-[15px] font-semibold text-slate-900">{title}</h2>}
          {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
        </header>
      )}
      <div className="p-5 pt-4">{children}</div>
    </section>
  );
}
const KPI = ({ value, label }) => (
  <div className="rounded-2xl border border-slate-200 bg-white p-5">
    <div className="text-2xl font-bold text-slate-900">{value}</div>
    <div className="text-xs text-slate-500 mt-1">{label}</div>
  </div>
);
const Badge = ({ tone="slate", children }) => {
  const tones = {
    slate: "bg-slate-100 text-slate-700",
    blue: "bg-blue-50 text-blue-700",
    red: "bg-rose-50 text-rose-700",
    green: "bg-emerald-50 text-emerald-700",
    amber: "bg-amber-50 text-amber-700",
    cyan: "bg-cyan-50 text-cyan-700",
  };
  return <span className={`px-2.5 py-1 rounded-full text-xs ${tones[tone]}`}>{children}</span>;
};

/** --- Fake API + helpers --- */
async function fetchJSON(url, fallback) {
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error();
    return await r.json();
  } catch {
    return fallback;
  }
}

// miniature progress bar cell
function BarCell({ pct }) {
  const p = Math.max(0, Math.min(1, pct));
  return (
    <div className="w-28 h-2.5 rounded-full bg-slate-100">
      <div className="h-2.5 rounded-full bg-indigo-500" style={{ width: `${Math.round(p*100)}%` }} />
    </div>
  );
}

/** --- Page --- */
export default function CompetitorTracker() {
  const [filters, setFilters] = useState({
    region: "All Regions",
    product: "All Products",
    frequency: "All",
    quarter: "Q4 2024",
  });
  const [rows, setRows] = useState([]);

  useEffect(() => {
    // Try real backend then fallback to sample mirroring your mock
    fetchJSON("/api/competitors/table", null).then((data) => {
      if (data) setRows(data);
      else {
        setRows([
          { competitor: "Salesforce", region: "North America", productLine: "Enterprise",   freq: 23, winRate: 0.45, avgDeal: 125_000, trend: "up" },
          { competitor: "HubSpot",    region: "Europe",        productLine: "Professional", freq: 18, winRate: 0.62, avgDeal:  89_000, trend: "down" },
          { competitor: "Microsoft",  region: "Asia Pacific",  productLine: "Enterprise",   freq: 15, winRate: 0.71, avgDeal: 156_000, trend: "up" },
          { competitor: "Pipedrive",  region: "Latin America", productLine: "Starter",      freq: 12, winRate: 0.35, avgDeal:  45_000, trend: "stable" },
          { competitor: "Zoho",       region: "North America", productLine: "Professional", freq:  9, winRate: 0.38, avgDeal:  67_000, trend: "down" },
        ]);
      }
    });
  }, []);

  const regions = useMemo(() => ["All Regions", ...Array.from(new Set(rows.map(r => r.region)))], [rows]);
  const products = useMemo(() => ["All Products", ...Array.from(new Set(rows.map(r => r.productLine)))], [rows]);

  const filtered = useMemo(() => {
    return rows.filter(r =>
      (filters.region === "All Regions" || r.region === filters.region) &&
      (filters.product === "All Products" || r.productLine === filters.product)
    );
  }, [rows, filters]);

  // KPIs
  const kActive = useMemo(() => new Set(filtered.map(r => r.competitor)).size, [filtered]);
  const kTotalDeals = useMemo(() => filtered.reduce((s,r)=>s+r.freq, 0), [filtered]);
  const kWinRate = useMemo(() => {
    const wins = filtered.reduce((s,r)=> s + r.freq * r.winRate, 0);
    return kTotalDeals ? wins / kTotalDeals : 0;
  }, [filtered, kTotalDeals]);
  const kAvgDeal = useMemo(() => {
    if (!filtered.length) return 0;
    const total = filtered.reduce((s,r)=> s + r.avgDeal, 0);
    return total / filtered.length;
  }, [filtered]);

  // Insights (sample)
  const insights = [
    {
      competitor: "Salesforce",
      text: "Strong in enterprise deals but losing ground on implementation speed. Focus on our faster deployment advantage.",
    },
  ];

  return (
    <div className="space-y-6">
      {/* Filters row */}
      <Card title="Competitor Analysis Filters">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <Select
            label="Region"
            value={filters.region}
            onChange={(v)=>setFilters(f=>({...f, region:v}))}
            options={regions}
          />
          <Select
            label="Product"
            value={filters.product}
            onChange={(v)=>setFilters(f=>({...f, product:v}))}
            options={products}
          />
          <Select
            label="Frequency"
            value={filters.frequency}
            onChange={(v)=>setFilters(f=>({...f, frequency:v}))}
            options={["All","High","Medium","Low"]}
          />
          <Select
            label="Quarter"
            value={filters.quarter}
            onChange={(v)=>setFilters(f=>({...f, quarter:v}))}
            options={["Q1 2024","Q2 2024","Q3 2024","Q4 2024"]}
          />
        </div>
      </Card>

      {/* KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        <KPI value={kActive} label="Active Competitors" />
        <KPI value={kTotalDeals} label="Total Competitive Deals" />
        <KPI value={`${Math.round(kWinRate*100)}%`} label="Overall Win Rate vs Competitors" />
        <KPI value={`$${Math.round(kAvgDeal/1000)}K`} label="Avg Deal Size (Competitive)" />
      </div>

      {/* Table */}
      <Card title="Competitor Performance Analysis" subtitle="Head-to-head performance metrics">
        <div className="overflow-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="text-left text-slate-500 border-b border-slate-200">
                <th className="py-2 pr-4">Competitor</th>
                <th className="py-2 pr-4">Region</th>
                <th className="py-2 pr-4">Product Line</th>
                <th className="py-2 pr-4">Frequency</th>
                <th className="py-2 pr-4">Win Rate vs Them</th>
                <th className="py-2 pr-4">Avg Deal Size</th>
                <th className="py-2 pr-4">Trend</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r, i) => (
                <tr key={i} className={`border-b border-slate-100 ${i%2 ? "" : "bg-slate-50/40"}`}>
                  <td className="py-2 pr-4 font-medium">{r.competitor}</td>
                  <td className="py-2 pr-4">{r.region}</td>
                  <td className="py-2 pr-4">{r.productLine}</td>
                  <td className="py-2 pr-4">{r.freq}</td>
                  <td className="py-2 pr-4">
                    <div className="flex items-center gap-2">
                      <BarCell pct={r.winRate} />
                      <span className="tabular-nums text-slate-700">{Math.round(r.winRate*100)}%</span>
                    </div>
                  </td>
                  <td className="py-2 pr-4">${r.avgDeal.toLocaleString()}</td>
                  <td className="py-2 pr-4">
                    {r.trend === "up" && <Badge tone="blue">up</Badge>}
                    {r.trend === "down" && <Badge tone="red">down</Badge>}
                    {r.trend === "stable" && <Badge tone="cyan">stable</Badge>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Insights */}
      <Card title="Competitive Intelligence Insights" subtitle="AI-generated insights from deal notes and market data">
        <div className="space-y-3">
          {insights.map((x, i) => (
            <div key={i} className="rounded-xl border border-indigo-100 bg-indigo-50 p-4">
              <div className="flex items-center gap-2 text-indigo-900">
                <span className="font-semibold">{x.competitor}</span>
              </div>
              <p className="mt-1 text-sm text-indigo-900/90">{x.text}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

/** Select with label */
function Select({ label, value, onChange, options }) {
  return (
    <label className="text-sm">
      <div className="text-slate-600 text-xs mb-1">{label}</div>
      <div className="relative">
        <select
          value={value}
          onChange={(e)=>onChange(e.target.value)}
          className="w-full appearance-none rounded-xl border border-slate-200 px-3 py-2 text-sm bg-white pr-8"
        >
          {options.map(opt => <option key={opt} value={opt}>{opt}</option>)}
        </select>
        <svg className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2" width="16" height="16" viewBox="0 0 24 24">
          <path fill="currentColor" d="m7 10l5 5l5-5H7Z"/>
        </svg>
      </div>
    </label>
  );
}
