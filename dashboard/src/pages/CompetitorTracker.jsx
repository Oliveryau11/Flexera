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

// miniature progress bar cell
function BarCell({ pct }) {
  const p = Math.max(0, Math.min(1, pct || 0));
  return (
    <div className="w-28 h-2.5 rounded-full bg-slate-100">
      <div className="h-2.5 rounded-full bg-indigo-500" style={{ width: `${Math.round(p*100)}%` }} />
    </div>
  );
}

/** -------------------- Data helpers (Excel + CSV) -------------------- */
async function fetchXLSXRows(url, sheetName="Opportunities") {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch ${url}`);
  const buf = await res.arrayBuffer();
  const XLSX = await import("xlsx");
  const wb = XLSX.read(buf, { type: "array" });
  const ws = wb.Sheets[sheetName];
  if (!ws) return [];
  return XLSX.utils.sheet_to_json(ws, { defval: null });
}
async function fetchCSVRows(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch ${url}`);
  const text = await res.text();
  const lines = text.trim().split(/\r?\n/);
  if (lines.length <= 1) return [];
  const headers = lines[0].split(",").map(h => h.trim());
  return lines.slice(1).map((line) => {
    const cols = line.split(",").map(c => c.trim());
    const row = {};
    headers.forEach((h, i) => (row[h] = cols[i]));
    return row;
  });
}
const toNum = (v) => {
  if (v === "" || v == null) return NaN;
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
};
const sanitizeCompetitor = (name) => {
  const s = String(name || "").trim().toLowerCase();
  if (!s) return "";
  const bad = new Set([
    "unknown","unknown competitor","no competitor","don't know","dont know","n/a","na","none","-","—"
  ]);
  return bad.has(s) ? "" : String(name).trim();
};
const modeOf = (arr, key) => {
  const c = {};
  for (const x of arr) {
    const k = (x[key] || "—").toString();
    c[k] = (c[k] || 0) + 1;
  }
  return Object.entries(c).sort((a,b)=>b[1]-a[1])[0]?.[0] || "—";
};
const dealOutcome = (stage, cro) => {
  const s = (stage || "").toString().toLowerCase();
  const winFlag = (cro || "").toString().toLowerCase();
  if (winFlag === "yes" || /closed won/.test(s)) return "Won";
  if (/closed lost/.test(s)) return "Lost";
  return "Open";
};
const deriveQuarter = (dstr) => {
  if (!dstr) return "—";
  const d = new Date(dstr);
  if (isNaN(d)) return "—";
  const q = Math.floor(d.getMonth()/3)+1;
  return `Q${q} ${d.getFullYear()}`;
};

/** -------------------- Page -------------------- */
export default function CompetitorTracker() {
  const [meta, setMeta] = useState({ updatedAt: null, source: "—" });
  const [deals, setDeals] = useState([]); // normalized per-opportunity rows

  const [filters, setFilters] = useState({
    region: "All",
    product: "All",
    frequency: "All",
    quarter: "All",
  });

  // Load Excel + CSV, normalize to minimal fields for competitor analysis
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const excel = await fetchXLSXRows("/api/RealDummyData.xlsx", "Opportunities");
        let csvRows = [];
        try { csvRows = await fetchCSVRows("/api/win_probabilities.csv"); } catch {}
        const csvMap = new Map();
        for (const r of csvRows) {
          const key = String(r["Opportunity ID"] || r.OpportunityID || r.id || "").trim();
          if (key) csvMap.set(key, r);
        }

        const merged = excel.map((e, i) => {
          const id = String(e["Opportunity ID"] || e.OpportunityID || e.id || `OPP-${i+1}`).trim();
          const csv = csvMap.get(id);
          const p = csv ? toNum(csv.win_prob ?? csv.winProb) : NaN;

          const region  = e["Account Region"] || e.region || "—";
          const product = e["Product Reporting Solution Area"] || e.product || "—";
          const competitorRaw = e["Primary Competitor"] || e.competitor || "";
          const competitor = sanitizeCompetitor(competitorRaw);
          const acv = Number.isFinite(toNum(e["Opportunity Line ACV USD"])) ? Math.round(toNum(e["Opportunity Line ACV USD"])) : 0;
          const stage = e["Stage"] || "";
          const cro   = e["CRO Win"] || "";
          const outcome = dealOutcome(stage, cro);
          const quarter = deriveQuarter(e["Close Date"] || e["Expected Close Date"]);
          const p_win = Number.isFinite(p) ? Math.max(0, Math.min(1, p)) : undefined;

          return { id, region, product, competitor, acv, stage, outcome, quarter, p_win };
        });

        if (!cancelled) {
          setDeals(merged);
          setMeta({
            updatedAt: new Date().toISOString(),
            source: csvRows.length ? "Excel + Model CSV (merged)" : "Excel only",
          });
        }
      } catch (e) {
        if (!cancelled) {
          setDeals([]);
          setMeta({ updatedAt: new Date().toISOString(), source: "Load error" });
        }
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Filter options
  const regions = useMemo(() => ["All", ...Array.from(new Set(deals.map(d => d.region))).filter(Boolean)], [deals]);
  const products = useMemo(() => ["All", ...Array.from(new Set(deals.map(d => d.product))).filter(Boolean)], [deals]);
  const quarters = useMemo(() => ["All", ...Array.from(new Set(deals.map(d => d.quarter))).filter(q => q && q !== "—")], [deals]);

  // Apply filters (only competitive deals with valid competitor)
  const filteredDeals = useMemo(() => {
    return deals.filter(d =>
      d.competitor && // 已清洗
      (filters.region  === "All" || d.region  === filters.region) &&
      (filters.product === "All" || d.product === filters.product) &&
      (filters.quarter === "All" || d.quarter === filters.quarter)
    );
  }, [deals, filters]);

  // Aggregate by competitor
  const tableAll = useMemo(() => {
    const g = new Map();
    for (const d of filteredDeals) {
      if (!g.has(d.competitor)) g.set(d.competitor, []);
      g.get(d.competitor).push(d);
    }
    const rows = [];
    g.forEach((list, comp) => {
      const freq = list.length;
      const closed = list.filter(x => x.outcome !== "Open");
      const winRate = closed.length ? (closed.filter(x => x.outcome === "Won").length / closed.length) : 0;
      const avgDeal = list.length ? Math.round(list.reduce((s,x)=>s+(x.acv||0),0)/list.length) : 0;

      rows.push({
        competitor: comp,
        region: modeOf(list, "region"),
        productLine: modeOf(list, "product"),
        freq,
        winRate,
        avgDeal,
        // 简单趋势：相对整体胜率
        _closed: closed.length,
      });
    });
    return rows.sort((a,b)=> b.freq - a.freq);
  }, [filteredDeals]);

  // Frequency filter (High ≥10; Medium 4–9; Low 1–3)
  const table = useMemo(() => {
    if (filters.frequency === "All") return tableAll;
    const inBucket = (n) =>
      filters.frequency === "High" ? n >= 10 :
      filters.frequency === "Medium" ? (n >= 4 && n <= 9) :
      (n >= 1 && n <= 3);
    return tableAll.filter(r => inBucket(r.freq));
  }, [tableAll, filters.frequency]);

  // Overall win rate（仅对 competitive 且已闭合的）
  const overallWinRate = useMemo(() => {
    const closed = filteredDeals.filter(d => d.outcome !== "Open");
    return closed.length ? (closed.filter(d => d.outcome === "Won").length / closed.length) : 0;
  }, [filteredDeals]);

  // 计算趋势标签
  const tableWithTrend = useMemo(() => {
    return table.map(r => {
      const trend = r.winRate > overallWinRate + 0.05 ? "up"
                  : r.winRate < overallWinRate - 0.05 ? "down"
                  : "stable";
      return { ...r, trend };
    });
  }, [table, overallWinRate]);

  // KPIs
  const kActive = useMemo(() => new Set(tableAll.map(r => r.competitor)).size, [tableAll]);
  const kTotalDeals = useMemo(() => filteredDeals.length, [filteredDeals]);
  const kAvgDeal = useMemo(() => {
    if (!tableAll.length) return 0;
    return Math.round(tableAll.reduce((s,r)=>s+r.avgDeal,0)/tableAll.length);
  }, [tableAll]);

  // Insights 占位（可对接真实 NLP）
  const insights = tableWithTrend.length ? [{
    competitor: tableWithTrend[0].competitor,
    text: `Strong in ${tableWithTrend[0].productLine.toLowerCase()} deals, trend "${tableWithTrend[0].trend}". Emphasize faster deployment and ROI in ${tableWithTrend[0].region}.`,
  }] : [];

  return (
    <div className="space-y-6">
      {/* Filters row */}
      <Card title="Competitor Analysis Filters"
            subtitle={`Last updated ${new Date(meta.updatedAt || Date.now()).toLocaleString()} · ${meta.source}`}>
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
            options={quarters}
          />
        </div>
      </Card>

      {/* KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        <KPI value={kActive} label="Active Competitors" />
        <KPI value={kTotalDeals} label="Total Competitive Deals" />
        <KPI value={`${Math.round(overallWinRate*100)}%`} label="Overall Win Rate vs Competitors" />
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
              {tableWithTrend.map((r, i) => (
                <tr key={r.competitor} className={`border-b border-slate-100 ${i%2 ? "" : "bg-slate-50/40"}`}>
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
              {!tableWithTrend.length && (
                <tr><td className="py-4 text-slate-500" colSpan={7}>No competitive deals under current filters.</td></tr>
              )}
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
          {!insights.length && <div className="text-sm text-slate-500">No insights available.</div>}
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
