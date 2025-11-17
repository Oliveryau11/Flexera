import React, { useEffect, useMemo, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, ReferenceLine,
} from "recharts";
import { Link } from "react-router-dom";

/* ------------ 读取 Excel（Opportunities）与 CSV（win_probabilities）并合并 ------------ */
// 需要：npm i xlsx
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

/* ------------ UI atoms（保持你的风格） ------------ */
const Card = ({ title, subtitle, children, className = "" }) => (
  <div className={`bg-white rounded-2xl shadow-sm border border-slate-200 ${className}`}>
    {title && (<div className="px-4 pt-4 text-slate-900 font-semibold text-[15px]">{title}</div>)}
    {subtitle && (<div className="px-4 text-slate-500 text-xs">{subtitle}</div>)}
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

/* ------------ 配色 ------------ */
const COLORS = {
  blue: "#0056D2", blueLight: "#93C5FD", gray: "#E5E7EB", slate: "#6B7280",
  green: "#22C55E", orange: "#F59E0B", red: "#EF4444", purple: "#8B5CF6",
};

function toNum(v) { if (v === "" || v == null) return NaN; const n = Number(v); return Number.isFinite(n) ? n : NaN; }

/* ------------ 映射：以 Excel 为主，p_win 来自 CSV；增加 country 字段 ------------ */
function mapFromExcelPlusCSV(excelRow, idx, csvMap) {
  const id = excelRow["Opportunity ID"] || excelRow.OpportunityID || excelRow.id || `OPP-${idx + 1}`;
  const csv = csvMap.get(String(id)) || null;

  const region = excelRow["Account Region"] || excelRow.region || ["NA","EMEA","APAC","LATAM"][idx % 4];
  const product = excelRow["Product Reporting Solution Area"] || excelRow.product || ["FlexSecure","CloudVis","ITAM","Optima"][idx % 4];
  const competitor = excelRow["Primary Competitor"] || excelRow.competitor || ["CompetitorX","CompetitorY","CompetitorZ","CompetitorW"][idx % 4];

  // 新增：country（优先 Account Country/Billing Country）
  const country =
    excelRow["Account Country"] ||
    excelRow["Billing Country"] ||
    excelRow.country ||
    ["USA", "Germany", "UK", "Japan"][idx % 4];

  const acv = Number.isFinite(toNum(excelRow["Opportunity Line ACV USD"])) ? Math.round(toNum(excelRow["Opportunity Line ACV USD"])) : 0;

  // 真实结果：CRO Win/Stage
  const stage = (excelRow["Stage"] || "").toString();
  const cro = (excelRow["CRO Win"] || "").toString().toLowerCase();
  const isWon = cro === "yes" || /closed won/i.test(stage);
  const isLost = /closed lost/i.test(stage);
  const outcome = isWon ? "Won" : (isLost ? "Lost" : "Lost"); // 未关闭并入 Lost，保持现有图两类堆叠

  // 概率来自 CSV：win_prob；若无则 NaN
  let p_win = NaN;
  if (csv) {
    const pRaw = toNum(csv.win_prob ?? csv.winProb);
    if (Number.isFinite(pRaw)) p_win = Math.max(0, Math.min(1, pRaw));
  }

  return {
    id,
    region,
    product,
    competitor,
    country,           // ← 新增字段
    acv,
    p_win,
    outcome,
    loss_reason: isWon ? "Closed Won" : (isLost ? "Closed Lost" : "Open"),
    owner: excelRow["Owner"] || excelRow["Opportunity Owner"] || "—",
  };
}

/* ------------ 主组件 ------------ */
export default function App() {
  // 新增 country 过滤
  const [filters, setFilters] = useState({ region: "All", product: "All", competitor: "All", country: "All" });
  const [opps, setOpps] = useState([]);
  const [meta, setMeta] = useState({ updatedAt: null, source: "—" });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // 1) Excel 主数据
        const excelRows = await fetchXLSXRows("/api/RealDummyData.xlsx", "Opportunities");

        // 2) CSV 概率数据
        let csvRows = [];
        try { csvRows = await fetchCSVRows("/api/win_probabilities.csv"); } catch {}

        const csvMap = new Map();
        for (const r of csvRows) {
          const key = String(r["Opportunity ID"] || r.OpportunityID || r.id || "").trim();
          if (key) csvMap.set(key, r);
        }

        if (!cancelled) {
          const merged = excelRows.map((row, i) => mapFromExcelPlusCSV(row, i, csvMap));
          setOpps(merged);
          setMeta({ updatedAt: new Date().toISOString(), source: csvRows.length ? "Excel + Model CSV (merged)" : "Excel only" });
        }
      } catch (e) {
        if (!cancelled) {
          // fallback mock 保底
          const regions = ["NA", "EMEA", "APAC", "LATAM"];
          const comps = ["CompetitorX", "CompetitorY", "CompetitorZ"];
          const products = ["FlexSecure", "CloudVis", "ITAM", "Optima"];
          const countries = ["USA", "Germany", "UK", "Japan"];
          const sample = Array.from({ length: 60 }).map((_, i) => {
            const p = Math.random() * 0.9 + 0.05;
            return {
              id: `OPP-${1000 + i}`,
              region: regions[i % regions.length],
              product: products[i % products.length],
              competitor: comps[i % comps.length],
              country: countries[i % countries.length],
              acv: Math.round(10000 + Math.random() * 150000),
              p_win: p,
              outcome: Math.random() < p ? "Won" : "Lost",
              loss_reason: ["Price", "Features", "Support", "Timeline", "Other"][i % 5],
              owner: ["Dana", "Arun", "Julia", "Morgan"][i % 4],
            };
          });
          setOpps(sample);
          setMeta({ updatedAt: new Date().toISOString(), source: "Mock" });
        }
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // 选项集合
  const regions = useMemo(() => Array.from(new Set(opps.map(o=>o.region))).filter(Boolean), [opps]);
  const products = useMemo(() => Array.from(new Set(opps.map(o=>o.product))).filter(Boolean), [opps]);
  const competitors = useMemo(() => Array.from(new Set(opps.map(o=>o.competitor))).filter(Boolean), [opps]);
  const countries = useMemo(() => {
    const base = filters.region === "All" ? opps : opps.filter(o => o.region === filters.region);
    return Array.from(new Set(base.map(o => o.country))).filter(Boolean);
  }, [opps, filters.region]);

  // 联动过滤（含 country）
  const filtered = useMemo(() => {
    return opps.filter((o) =>
      (filters.region === "All" || o.region === filters.region) &&
      (filters.product === "All" || o.product === filters.product) &&
      (filters.competitor === "All" || o.competitor === filters.competitor) &&
      (filters.country === "All" || o.country === filters.country)
    );
  }, [opps, filters]);

  // KPI
  const winRate = useMemo(() => {
    if (filtered.length === 0) return 0;
    const won = filtered.filter((o) => o.outcome === "Won").length;
    return won / filtered.length;
  }, [filtered]);

  const avgP = useMemo(() => {
    if (filtered.length === 0) return 0;
    return filtered.reduce((s, o) => s + (Number.isFinite(o.p_win) ? o.p_win : 0), 0) / filtered.length;
  }, [filtered]);

  const highConfCount = useMemo(() => filtered.filter((o) => Number.isFinite(o.p_win) && o.p_win >= 0.7).length, [filtered]);

  // 图表数据（随筛选变化）
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

  // 原“Win/Loss Reasons”（仅 Lost），自动使用 Region+Country 过滤后的 filtered
  const reasons = useMemo(() => {
    const g = {};
    for (const o of filtered) {
      if (o.outcome !== "Lost") continue;
      const r = o.loss_reason || "Other";
      g[r] = (g[r] || 0) + 1;
    }
    return Object.entries(g).map(([name, value]) => ({ name, value }));
  }, [filtered]);

  // 新增：Loss Reasons by Country（在当前 Region 下，按 Country 展开 Lost 的原因堆叠）
  const reasonsByCountry = useMemo(() => {
    const source = filters.country === "All" ? filtered : filtered.filter(o => o.country === filters.country);
    const g = {};
    for (const o of source) {
      if (o.outcome !== "Lost") continue;
      const c = o.country || "—";
      g[c] ??= {};
      const r = o.loss_reason || "Other";
      g[c][r] = (g[c][r] || 0) + 1;
    }
    const keys = Array.from(new Set(Object.values(g).flatMap(obj => Object.keys(obj))));
    const rows = Object.entries(g).map(([country, obj]) => {
      const base = { country };
      for (const k of keys) base[k] = obj[k] || 0;
      return base;
    });
    return { rows, keys };
  }, [filtered, filters.country]);

  const dist = useMemo(() => {
    const bins = [0,0,0,0,0,0,0,0,0,0];
    for (const o of filtered) {
      const val = Number.isFinite(o.p_win) ? o.p_win : 0;
      const idx = Math.min(9, Math.floor(val * 10));
      bins[idx]++;
    }
    return bins.map((v,i)=> ({ bucket: `${(i*10)}-${(i+1)*10}%`, count: v }));
  }, [filtered]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900" style={{ margin: 0, width: '100vw', minHeight: '100vh' }}>
      <div className="mx-auto max-w-none px-5 pt-4 pb-1 text-xs text-slate-500">
        Last updated {new Date(meta.updatedAt || Date.now()).toLocaleString()} · Data source: {meta.source}
      </div>

      {/* Filter bar */}
      <div className="mx-auto max-w-none px-5">
        <Card className="p-0">
          <div className="p-3 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-7 gap-3">
            <select
              className="px-3 py-2 rounded-xl border border-slate-200 text-sm"
              value={filters.region}
              onChange={(e) => setFilters(f => ({ ...f, region: e.target.value, country: "All" }))}
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

            {/* 新增 Country */}
            <select
              className="px-3 py-2 rounded-xl border border-slate-200 text-sm"
              value={filters.country}
              onChange={(e) => setFilters(f => ({ ...f, country: e.target.value }))}
            >
              <option value="All">All Countries</option>
              {countries.map(c => <option key={c} value={c}>{c}</option>)}
            </select>

            <button
              className="px-3 py-2 rounded-xl text-sm border border-slate-200 hover:bg-slate-50"
              onClick={()=>setFilters({region:"All", product:"All", competitor:"All", country:"All"})}
            >
              Reset
            </button>

            <div className="hidden md:flex items-center text-xs text-slate-500 col-span-2">
              {filtered.length} opportunities
            </div>
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

        {/* Win/Loss Reasons（随 Region + Country 联动） */}
        {/* <Card
          title="Win/Loss Reasons"
          subtitle={`Lost deals only · Region: ${filters.region} · Country: ${filters.country}`}
        >
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={reasons} dataKey="value" nameKey="name" innerRadius={55} outerRadius={90} paddingAngle={2}>
                  {reasons.map((entry, idx) => (
                    <Cell key={idx} fill={[COLORS.blue, COLORS.orange, COLORS.green, COLORS.purple, "#94A3B8", "#EF4444"][idx % 6]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card> */}

        {/* 新增：按 Country 的原因堆叠条形（便于对比不同国家原因结构） */}
        <Card title="Most by Country" subtitle={`Region filter: ${filters.region} · Country filter: ${filters.country}`}>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={reasonsByCountry.rows} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={COLORS.gray} />
                <XAxis dataKey="country" tick={{ fontSize: 12 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                <Tooltip />
                {reasonsByCountry.keys.map((k, i) => (
                  <Bar key={k} dataKey={k} stackId="r" fill={["#2563EB","#10B981","#F59E0B","#8B5CF6","#EF4444","#64748B"][i % 6]} radius={[6,6,0,0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* 你原来的 p(Win) 分布图保持不变 */}
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
                  <th className="py-2 pr-4">Country</th>
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
                      <Link to={`/deal/${encodeURIComponent(o.id)}`} state={{ deal: o }} className="text-indigo-600 hover:underline">
                        {o.id}
                      </Link>
                    </td>
                    <td className="py-2 pr-4">{o.region}</td>
                    <td className="py-2 pr-4">{o.country}</td>
                    <td className="py-2 pr-4">{o.product}</td>
                    <td className="py-2 pr-4">{o.competitor}</td>
                    <td className="py-2 pr-4">${o.acv.toLocaleString()}</td>
                    <td className="py-2 pr-4">
                      <span className={`px-2 py-1 rounded-lg text-xs ${
                        Number.isFinite(o.p_win) && o.p_win>=0.7 ? "bg-green-50 text-green-700" :
                        Number.isFinite(o.p_win) && o.p_win>=0.4 ? "bg-amber-50 text-amber-700" :
                        "bg-red-50 text-red-700"
                      }`}>
                        {Number.isFinite(o.p_win) ? (o.p_win*100).toFixed(1) + "%" : "—"}
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
