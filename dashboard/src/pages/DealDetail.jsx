import { useEffect, useMemo, useState } from "react";
import { useParams, useLocation, Link } from "react-router-dom";

/* ----------------- 小UI ----------------- */
function Badge({ children, tone = "slate" }) {
  const tones = {
    slate: "bg-slate-100 text-slate-700",
    green: "bg-emerald-50 text-emerald-700",
    blue:  "bg-blue-50 text-blue-700",
    red:   "bg-rose-50 text-rose-700",
    amber: "bg-amber-50 text-amber-700",
  };
  return <span className={`inline-flex items-center h-7 px-3 rounded-full text-sm ${tones[tone]}`}>{children}</span>;
}

function Card({ title, subtitle, action, children, className = "" }) {
  return (
    <section className={`rounded-2xl bg-white border border-slate-200 p-5 ${className}`}>
      <div className="flex items-start justify-between gap-4">
        <div>
          {title && <h2 className="text-base font-semibold">{title}</h2>}
          {subtitle && <div className="text-xs text-slate-500 mt-0.5">{subtitle}</div>}
        </div>
        {action}
      </div>
      <div className="mt-4">{children}</div>
    </section>
  );
}

/* ----------------- 数据读取（与首页口径一致） ----------------- */
async function fetchXLSXRows(url, sheetName = "Opportunities") {
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

/* 把首页表格行（或 Excel+CSV 合并后的行）规范化为详情页需要的字段 */
function normalizeFromRow(row) {
  const p = Number.isFinite(row.p_win) ? row.p_win : Number.isFinite(row.winProbability) ? row.winProbability : 0.5;
  const stage =
    row.stage ||
    (p >= 0.7 ? "Negotiation" : p >= 0.4 ? "Proposal" : "Discovery");

  return {
    id: row.id,
    title: row.name || row.title || row["Opportunity Name"] || `${row.product || "Deal"} — ${row.region || ""}`.trim(),
    accountExecutive: row.owner || row["Owner"] || row["Opportunity Owner"] || "—",
    region: row.region || row["Account Region"] || "—",
    amount: Number.isFinite(toNum(row.acv)) ? Math.round(toNum(row.acv)) :
            Number.isFinite(toNum(row["Opportunity Line ACV USD"])) ? Math.round(toNum(row["Opportunity Line ACV USD"])) : 0,
    stage,
    winProbability: p,
    competitor: { name: row.competitor || row["Primary Competitor"] || "—", intel: "" },
    // 占位：这些可以之后替换为真实字段
    factors: [],
    recommendation: "—",
    activity: [],
    notes: { themes: [], decisionMakers: [] },
    updatedAt: new Date().toISOString(),
  };
}

/* 若直接打开详情页：从 Excel+CSV 合并找到该 id */
async function loadDealById(id) {
  const excelRows = await fetchXLSXRows("/api/RealDummyData.xlsx", "Opportunities");
  let csvRows = [];
  try { csvRows = await fetchCSVRows("/api/win_probabilities.csv"); } catch {}
  const csvMap = new Map();
  for (const r of csvRows) {
    const key = String(r["Opportunity ID"] || r.OpportunityID || r.id || "").trim();
    if (key) csvMap.set(key, r);
  }

  // 合并出我们需要的最小字段
  for (let i = 0; i < excelRows.length; i++) {
    const e = excelRows[i];
    const oppId = String(e["Opportunity ID"] || e.OpportunityID || e.id || "").trim();
    if (String(oppId) !== String(id)) continue;

    const csv = csvMap.get(String(oppId));
    const pRaw = csv ? toNum(csv.win_prob ?? csv.winProb) : NaN;

    const row = {
      id: oppId,
      name: e["Opportunity Name"],
      owner: e["Owner"] || e["Opportunity Owner"],
      region: e["Account Region"],
      acv: e["Opportunity Line ACV USD"],
      stage: e["Stage"],
      competitor: e["Primary Competitor"],
      p_win: Number.isFinite(pRaw) ? Math.max(0, Math.min(1, pRaw)) : undefined,
      product: e["Product Reporting Solution Area"],
    };
    return normalizeFromRow(row);
  }
  return null;
}

async function loadAllDeals() {
  const excelRows = await fetchXLSXRows("/api/RealDummyData.xlsx", "Opportunities");
  let csvRows = [];
  try { csvRows = await fetchCSVRows("/api/win_probabilities.csv"); } catch {}

  const csvMap = new Map();
  for (const r of csvRows) {
    const key = String(r["Opportunity ID"] || r.OpportunityID || r.id || "").trim();
    if (key) csvMap.set(key, r);
  }

  const deals = [];
  for (const e of excelRows) {
    const oppId = String(e["Opportunity ID"] || e.OpportunityID || e.id || "").trim();
    if (!oppId) continue;

    const csv = csvMap.get(oppId);
    const pRaw = csv ? toNum(csv.win_prob ?? csv.winProb) : NaN;

    const row = {
      id: oppId,
      name: e["Opportunity Name"],
      owner: e["Owner"] || e["Opportunity Owner"],
      region: e["Account Region"],
      acv: e["Opportunity Line ACV USD"],
      stage: e["Stage"],
      competitor: e["Primary Competitor"],
      product: e["Product Reporting Solution Area"],
      p_win: Number.isFinite(pRaw) ? Math.max(0, Math.min(1, pRaw)) : undefined,
    };

    deals.push(normalizeFromRow(row));
  }

  return deals;
}


/* ----------------- 页面 ----------------- */
export default function DealDetail() {
  const { id } = useParams();
  const location = useLocation();
  const [deal, setDeal] = useState(null);      // single deal
  const [deals, setDeals] = useState([]);  
  const [error, setError] = useState(null);
  const isOverview = !id && !location.state?.deal;

    useEffect(() => {
      let mounted = true;

      // 1) Coming from table with state (single deal)
      const fromState = location.state?.deal;
      if (fromState && mounted) {
        setDeal(normalizeFromRow(fromState));
        setDeals([]);
        return () => { mounted = false; };
      }

      // 2) No state → either /deal/:id or /deal
      (async () => {
        try {
          if (id) {
            const d = await loadDealById(id);
            if (!mounted) return;
            if (d) {
              setDeal(d);
              setDeals([]);
            } else {
              setError("Deal not found");
            }
          } else {
            // /deal → overview mode
            const all = await loadAllDeals();
            if (!mounted) return;
            setDeals(all);
            setDeal(null);
          }
        } catch (e) {
          if (mounted) setError("No Deal");
        }
    })();

    return () => { mounted = false; };
  }, [id, location.key]);


  const kpis = useMemo(() => {
    if (!deal) return [];
    return [
      { label: "Account Executive", value: deal.accountExecutive },
      { label: "Region", value: deal.region },
      { label: "Amount", value: `$${(deal.amount || 0).toLocaleString()}` },
      { label: "Stage", value: deal.stage, badge: true },
    ];
  }, [deal]);

  if (error) {
    return (
      <div className="p-5">
        <div className="text-rose-600 font-medium mb-2">{error}</div>
        <Link to="/" className="text-indigo-600 text-sm hover:underline">← Back to dashboard</Link>
      </div>
    );
  }
    if (isOverview) {
    if (!deals.length) {
      return <div className="text-sm text-slate-500 p-5">Loading…</div>;
    }

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">All Deals</h2>
          <Link
            to="/"
            className="text-sm text-indigo-600 hover:underline"
          >
            ← Back to dashboard
          </Link>
        </div>

        <div className="rounded-2xl bg-white border border-slate-200 p-4">
          <table className="min-w-full text-sm">
            <thead className="text-xs uppercase text-slate-500 border-b border-slate-200">
              <tr>
                <th className="py-2 text-left">Deal #</th>
                <th className="py-2 text-left">Opportunity</th>
                <th className="py-2 text-left">AE</th>
                <th className="py-2 text-left">Region</th>
                <th className="py-2 text-right">Amount</th>
                <th className="py-2 text-left">Stage</th>
                <th className="py-2 text-right">Win %</th>
              </tr>
            </thead>
            <tbody>
              {deals.map((d) => (
                <tr
                  key={d.id}
                  className="border-b border-slate-100 hover:bg-slate-50 cursor-pointer"
                  onClick={() =>
                    // go to single-deal view when you click a row
                    (window.location.href = `/deal/${d.id}`)
                  }
                >
                  {/* Deal number */}
                  <td className="py-2 pr-2">{d.id}</td>

                  {/* Human-readable name */}
                  <td className="py-2 pr-2">{d.title}</td>
                  <td className="py-2 pr-2">{d.accountExecutive}</td>
                  <td className="py-2 pr-2">{d.region}</td>
                  <td className="py-2 pr-2 text-right">
                    ${ (d.amount || 0).toLocaleString() }
                  </td>
                  <td className="py-2 pr-2">{d.stage}</td>
                  <td className="py-2 text-right">
                    {Math.round((d.winProbability || 0) * 100)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }
  if (!deal) return <div className="text-sm text-slate-500 p-5">Loading…</div>;

  return (
    <div className="space-y-6">
      {/* Title strip */}
      <div className="rounded-2xl bg-white border border-slate-200 p-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-xl md:text-2xl font-semibold">{deal.title}</h1>
            <div className="text-xs text-slate-500 mt-1">
              Updated {new Date(deal.updatedAt).toLocaleString()}
            </div>
          </div>
          <button
            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm hover:bg-slate-50"
            onClick={() => window.print()}
            title="Export to PPT/Excel (placeholder)"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" className="opacity-70">
              <path d="M5 20h14v-2H5v2Zm14-9h-4V3H9v8H5l7 7 7-7Z" fill="currentColor"></path>
            </svg>
            Export to PPT/Excel
          </button>
        </div>

        {/* KPI line */}
        <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {kpis.map((k) => (
            <div key={k.label} className="rounded-2xl border border-slate-200 p-4">
              <div className="text-xs uppercase tracking-wide text-slate-500">{k.label}</div>
              <div className="mt-1 text-lg font-semibold">
                {k.badge ? <Badge tone="amber">{k.value}</Badge> : k.value}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Top row: Activity (left) + AI score (right) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card
          title="Activity Timeline"
          subtitle="Recent activities (placeholder)"
          className="lg:col-span-2"
        >
          {/* 占位：如需接 Salesforce/日志，替换下面数组即可 */}
          <ul className="space-y-3">
            {[
              { label: "Initial discovery meeting", ago: "2 weeks ago", source: "CRM", color: "bg-sky-500" },
              { label: "Requirements call", ago: "1 week ago", source: "CRM", color: "bg-amber-500" },
              { label: "Pricing proposal sent", ago: "4 days ago", source: "Email", color: "bg-violet-500" },
            ].map((a, i) => (
              <li key={i} className="flex items-start gap-3">
                <span className={`mt-1 h-2.5 w-2.5 rounded-full ${a.color}`} />
                <div>
                  <div className="text-sm text-slate-800">{a.label}</div>
                  <div className="text-xs text-slate-500">{a.ago} · {a.source}</div>
                </div>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="AI Score vs Outcome" subtitle="AI prediction analysis">
          <div>
            <div className="text-sm font-medium">Win Probability</div>
            <div className="mt-2 h-2.5 w-full rounded-full bg-slate-100">
              <div
                className="h-2.5 rounded-full bg-emerald-500"
                style={{ width: `${Math.round((deal.winProbability ?? 0) * 100)}%` }}
              />
            </div>
            <div className="mt-2 text-right text-emerald-600 text-sm font-semibold">
              {Math.round((deal.winProbability ?? 0) * 100)}%
            </div>

            {/* 占位因子，可替换为真实解释 */}
            <div className="mt-4">
              <div className="text-sm font-medium">Key Factors:</div>
              <ul className="mt-2 space-y-1 text-sm text-slate-700">
                {deal.factors?.length ? deal.factors.map((f) => (
                  <li key={f.label}>• {f.label} ({f.impact > 0 ? "+" : ""}{Math.round(f.impact * 100)}%)</li>
                )) : (
                  <>
                    <li>• Region: {deal.region}</li>
                    {deal.competitor?.name && <li>• Competitive pressure: {deal.competitor.name}</li>}
                    <li>• Deal size: ${deal.amount.toLocaleString()}</li>
                  </>
                )}
              </ul>
            </div>

            <div className="mt-4 border-t pt-3 text-sm">
              <span className="font-medium">Recommendation:</span>{" "}
              {deal.recommendation && deal.recommendation !== "—"
                ? deal.recommendation
                : "Highlight ROI and implementation speed versus competitors."}
            </div>
          </div>
        </Card>
      </div>

      {/* Bottom row: Competitor Analysis (left) + Sales Notes (right) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Competitor Analysis" subtitle="NLP-based competitor detection">
          <div className="flex items-center gap-2 text-sm">
            <span className="font-medium">Primary Competitor</span>
            <Badge tone="red">{deal.competitor?.name || "—"}</Badge>
          </div>

          <blockquote className="mt-3 rounded-xl bg-indigo-50 text-indigo-900 p-4 text-sm leading-relaxed">
            “{deal.competitor?.intel || "No explicit competitor intel found. Continue discovery to confirm evaluation set."}”
          </blockquote>
        </Card>

        <Card title="Sales Notes Summary" subtitle="AI-extracted key phrases">
          <div className="text-sm">
            <div className="font-medium">Key Themes:</div>
            <div className="mt-2 flex flex-wrap gap-2">
              {(deal.notes?.themes?.length ? deal.notes.themes : [deal.region, deal.competitor?.name].filter(Boolean)).map((t) => (
                <span key={t} className="px-2.5 py-1 rounded-full text-xs bg-slate-100 text-slate-700">
                  {t}
                </span>
              ))}
            </div>

            <div className="mt-4 font-medium">Decision Makers:</div>
            <ul className="mt-2 space-y-1">
              {(deal.notes?.decisionMakers?.length ? deal.notes.decisionMakers : []).map((p) => (
                <li key={p.name} className="flex items-center justify-between">
                  <span>{p.name}</span>
                  <span className="text-slate-500 text-xs">{p.role}</span>
                </li>
              ))}
              {!deal.notes?.decisionMakers?.length && (
                <li className="text-xs text-slate-500">No contacts listed.</li>
              )}
            </ul>
          </div>
        </Card>
      </div>
    </div>
  );
}
