import { useEffect, useMemo, useState } from "react";
import { useParams, useLocation, Link } from "react-router-dom";

/* ------------ Data Fetching ------------ */
async function fetchCSVRows(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch ${url}`);
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

function transformDeal(row, idx) {
  const id = row["Opportunity ID"] || `OPP-${idx + 1}`;
  const name = row["Opportunity Name"] || "Unnamed Opportunity";
  const stage = row["Stage"] || "Unknown";
  const owner = row["Owner"] || "Unassigned";
  const pWin = toNum(row["win_prob"]);
  const predAtThr = row["pred_at_prec_thr"] === "1";
  
  const isWon = /closed won/i.test(stage);
  const isLost = /closed lost/i.test(stage);
  
  let oppType = "Other";
  if (/renewal/i.test(name)) oppType = "Renewal";
  else if (/new|expansion/i.test(name)) oppType = "New Business";
  else if (/upsell/i.test(name)) oppType = "Upsell";

  let confidence = "Low";
  if (Number.isFinite(pWin)) {
    if (pWin >= 0.9) confidence = "High";
    else if (pWin >= 0.7) confidence = "Medium";
  }

  return {
    id, name, fullName: name, stage, owner,
    p_win: Number.isFinite(pWin) ? pWin : null,
    pred_win: predAtThr, oppType, confidence,
    status: isWon ? "Won" : (isLost ? "Lost" : "Open"),
  };
}

export default function DealDetail() {
  const { id } = useParams();
  const location = useLocation();
  const [deal, setDeal] = useState(null);
  const [allDeals, setAllDeals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [filter, setFilter] = useState("All");

  const isOverview = !id && !location.state?.deal;

  useEffect(() => {
    let mounted = true;
    const fromState = location.state?.deal;
    if (fromState && mounted) {
      setDeal(fromState);
      setLoading(false);
      return () => { mounted = false; };
    }

    (async () => {
      try {
        const csvRows = await fetchCSVRows("/api/win_probabilities.csv");
        const seen = new Set();
        const deals = [];
        for (let i = 0; i < csvRows.length; i++) {
          const row = csvRows[i];
          const oppId = row["Opportunity ID"];
          if (oppId && !seen.has(oppId)) {
            seen.add(oppId);
            deals.push(transformDeal(row, i));
          }
        }
        deals.sort((a, b) => (b.p_win ?? 0) - (a.p_win ?? 0));
        
        if (!mounted) return;
        if (id) {
          const found = deals.find(d => d.id === id);
          if (found) setDeal(found);
          else setError("Deal not found");
        } else {
          setAllDeals(deals);
        }
        setLoading(false);
      } catch (e) {
        if (mounted) { setError("Failed to load"); setLoading(false); }
      }
    })();
    return () => { mounted = false; };
  }, [id, location.key, location.state?.deal]);

  const filtered = useMemo(() => {
    return allDeals.filter(d => {
      if (searchTerm && !d.name.toLowerCase().includes(searchTerm.toLowerCase()) && 
          !d.id.toLowerCase().includes(searchTerm.toLowerCase())) return false;
      if (filter !== "All" && d.confidence !== filter) return false;
      return true;
    });
  }, [allDeals, searchTerm, filter]);

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-slate-50 text-slate-800 p-8">
        <div className="max-w-[1400px] mx-auto">
          <div className="text-rose-600 mb-4">{error}</div>
          <Link to="/" className="text-blue-600 text-sm font-medium hover:underline">← Back</Link>
        </div>
      </div>
    );
  }

  if (isOverview) {
    return (
      <div className="min-h-screen bg-slate-50 text-slate-800">
        <div className="max-w-[1400px] mx-auto px-8 py-8">
          <div className="flex items-end justify-between mb-8">
            <div>
              <div className="text-xs text-slate-400 uppercase tracking-[0.2em] mb-1">Browse</div>
              <h1 className="text-2xl font-semibold text-slate-800">All Opportunities</h1>
            </div>
            <div className="text-sm text-slate-500 font-medium">{filtered.length.toLocaleString()} deals</div>
          </div>

          <div className="bg-white rounded-xl border border-slate-200 p-4 mb-6 shadow-sm">
            <div className="flex gap-4">
              <input type="text" placeholder="Search…" value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="flex-1 bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100 placeholder:text-slate-400" />
              <select value={filter} onChange={(e) => setFilter(e.target.value)}
                className="bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-400">
                <option value="All">All confidence</option>
                <option value="High">High</option>
                <option value="Medium">Medium</option>
                <option value="Low">Low</option>
              </select>
            </div>
          </div>

          <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-slate-400 text-xs uppercase tracking-wider border-b border-slate-100 bg-slate-50">
                  <th className="px-5 py-3 font-medium">Opportunity</th>
                  <th className="px-5 py-3 font-medium">Type</th>
                  <th className="px-5 py-3 font-medium">Owner</th>
                  <th className="px-5 py-3 font-medium text-right">Probability</th>
                  <th className="px-5 py-3 font-medium w-20"></th>
                </tr>
              </thead>
              <tbody>
                {filtered.slice(0, 100).map((d, i) => (
                  <tr key={d.id} className={`border-b border-slate-100 hover:bg-blue-50/50 transition-colors ${i % 2 === 0 ? 'bg-slate-50/30' : ''}`}>
                    <td className="px-5 py-3">
                      <div className="text-slate-700 font-medium">{d.name.substring(0, 50)}{d.name.length > 50 ? "…" : ""}</div>
                      <div className="text-xs text-slate-400 mt-0.5">{d.id}</div>
                    </td>
                    <td className="px-5 py-3 text-slate-600">{d.oppType}</td>
                    <td className="px-5 py-3 text-slate-600">{d.owner}</td>
                    <td className="px-5 py-3 text-right">
                      <span className={`font-semibold ${d.p_win >= 0.9 ? 'text-emerald-600' : d.p_win >= 0.7 ? 'text-amber-600' : 'text-rose-600'}`}>
                        {d.p_win !== null ? `${(d.p_win * 100).toFixed(1)}%` : '—'}
                      </span>
                    </td>
                    <td className="px-5 py-3">
                      <Link to={`/deal/${encodeURIComponent(d.id)}`} state={{ deal: d }}
                        className="text-blue-600 hover:underline text-xs font-medium">View</Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {filtered.length > 100 && (
              <div className="px-5 py-3 text-xs text-slate-500 border-t border-slate-100 bg-slate-50">
                Showing 100 of {filtered.length.toLocaleString()}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (!deal) return null;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      <div className="max-w-[1400px] mx-auto px-8 py-8">
        <Link to="/deal" className="text-blue-600 text-sm font-medium hover:underline mb-6 inline-block">
          ← All Deals
        </Link>

        <div className="grid grid-cols-3 gap-8">
          {/* Main Info */}
          <div className="col-span-2 space-y-6">
            <div>
              <h1 className="text-2xl font-semibold text-slate-800 mb-2">{deal.name}</h1>
              <div className="flex items-center gap-4 text-sm">
                <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${
                  deal.status === 'Won' ? 'bg-emerald-100 text-emerald-700' :
                  deal.status === 'Lost' ? 'bg-rose-100 text-rose-700' :
                  'bg-slate-100 text-slate-600'
                }`}>{deal.status}</span>
                <span className="text-slate-500">{deal.oppType}</span>
                <span className="text-slate-400">{deal.id}</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <div className="text-slate-400 text-xs uppercase tracking-wider mb-2">Owner</div>
                <div className="text-lg font-semibold text-slate-700">{deal.owner}</div>
              </div>
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <div className="text-slate-400 text-xs uppercase tracking-wider mb-2">Stage</div>
                <div className="text-lg font-semibold text-slate-700">{deal.stage}</div>
              </div>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <div className="text-slate-400 text-xs uppercase tracking-wider mb-4">Analysis</div>
              <p className="text-slate-600 text-sm leading-relaxed">
                {deal.p_win >= 0.9 
                  ? "This opportunity shows strong indicators for success based on historical patterns. The model has high confidence in a positive outcome."
                  : deal.p_win >= 0.7
                  ? "Moderate confidence level. Consider reviewing competitive positioning and addressing any outstanding concerns."
                  : "Lower confidence score suggests this deal may need additional attention. Review qualification criteria and engagement strategy."}
              </p>
            </div>
          </div>

          {/* Probability Panel */}
          <div className="space-y-6">
            <div className={`rounded-xl p-6 shadow-lg ${
              deal.p_win >= 0.9 ? 'bg-gradient-to-br from-emerald-500 to-emerald-600 shadow-emerald-500/20' :
              deal.p_win >= 0.7 ? 'bg-gradient-to-br from-amber-500 to-amber-600 shadow-amber-500/20' :
              'bg-gradient-to-br from-rose-500 to-rose-600 shadow-rose-500/20'
            } text-white`}>
              <div className="text-white/80 text-xs uppercase tracking-wider mb-2">Win Probability</div>
              <div className="text-5xl font-bold">
                {deal.p_win !== null ? `${(deal.p_win * 100).toFixed(1)}%` : '—'}
              </div>
              <div className="mt-4 h-2 bg-white/20 rounded-full overflow-hidden">
                <div className="h-full bg-white rounded-full" style={{ width: `${(deal.p_win || 0) * 100}%` }} />
              </div>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <div className="text-slate-400 text-xs uppercase tracking-wider mb-3">Confidence</div>
              <div className={`text-lg font-semibold ${
                deal.confidence === 'High' ? 'text-emerald-600' :
                deal.confidence === 'Medium' ? 'text-amber-600' : 'text-rose-600'
              }`}>{deal.confidence}</div>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <div className="text-slate-400 text-xs uppercase tracking-wider mb-3">Model Prediction</div>
              <div className={deal.pred_win ? 'text-emerald-600 font-medium' : 'text-slate-500'}>
                {deal.pred_win ? '✓ Predicted Win' : 'Below Threshold'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
