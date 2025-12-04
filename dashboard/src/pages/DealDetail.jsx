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
      <div className="min-h-screen bg-[#0f0f0f] flex items-center justify-center">
        <div className="w-6 h-6 border-2 border-amber-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-[#0f0f0f] text-neutral-100 p-8">
        <div className="max-w-[1400px] mx-auto">
          <div className="text-red-400 mb-4">{error}</div>
          <Link to="/" className="text-amber-500 text-sm hover:underline">← Back</Link>
        </div>
      </div>
    );
  }

  if (isOverview) {
    return (
      <div className="min-h-screen bg-[#0f0f0f] text-neutral-100">
        <div className="max-w-[1400px] mx-auto px-8 py-8">
          <div className="flex items-end justify-between mb-8">
            <div>
              <div className="text-xs text-neutral-500 uppercase tracking-[0.2em] mb-1">Browse</div>
              <h1 className="text-2xl font-light">All Opportunities</h1>
            </div>
            <div className="text-sm text-neutral-500">{filtered.length.toLocaleString()} deals</div>
          </div>

          <div className="bg-neutral-900/30 border border-neutral-800 p-4 mb-6">
            <div className="flex gap-4">
              <input type="text" placeholder="Search…" value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="flex-1 bg-transparent border border-neutral-700 px-4 py-2 text-sm focus:outline-none focus:border-amber-500/50 placeholder:text-neutral-600" />
              <select value={filter} onChange={(e) => setFilter(e.target.value)}
                className="bg-transparent border border-neutral-700 px-4 py-2 text-sm focus:outline-none">
                <option value="All" className="bg-neutral-900">All confidence</option>
                <option value="High" className="bg-neutral-900">High</option>
                <option value="Medium" className="bg-neutral-900">Medium</option>
                <option value="Low" className="bg-neutral-900">Low</option>
              </select>
            </div>
          </div>

          <div className="bg-neutral-900/30 border border-neutral-800">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-neutral-500 text-xs uppercase tracking-wider border-b border-neutral-800">
                  <th className="px-5 py-3 font-medium">Opportunity</th>
                  <th className="px-5 py-3 font-medium">Type</th>
                  <th className="px-5 py-3 font-medium">Owner</th>
                  <th className="px-5 py-3 font-medium text-right">Probability</th>
                  <th className="px-5 py-3 font-medium w-20"></th>
                </tr>
              </thead>
              <tbody>
                {filtered.slice(0, 100).map((d, i) => (
                  <tr key={d.id} className={`border-b border-neutral-800/50 hover:bg-neutral-800/30 ${i % 2 === 0 ? 'bg-neutral-900/20' : ''}`}>
                    <td className="px-5 py-3">
                      <div className="text-neutral-100">{d.name.substring(0, 50)}{d.name.length > 50 ? "…" : ""}</div>
                      <div className="text-xs text-neutral-600 mt-0.5">{d.id}</div>
                    </td>
                    <td className="px-5 py-3 text-neutral-400">{d.oppType}</td>
                    <td className="px-5 py-3 text-neutral-400">{d.owner}</td>
                    <td className="px-5 py-3 text-right">
                      <span className={d.p_win >= 0.9 ? 'text-emerald-500' : d.p_win >= 0.7 ? 'text-amber-500' : 'text-red-400'}>
                        {d.p_win !== null ? `${(d.p_win * 100).toFixed(1)}%` : '—'}
                      </span>
                    </td>
                    <td className="px-5 py-3">
                      <Link to={`/deal/${encodeURIComponent(d.id)}`} state={{ deal: d }}
                        className="text-amber-500 hover:underline text-xs">View</Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {filtered.length > 100 && (
              <div className="px-5 py-3 text-xs text-neutral-500 border-t border-neutral-800">
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
    <div className="min-h-screen bg-[#0f0f0f] text-neutral-100">
      <div className="max-w-[1400px] mx-auto px-8 py-8">
        <Link to="/deal" className="text-amber-500 text-xs uppercase tracking-wider hover:underline mb-6 inline-block">
          ← All Deals
        </Link>

        <div className="grid grid-cols-3 gap-8">
          {/* Main Info */}
          <div className="col-span-2 space-y-6">
            <div>
              <h1 className="text-2xl font-light mb-2">{deal.name}</h1>
              <div className="flex items-center gap-4 text-sm">
                <span className={`px-2 py-0.5 text-xs ${
                  deal.status === 'Won' ? 'bg-emerald-500/20 text-emerald-400' :
                  deal.status === 'Lost' ? 'bg-red-500/20 text-red-400' :
                  'bg-neutral-700/50 text-neutral-400'
                }`}>{deal.status}</span>
                <span className="text-neutral-500">{deal.oppType}</span>
                <span className="text-neutral-600">{deal.id}</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-neutral-900/50 border border-neutral-800 p-5">
                <div className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Owner</div>
                <div className="text-lg">{deal.owner}</div>
              </div>
              <div className="bg-neutral-900/50 border border-neutral-800 p-5">
                <div className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Stage</div>
                <div className="text-lg">{deal.stage}</div>
              </div>
            </div>

            <div className="bg-neutral-900/50 border border-neutral-800 p-5">
              <div className="text-neutral-500 text-xs uppercase tracking-wider mb-4">Analysis</div>
              <p className="text-neutral-400 text-sm leading-relaxed">
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
            <div className={`p-6 border ${
              deal.p_win >= 0.9 ? 'bg-emerald-950/30 border-emerald-900/50' :
              deal.p_win >= 0.7 ? 'bg-amber-950/30 border-amber-900/50' :
              'bg-red-950/30 border-red-900/50'
            }`}>
              <div className="text-xs uppercase tracking-wider mb-2 opacity-70">Win Probability</div>
              <div className={`text-5xl font-light ${
                deal.p_win >= 0.9 ? 'text-emerald-400' :
                deal.p_win >= 0.7 ? 'text-amber-400' : 'text-red-400'
              }`}>
                {deal.p_win !== null ? `${(deal.p_win * 100).toFixed(1)}%` : '—'}
              </div>
              <div className="mt-4 h-1.5 bg-neutral-800">
                <div className={`h-full ${
                  deal.p_win >= 0.9 ? 'bg-emerald-500' :
                  deal.p_win >= 0.7 ? 'bg-amber-500' : 'bg-red-500'
                }`} style={{ width: `${(deal.p_win || 0) * 100}%` }} />
              </div>
            </div>

            <div className="bg-neutral-900/50 border border-neutral-800 p-5">
              <div className="text-neutral-500 text-xs uppercase tracking-wider mb-3">Confidence</div>
              <div className={`text-lg ${
                deal.confidence === 'High' ? 'text-emerald-400' :
                deal.confidence === 'Medium' ? 'text-amber-400' : 'text-red-400'
              }`}>{deal.confidence}</div>
            </div>

            <div className="bg-neutral-900/50 border border-neutral-800 p-5">
              <div className="text-neutral-500 text-xs uppercase tracking-wider mb-3">Model Prediction</div>
              <div className={deal.pred_win ? 'text-emerald-400' : 'text-neutral-400'}>
                {deal.pred_win ? '✓ Predicted Win' : 'Below Threshold'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
