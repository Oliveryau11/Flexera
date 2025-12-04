import { useEffect, useState, useMemo } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from "recharts";
import { Link } from "react-router-dom";

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

async function fetchJSON(url) {
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) return null;
    return res.json();
  } catch { return null; }
}

export default function ModelInsights() {
  const [registry, setRegistry] = useState(null);
  const [featureImportance, setFeatureImportance] = useState([]);
  const [calibration, setCalibration] = useState([]);
  const [segmentPerf, setSegmentPerf] = useState([]);
  const [lossReasons, setLossReasons] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const [reg, fi, cal, seg, loss] = await Promise.all([
        fetchJSON("/api/model_registry.json"),
        fetchCSVRows("/api/feedback/feature_importance.csv"),
        fetchCSVRows("/api/feedback/calibration_table.csv"),
        fetchCSVRows("/api/feedback/segment_perf_regions.csv"),
        fetchCSVRows("/api/feedback/loss_reasons_overall.csv"),
      ]);
      if (!cancelled) {
        setRegistry(reg);
        setFeatureImportance(fi.slice(0, 15).map(r => ({
          feature: (r.feature || "").replace(/_/g, " ").substring(0, 25),
          importance: parseFloat(r.importance) || 0,
        })));
        setCalibration(cal.map(r => ({
          bin: r.bin || r.prob_bin || "",
          predicted: parseFloat(r.mean_predicted) || 0,
          actual: parseFloat(r.mean_actual) || 0,
        })));
        setSegmentPerf(seg.map(r => ({
          region: r.segment_value || "",
          auc: parseFloat(r.auc) || 0,
          samples: parseInt(r.n_samples) || 0,
        })).filter(r => r.region && r.region !== "Unknown"));
        setLossReasons(loss.filter(r => {
          const val = (r.reason_value || "").toLowerCase();
          return !val.includes("unknown") && !val.includes("duplicate") && !val.includes("merged");
        }).slice(0, 10).map(r => ({
          reason: (r.reason_value || "").substring(0, 30),
          count: parseInt(r.n_lost) || 0,
        })));
        setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const bestModel = registry?.best_model || "XGBoost";
  const metrics = useMemo(() => {
    if (!registry?.metrics) return null;
    return registry.metrics.find(m => m.model === bestModel) || registry.metrics[0];
  }, [registry, bestModel]);

  const allMetrics = registry?.metrics || [];

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
            <div className="text-xs text-neutral-500 uppercase tracking-[0.2em] mb-1">Machine Learning</div>
            <h1 className="text-2xl font-light">Model Performance</h1>
          </div>
          <Link to="/" className="text-amber-500 text-xs uppercase tracking-wider hover:underline">
            ← Dashboard
          </Link>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-5 gap-4 mb-8">
          <div className="bg-amber-950/30 border border-amber-900/50 p-5">
            <div className="text-amber-400/70 text-xs uppercase tracking-wider mb-2">Active Model</div>
            <div className="text-2xl font-light text-amber-400">{bestModel}</div>
          </div>
          <div className="bg-neutral-900/50 border border-neutral-800 p-5">
            <div className="text-neutral-500 text-xs uppercase tracking-wider mb-2">AUC-ROC</div>
            <div className="text-2xl font-light">{metrics ? (parseFloat(metrics.val_auc) * 100).toFixed(1) : "—"}%</div>
          </div>
          <div className="bg-neutral-900/50 border border-neutral-800 p-5">
            <div className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Precision</div>
            <div className="text-2xl font-light">{metrics ? (parseFloat(metrics.val_precision) * 100).toFixed(1) : "—"}%</div>
          </div>
          <div className="bg-neutral-900/50 border border-neutral-800 p-5">
            <div className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Recall</div>
            <div className="text-2xl font-light">{metrics ? (parseFloat(metrics.val_recall) * 100).toFixed(1) : "—"}%</div>
          </div>
          <div className="bg-neutral-900/50 border border-neutral-800 p-5">
            <div className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Samples</div>
            <div className="text-2xl font-light">{registry?.train_samples?.toLocaleString() || "—"}</div>
          </div>
        </div>

        {/* Model Comparison */}
        <div className="bg-neutral-900/30 border border-neutral-800 mb-8">
          <div className="px-5 py-4 border-b border-neutral-800">
            <div className="text-xs text-neutral-500 uppercase tracking-wider">Model Comparison</div>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-neutral-500 text-xs uppercase tracking-wider border-b border-neutral-800">
                <th className="px-5 py-3 font-medium">Model</th>
                <th className="px-5 py-3 font-medium text-right">AUC</th>
                <th className="px-5 py-3 font-medium text-right">Precision</th>
                <th className="px-5 py-3 font-medium text-right">Recall</th>
                <th className="px-5 py-3 font-medium text-right">F1</th>
                <th className="px-5 py-3 font-medium w-16"></th>
              </tr>
            </thead>
            <tbody>
              {allMetrics.map((m, i) => (
                <tr key={m.model} className={`border-b border-neutral-800/50 ${m.model === bestModel ? 'bg-amber-950/20' : ''}`}>
                  <td className="px-5 py-3 font-medium">
                    {m.model}
                    {m.model === bestModel && <span className="ml-2 text-xs text-amber-500">●</span>}
                  </td>
                  <td className="px-5 py-3 text-right">{(parseFloat(m.val_auc) * 100).toFixed(1)}%</td>
                  <td className="px-5 py-3 text-right">{(parseFloat(m.val_precision) * 100).toFixed(1)}%</td>
                  <td className="px-5 py-3 text-right">{(parseFloat(m.val_recall) * 100).toFixed(1)}%</td>
                  <td className="px-5 py-3 text-right">{(parseFloat(m.val_f1) * 100).toFixed(1)}%</td>
                  <td className="px-5 py-3 text-center">
                    {m.model === bestModel && <span className="text-xs text-amber-500">Best</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-2 gap-6 mb-8">
          <div className="bg-neutral-900/30 border border-neutral-800 p-5">
            <div className="text-xs text-neutral-500 uppercase tracking-wider mb-4">Feature Importance</div>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={featureImportance} layout="vertical">
                  <CartesianGrid stroke="#262626" strokeDasharray="3 3" />
                  <XAxis type="number" tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#404040' }} />
                  <YAxis type="category" dataKey="feature" tick={{ fill: '#a3a3a3', fontSize: 9 }} width={120} axisLine={{ stroke: '#404040' }} />
                  <Tooltip contentStyle={{ background: '#171717', border: '1px solid #404040', borderRadius: 0 }} />
                  <Bar dataKey="importance" fill="#f59e0b" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-neutral-900/30 border border-neutral-800 p-5">
            <div className="text-xs text-neutral-500 uppercase tracking-wider mb-4">Calibration Curve</div>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={calibration}>
                  <CartesianGrid stroke="#262626" strokeDasharray="3 3" />
                  <XAxis dataKey="bin" tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#404040' }} />
                  <YAxis tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#404040' }} domain={[0, 1]} />
                  <Tooltip contentStyle={{ background: '#171717', border: '1px solid #404040', borderRadius: 0 }} />
                  <Line type="monotone" dataKey="predicted" stroke="#f59e0b" name="Predicted" strokeWidth={2} dot={{ fill: '#f59e0b' }} />
                  <Line type="monotone" dataKey="actual" stroke="#10b981" name="Actual" strokeWidth={2} dot={{ fill: '#10b981' }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="flex justify-center gap-6 mt-3 text-xs">
              <div className="flex items-center gap-2"><div className="w-3 h-0.5 bg-amber-500" /> Predicted</div>
              <div className="flex items-center gap-2"><div className="w-3 h-0.5 bg-emerald-500" /> Actual</div>
            </div>
          </div>
        </div>

        {/* Regional Performance & Loss Reasons */}
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-neutral-900/30 border border-neutral-800 p-5">
            <div className="text-xs text-neutral-500 uppercase tracking-wider mb-4">Performance by Region</div>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={segmentPerf}>
                  <CartesianGrid stroke="#262626" strokeDasharray="3 3" />
                  <XAxis dataKey="region" tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#404040' }} />
                  <YAxis tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#404040' }} domain={[0.5, 1]} />
                  <Tooltip contentStyle={{ background: '#171717', border: '1px solid #404040', borderRadius: 0 }}
                    formatter={(v) => [(v * 100).toFixed(1) + "%", "AUC"]} />
                  <Bar dataKey="auc" fill="#f59e0b" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-neutral-900/30 border border-neutral-800 p-5">
            <div className="text-xs text-neutral-500 uppercase tracking-wider mb-4">Top Loss Reasons</div>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={lossReasons} layout="vertical">
                  <CartesianGrid stroke="#262626" strokeDasharray="3 3" />
                  <XAxis type="number" tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#404040' }} />
                  <YAxis type="category" dataKey="reason" tick={{ fill: '#a3a3a3', fontSize: 9 }} width={110} axisLine={{ stroke: '#404040' }} />
                  <Tooltip contentStyle={{ background: '#171717', border: '1px solid #404040', borderRadius: 0 }} />
                  <Bar dataKey="count" fill="#ef4444" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Model Info */}
        <div className="mt-8 bg-neutral-900/30 border border-neutral-800 p-5">
          <div className="text-xs text-neutral-500 uppercase tracking-wider mb-4">Training Details</div>
          <div className="grid grid-cols-4 gap-6 text-sm">
            <div>
              <div className="text-neutral-500 mb-1">Trained At</div>
              <div>{registry?.trained_at ? new Date(registry.trained_at).toLocaleString() : "—"}</div>
            </div>
            <div>
              <div className="text-neutral-500 mb-1">Training Samples</div>
              <div>{registry?.train_samples?.toLocaleString() || "—"}</div>
            </div>
            <div>
              <div className="text-neutral-500 mb-1">Validation Samples</div>
              <div>{registry?.val_samples?.toLocaleString() || "—"}</div>
            </div>
            <div>
              <div className="text-neutral-500 mb-1">Precision Threshold</div>
              <div>{registry?.precision_threshold || "0.75"}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
