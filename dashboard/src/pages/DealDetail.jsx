import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";

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

// Fake API for now; replace with your backend
async function fetchDeal(id) {
  // try real API first
  try {
    const res = await fetch(`/api/deals/${id}`);
    if (res.ok) return await res.json();
  } catch (_) {}
  // fallback sample to mirror your screenshot
  return {
    id,
    title: "Enterprise Software Implementation – Acme Corp",
    accountExecutive: "Sarah Johnson",
    region: "North America",
    amount: 125000,
    stage: "Negotiation",
    winProbability: 0.74,
    factors: [
      { label: "Strong technical fit", impact: +0.15 },
      { label: "Budget approved", impact: +0.12 },
      { label: "Multiple stakeholder engagement", impact: +0.08 },
      { label: "Competitive pressure", impact: -0.06 },
      { label: "Timeline constraints", impact: -0.05 },
    ],
    recommendation: "Focus on value proposition to counter competitive pressure",
    activity: [
      { label: "Demo scheduled with technical team", ago: "2 days ago", source: "Salesforce", color: "bg-emerald-500" },
      { label: "Pricing proposal sent", ago: "4 days ago", source: "People.ai", color: "bg-violet-500" },
      { label: "Technical requirements call", ago: "1 week ago", source: "Salesforce", color: "bg-amber-500" },
      { label: "Initial discovery meeting", ago: "2 weeks ago", source: "People.ai", color: "bg-sky-500" },
    ],
    competitor: {
      name: "Salesforce",
      intel:
        'Client mentioned Salesforce pricing in last call. They seem impressed with their reporting capabilities but concerned about implementation timeline.',
    },
    notes: {
      themes: ["Integration Requirements", "Budget Approval", "Timeline Pressure", "Technical Evaluation"],
      decisionMakers: [
        { name: "John Smith", role: "CTO (Technical Decision)" },
        { name: "Maria Lopez", role: "CFO (Budget Approval)" },
      ],
    },
    updatedAt: new Date().toISOString(),
  };
}

export default function DealDetail() {
  const { id } = useParams();
  const [deal, setDeal] = useState(null);

  useEffect(() => {
    let mounted = true;
    (async () => {
      const data = await fetchDeal(id);
      if (mounted) setDeal(data);
    })();
    return () => (mounted = false);
  }, [id]);

  const kpis = useMemo(() => {
    if (!deal) return [];
    return [
      { label: "Account Executive", value: deal.accountExecutive },
      { label: "Region", value: deal.region },
      { label: "Amount", value: `$${deal.amount.toLocaleString()}` },
      { label: "Stage", value: deal.stage, badge: true },
    ];
  }, [deal]);

  if (!deal) return <div className="text-sm text-slate-500">Loading…</div>;

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
          subtitle="Recent activities from Salesforce & People.ai"
          className="lg:col-span-2"
        >
          <ul className="space-y-3">
            {deal.activity.map((a, i) => (
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

            <div className="mt-4">
              <div className="text-sm font-medium">Key Factors:</div>
              <ul className="mt-2 space-y-1 text-sm text-slate-700">
                {deal.factors.map((f) => (
                  <li key={f.label}>
                    • {f.label} ({f.impact > 0 ? "+" : ""}{Math.round(f.impact * 100)}%)
                  </li>
                ))}
              </ul>
            </div>

            <div className="mt-4 border-t pt-3 text-sm">
              <span className="font-medium">Recommendation:</span>{" "}
              {deal.recommendation}.
            </div>
          </div>
        </Card>
      </div>

      {/* Bottom row: Competitor Analysis (left) + Sales Notes (right) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Competitor Analysis" subtitle="NLP-based competitor detection">
          <div className="flex items-center gap-2 text-sm">
            <span className="font-medium">Primary Competitor</span>
            <Badge tone="red">{deal.competitor.name}</Badge>
          </div>

          <blockquote className="mt-3 rounded-xl bg-indigo-50 text-indigo-900 p-4 text-sm leading-relaxed">
            “{deal.competitor.intel}”
          </blockquote>
        </Card>

        <Card title="Sales Notes Summary" subtitle="AI-extracted key phrases">
          <div className="text-sm">
            <div className="font-medium">Key Themes:</div>
            <div className="mt-2 flex flex-wrap gap-2">
              {deal.notes.themes.map((t) => (
                <span key={t} className="px-2.5 py-1 rounded-full text-xs bg-slate-100 text-slate-700">
                  {t}
                </span>
              ))}
            </div>

            <div className="mt-4 font-medium">Decision Makers:</div>
            <ul className="mt-2 space-y-1">
              {deal.notes.decisionMakers.map((p) => (
                <li key={p.name} className="flex items-center justify-between">
                  <span>{p.name}</span>
                  <span className="text-slate-500 text-xs">{p.role}</span>
                </li>
              ))}
            </ul>
          </div>
        </Card>
      </div>
    </div>
  );
}
