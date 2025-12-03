import { Routes, Route, NavLink } from "react-router-dom";
import App from "./App.jsx";
import DealDetail from "./pages/DealDetail.jsx";
import CompetitorTracker from "./pages/CompetitorTracker.jsx";

export default function Root() {
  const tab = ({ isActive }) =>
    "px-3 py-2 rounded-xl text-sm font-medium " +
    (isActive ? "bg-[#0056D2] border-[#0056D2] text-white"     : "bg-white text-slate-700 border-slate-200 hover:border-[#0056D2] hover:text-[#0056D2] hover:bg-slate-50");

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="px-6 pt-6">
        <h1 className="text-3xl font-bold">Win/Loss AI Dashboard</h1>
        <p className="text-slate-600 mt-1">AI-powered sales analytics and competitive intelligence</p>

        <nav className="mt-4 flex gap-2">
          <NavLink className={tab} to="/">Dashboard Home</NavLink>
          <NavLink className={tab} to="/deal">Deal Detail View</NavLink>
          <NavLink className={tab} to="/competitors">Competitor Tracker</NavLink>
        </nav>
      </header>

      <main className="p-6">
        <Routes>
          <Route path="/" element={<App />} />
          <Route path="/deal" element={<DealDetail />} /> 
          <Route path="/deal/:id" element={<DealDetail />} />
          <Route path="/competitors" element={<CompetitorTracker />} />
          {/* add other routes later */}
        </Routes>
      </main>
    </div>
  );
}
