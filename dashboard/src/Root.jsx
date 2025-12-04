import { Routes, Route, NavLink } from "react-router-dom";
import App from "./App.jsx";
import DealDetail from "./pages/DealDetail.jsx";
import CompetitorTracker from "./pages/CompetitorTracker.jsx";
import ModelInsights from "./pages/ModelInsights.jsx";

export default function Root() {
  return (
    <div className="min-h-screen bg-slate-50">
      {/* Navigation */}
      <nav className="bg-white border-b border-slate-200 shadow-sm sticky top-0 z-50">
        <div className="max-w-[1400px] mx-auto px-8 flex items-center justify-between h-14">
          <NavLink to="/" className="text-slate-800 hover:text-blue-600 font-semibold text-sm uppercase tracking-wider">
            Flexera
          </NavLink>
          <div className="flex items-center gap-8">
            <NavLink 
              to="/" 
              className={({ isActive }) => 
                `text-sm font-medium transition-colors ${isActive ? 'text-blue-600' : 'text-slate-400 hover:text-slate-600'}`
              }
            >
              Dashboard
            </NavLink>
            <NavLink 
              to="/model-insights" 
              className={({ isActive }) => 
                `text-sm font-medium transition-colors ${isActive ? 'text-blue-600' : 'text-slate-400 hover:text-slate-600'}`
              }
            >
              Model
            </NavLink>
            <NavLink 
              to="/deal" 
              className={({ isActive }) => 
                `text-sm font-medium transition-colors ${isActive ? 'text-blue-600' : 'text-slate-400 hover:text-slate-600'}`
              }
            >
              Deals
            </NavLink>
            <NavLink 
              to="/competitors" 
              className={({ isActive }) => 
                `text-sm font-medium transition-colors ${isActive ? 'text-blue-600' : 'text-slate-400 hover:text-slate-600'}`
              }
            >
              Analytics
            </NavLink>
          </div>
        </div>
      </nav>

      <main>
        <Routes>
          <Route path="/" element={<App />} />
          <Route path="/model-insights" element={<ModelInsights />} />
          <Route path="/deal" element={<DealDetail />} /> 
          <Route path="/deal/:id" element={<DealDetail />} />
          <Route path="/competitors" element={<CompetitorTracker />} />
        </Routes>
      </main>
    </div>
  );
}
