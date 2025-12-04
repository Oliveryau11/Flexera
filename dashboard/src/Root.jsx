import { Routes, Route, NavLink } from "react-router-dom";
import App from "./App.jsx";
import DealDetail from "./pages/DealDetail.jsx";
import CompetitorTracker from "./pages/CompetitorTracker.jsx";
import ModelInsights from "./pages/ModelInsights.jsx";

export default function Root() {
  return (
    <div className="min-h-screen bg-[#0f0f0f]">
      {/* Minimal top bar */}
      <nav className="border-b border-neutral-800 bg-[#0f0f0f]">
        <div className="max-w-[1400px] mx-auto px-8 flex items-center justify-between h-12">
          <NavLink to="/" className="text-neutral-400 hover:text-neutral-100 text-xs uppercase tracking-[0.15em]">
            Flexera
          </NavLink>
          <div className="flex items-center gap-6">
            <NavLink 
              to="/" 
              className={({ isActive }) => 
                `text-xs uppercase tracking-wider transition-colors ${isActive ? 'text-amber-500' : 'text-neutral-500 hover:text-neutral-300'}`
              }
            >
              Dashboard
            </NavLink>
            <NavLink 
              to="/model-insights" 
              className={({ isActive }) => 
                `text-xs uppercase tracking-wider transition-colors ${isActive ? 'text-amber-500' : 'text-neutral-500 hover:text-neutral-300'}`
              }
            >
              Model
            </NavLink>
            <NavLink 
              to="/deal" 
              className={({ isActive }) => 
                `text-xs uppercase tracking-wider transition-colors ${isActive ? 'text-amber-500' : 'text-neutral-500 hover:text-neutral-300'}`
              }
            >
              Deals
            </NavLink>
            <NavLink 
              to="/competitors" 
              className={({ isActive }) => 
                `text-xs uppercase tracking-wider transition-colors ${isActive ? 'text-amber-500' : 'text-neutral-500 hover:text-neutral-300'}`
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
