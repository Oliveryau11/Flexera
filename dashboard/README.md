# Win/Loss AI Dashboard

A React-based dashboard for visualizing sales opportunity predictions and model performance from the Flexera MLModel.

## Features

- **Dashboard Home**: Overview of win rates, model predictions, and deal analytics
- **Model Insights**: Detailed model performance metrics, calibration charts, and feature importance
- **Deal Detail View**: Individual opportunity analysis with AI-predicted win probability
- **Competitor Tracker**: Competitive intelligence and head-to-head analysis

## Data Integration

The dashboard integrates with the MLModel outputs:

### Model Outputs (synced to `public/api/`)
- `model_metrics.csv` - Performance metrics for all trained models
- `win_probabilities.csv` - Predictions for open opportunities  
- `model_registry.json` - Model metadata and configuration
- `RealDummyData.xlsx` - Source opportunity data

### Feedback Data (`public/api/feedback/`)
- `calibration_table.csv` - Model calibration bins
- `feature_importance_grouped.csv` - Feature importance scores
- `segment_perf_regions.csv` - Regional performance breakdown
- `loss_reasons_overall.csv` - Top loss reasons analysis
- And more...

## Getting Started

### Prerequisites
- Node.js 18+
- npm or yarn

### Installation

```bash
cd dashboard
npm install
```

### Development

```bash
npm run dev
```

The dashboard will be available at http://localhost:5173

### Syncing Model Data

After training the model, sync the outputs to the dashboard:

```bash
# From the project root
python sync_model_to_dashboard.py
```

This copies all model outputs and feedback files to `dashboard/public/api/`.

### Full Workflow

1. **Train the model** (from `MLModel/` directory):
   ```bash
   cd MLModel
   python PreliminaryModel.py
   ```

2. **Sync to dashboard** (from project root):
   ```bash
   python sync_model_to_dashboard.py
   ```

3. **Start the dashboard**:
   ```bash
   cd dashboard
   npm run dev
   ```

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **Recharts** - Charts and visualizations
- **Tailwind CSS** - Styling
- **React Router** - Client-side routing
- **xlsx** - Excel file parsing

## Pages

### Dashboard (`/`)
- Model performance strip with AUC, Precision, Recall
- KPI cards: Win Rate, Avg p(Win), High-Confidence Deals, Weighted Pipeline
- Charts: Win Rate by Region, Competitors, Loss by Country, p(Win) Distribution
- Interactive deals table with filters

### Model Insights (`/model-insights`)
- Model comparison table (LogReg, RandomForest, XGBoost)
- Calibration chart (predicted vs actual)
- Feature importance chart
- Regional performance breakdown
- Top loss reasons analysis

### Deal Detail (`/deal/:id`)
- Individual opportunity details
- Win probability with key factors
- Activity timeline (placeholder)
- Competitor analysis

### Competitor Tracker (`/competitors`)
- Competitor performance metrics
- Win rate vs competitors
- Trend analysis
