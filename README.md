# Flexera Win/Loss Prediction System

An end-to-end machine learning system for predicting sales opportunity outcomes, integrated with a modern React dashboard for visualization.

## Project Structure

```
Flexera/
├── MLModel/                    # Machine Learning Pipeline
│   ├── PreliminaryModel.py     # Main training script
│   ├── data.xlsx               # Training data
│   ├── model_metrics.csv       # Model performance metrics
│   ├── win_probabilities.csv   # Predictions for open opportunities
│   ├── model_registry.json     # Model metadata
│   └── feedback/               # Model diagnostics
│       ├── calibration_table.csv
│       ├── feature_importance_grouped.csv
│       ├── segment_perf_regions.csv
│       ├── loss_reasons_overall.csv
│       └── ...
│
├── dashboard/                  # React Dashboard
│   ├── src/
│   │   ├── App.jsx            # Main dashboard view
│   │   ├── Root.jsx           # App shell with navigation
│   │   └── pages/
│   │       ├── ModelInsights.jsx    # Model performance page
│   │       ├── DealDetail.jsx       # Individual deal view
│   │       └── CompetitorTracker.jsx # Competitor analysis
│   └── public/api/            # Synced model outputs
│
└── sync_model_to_dashboard.py  # Data sync script
```

## Quick Start

### 1. Train the Model

```bash
cd MLModel
python PreliminaryModel.py
```

**Outputs:**
- Best model: XGBoost (AUC ~99.1%)
- Precision at target: ~95%
- Predictions for all open opportunities

### 2. Sync to Dashboard

```bash
python sync_model_to_dashboard.py
```

This copies model outputs to `dashboard/public/api/` for the frontend.

### 3. Run the Dashboard

```bash
cd dashboard
npm install  # First time only
npm run dev
```

Open http://localhost:5173

## Model Details

### Features
The model uses 60+ features including:
- **Geographic**: Account Region, Country
- **Temporal**: Created Date, Close Date, Meeting Dates
- **Product**: Product Line, Solution Area
- **Engagement**: Meeting counts, Last Touch dates

### Performance (XGBoost)
| Metric | Value |
|--------|-------|
| ROC-AUC | 99.14% |
| PR-AUC | 99.80% |
| Precision @ 95% target | 95.01% |
| Recall @ 95% target | 98.65% |
| Brier Score | 0.0259 |

### Outputs
- **win_probabilities.csv**: Win probability for each open opportunity
- **model_metrics.csv**: Comparison of all trained models
- **feedback/**: Diagnostic artifacts for model understanding

## Dashboard Features

### Dashboard Home
- Real-time model performance metrics
- Win rate analytics by region and competitor
- Deal pipeline with weighted values
- Interactive filtering

### Model Insights
- Model comparison (LogReg, RandomForest, XGBoost)
- Calibration chart
- Feature importance
- Regional performance breakdown
- Loss reason analysis

### Deal Detail
- Individual opportunity analysis
- Win probability explanation
- Key factors driving prediction

### Competitor Tracker
- Head-to-head win rates
- Competitor frequency analysis
- Trend indicators

## Development

### Prerequisites
- Python 3.8+ with pandas, sklearn, xgboost
- Node.js 18+ with npm

### ML Model Development
```bash
cd MLModel
pip install pandas scikit-learn xgboost openpyxl
python PreliminaryModel.py
```

### Dashboard Development
```bash
cd dashboard
npm install
npm run dev
```

## License

Proprietary - Flexera
