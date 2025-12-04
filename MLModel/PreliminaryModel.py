"""
Precision-First Win/Loss Modeling (Flexera-style) — with Robust Target Parsing
-----------------------------------------------------------------------------
- Robust target mapping for CRO Win (handles 'Yes/No', 'Y/N', 'True/False', 'Closed Won/Lost', case/whitespace)
- Leakage-safe feature selection (no stage/forecast/win/loss; drop ID-like cols)
- Calibrated probabilities (isotonic)
- Threshold tuned to hit target Precision (fallback to best-F1)
- Feedback artifacts + diagnostics

Outputs:
  model_metrics.csv
  win_probabilities.csv
  feedback/error_analysis_val.csv
  feedback/feature_importance.csv
  feedback/feature_importance_grouped.csv
  feedback/calibration_table.csv
  feedback/loss_drivers.csv
  feedback/used_features.csv
  feedback/segment_perf_regions.csv
  feedback/segment_perf_dealtypes.csv
  feedback/segment_perf_region_x_saletype.csv
  feedback/loss_reasons_by_region.csv
  feedback/loss_reasons_by_country.csv
  feedback/loss_reasons_by_dealsize.csv
  feedback/loss_reasons_overall.csv
"""

# ==============================================================
#  Flexera MLModel - PreliminaryModel.py
# ==============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd
import re
from typing import Optional, List, Dict
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    precision_recall_curve
)
from sklearn.inspection import permutation_importance
from scipy import sparse

# NEW: XGBoost
from xgboost import XGBClassifier

# ----------------------- Config -----------------------
EXCEL_PATH = Path("data.xlsx")
SHEET = 0
TARGET_COL = "CRO Win"

# Business fields (if present) for outputs
ID_CANDIDATES    = ["Id", "Opportunity Id", "Opportunity ID"]
NAME_CANDIDATES  = ["Name", "Opportunity Name"]
OWNER_CANDIDATES = ["Opportunity Owner Name", "Owner", "Opportunity Owner"]
AMOUNT_CANDIDATES= ["Amount", "Opportunity Line ACV USD", "ACV", "ARR"]
STAGE_COL        = "Stage"
CLOSE_DATE_COL   = "Close Date"
CREATED_DATE_COL = "Created Date"   # for optional temporal split

TARGET_PRECISION = 0.95
TEST_SIZE        = 0.30
RANDOM_STATE     = 42

# ---- New toggles: K-Fold & Temporal Split ----
USE_KFOLD_MODEL_SELECTION = False
N_FOLDS                    = 5

USE_TEMPORAL_SPLIT   = True
TEMP_SPLIT_COL       = CREATED_DATE_COL
TEMP_SPLIT_FRACTION  = 0.8

OUT_METRICS = Path("model_metrics.csv")
OUT_PROBS   = Path("win_probabilities.csv")
FEEDBACK_DIR = Path("feedback")
FEEDBACK_DIR.mkdir(exist_ok=True)

MODEL_REGISTRY_PATH = Path("model_registry.json")

UNKNOWN_LABELS = {"Unknown", "unknown", "na", "n/a", "", "none", "nan", None}
LEAK_PATTERNS  = ["forecast", "cro win", "win", "loss"]  # training-time exclusion
# ------------------------------------------------------

# ----------------------- Utilities -----------------------
def first_present(df, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _combine_date_from_parts(df: pd.DataFrame,
                             year_col: str,
                             month_col: str,
                             day_col: str,
                             out_col: str) -> None:
    """
    Combine Y/M/D integer columns from Power BI export into a single datetime column.
    Fills df[out_col] in-place if all three component columns exist.
    """
    if all(col in df.columns for col in [year_col, month_col, day_col]):
        try:
            df[out_col] = pd.to_datetime(
                dict(
                    year=df[year_col].astype("Int64"),
                    month=df[month_col].astype("Int64"),
                    day=df[day_col].astype("Int64"),
                ),
                errors="coerce",
            )
            print(f"[Prep] Built datetime column '{out_col}' from "
                  f"'{year_col}/{month_col}/{day_col}'")
        except Exception as e:
            print(f"[Prep] Failed to build '{out_col}' from Y/M/D parts: {e}")

def load_data() -> pd.DataFrame:
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET)
    # Remove fully duplicated columns (some Excel exports repeat headers)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    # Drop fully blank rows (all NaN)
    df = df.dropna(how="all")

    # --- NEW: build Created Date & Close Date from PowerBI Y/M/Day columns ---
    _combine_date_from_parts(
        df,
        "Opportunity Created Date - Year",
        "Opportunity Created Date - Month",
        "Opportunity Created Date - Day",
        CREATED_DATE_COL,
    )
    _combine_date_from_parts(
        df,
        "Opportunity Close Date - Year",
        "Opportunity Close Date - Month",
        "Opportunity Close Date - Day",
        CLOSE_DATE_COL,
    )

    return df

def normalize_str_col(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

def map_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map CRO Win to y in {0,1}, robust to variants like
    'Closed Won', 'Closed Lost', 'Yes/No', etc.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in data.")

    col_norm = normalize_str_col(df[TARGET_COL]).str.lower()

    pos_exact = {"yes", "y", "true", "won", "win", "1", "closed won"}
    neg_exact = {"no", "n", "false", "lost", "lose", "0", "closed lost"}

    y = pd.Series(np.nan, index=df.index, dtype="float")

    # exact matches
    y[col_norm.isin(pos_exact)] = 1
    y[col_norm.isin(neg_exact)] = 0

    # fuzzy contains for 'won' / 'lost'
    won_mask = col_norm.str.contains("won", na=False) | col_norm.str.contains("赢", na=False)
    lost_mask = col_norm.str.contains("lost", na=False) | col_norm.str.contains("输", na=False)

    y[won_mask & y.isna()] = 1
    y[lost_mask & y.isna()] = 0

    df = df.copy()
    df["y"] = y
    return df

# -------- Feature picking (leak-safe + ID-like value guard) --------
ID_VALUE_RE = re.compile(r"^[A-Za-z0-9\-_/]{6,}$")  # token-ish values

def looks_like_id_value(series: pd.Series, sample_size: int = 200, ratio_thresh: float = 0.6) -> bool:
    s = series.dropna().astype(str)
    if s.empty:
        return False
    sample = s.sample(min(len(s), sample_size), random_state=42)
    hits = sample.str.fullmatch(ID_VALUE_RE).fillna(False).mean()
    return hits >= ratio_thresh

def pick_cols(df: pd.DataFrame):
    """
    Safer, leak-resistant column picking (name + value based):
    - drop all-missing
    - drop columns whose NAME contains: id, stage, forecast, cro win, win, loss
    - for categoricals: drop ID-like VALUE columns and high-cardinality (likely-ID) columns
    """
    # 0) drop all-missing
    keep = [c for c in df.columns if df[c].notna().any()]
    df = df[keep].copy()

    # 1) normalize booleans
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(int)

    def bad_name(col: str) -> bool:
        cl = col.lower()
        if any(k in cl for k in ["id", "stage"]):
            return True
        return any(p in cl for p in LEAK_PATTERNS)

    prelim_num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "y"]
    prelim_cat = [c for c in df.columns if df[c].dtype == "object"]

    prelim_num = [c for c in prelim_num if not bad_name(c)]
    prelim_cat = [c for c in prelim_cat if not bad_name(c)]

    # value-level guards on categoricals
    def is_high_card(col):
        s = df[col].dropna()
        if len(s) == 0:
            return False
        u = s.nunique()
        ratio = u / len(s)
        return (u >= 50) and (ratio > 0.2)

    cat_cols = []
    for c in prelim_cat:
        if looks_like_id_value(df[c]):  # token-like column
            continue
        if is_high_card(c):             # likely ID-valued column
            continue
        cat_cols.append(c)

    num_cols = prelim_num

    # dedup & disjoint
    cat_cols = list(dict.fromkeys(cat_cols))
    num_cols = [c for c in dict.fromkeys(num_cols) if c not in cat_cols]
    return cat_cols, num_cols

def build_preprocessor(cat_cols, num_cols):
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("to_str", FunctionTransformer(lambda X: np.asarray(X, dtype=str))),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    return pre

def get_models(pre):
    logreg = Pipeline([
        ("prep", pre),
        ("clf", LogisticRegression(
            solver="liblinear",
            penalty="l2",
            C=0.2,
            max_iter=4000,
            class_weight="balanced",
        ))
    ])
    rf = Pipeline([
        ("prep", pre),
        ("clf", RandomForestClassifier(
            n_estimators=600, random_state=RANDOM_STATE, n_jobs=-1,
            max_depth=None, min_samples_leaf=2, class_weight="balanced_subsample"
        ))
    ])
    xgb = Pipeline([
        ("prep", pre),
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            tree_method="hist",
        ))
    ])
    return {
        "LogisticRegression": logreg,
        "RandomForest": rf,
        "XGBoost": xgb,
    }

def evaluate_at_threshold(y_true, prob, thr):
    pred = (prob >= thr).astype(int)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "balacc": balanced_accuracy_score(y_true, pred),
        "cm": confusion_matrix(y_true, pred).tolist()
    }

def pick_precision_first_threshold(y_true, prob, target_prec=0.8):
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    thresholds = np.append(thresholds, 1.0)
    idx = np.where(precision >= target_prec)[0]
    if len(idx) > 0:
        best_i = idx[np.argmax(recall[idx])]
        thr = float(thresholds[best_i]) if best_i < len(thresholds) else 1.0
        return thr, True
    # fallback: best-F1
    f1s = 2 * (precision * recall) / np.maximum(precision + recall, 1e-9)
    best_i = np.nanargmax(f1s)
    thr = float(thresholds[best_i]) if best_i < len(thresholds) else 0.5
    return thr, False

def get_underlying_estimator(calibrated):
    if hasattr(calibrated, "estimator"):       # sklearn >= 1.6
        return calibrated.estimator
    if hasattr(calibrated, "base_estimator"):  # sklearn <= 1.5
        return calibrated.base_estimator
    raise AttributeError("Cannot find underlying estimator on CalibratedClassifierCV")

# ------------------- Feedback helpers -------------------
def calibration_table(y_true, prob, bins=10):
    df = pd.DataFrame({"y": y_true, "p": prob})
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    agg = df.groupby("bin", observed=False).agg(
        n=("y", "size"),
        mean_pred=("p", "mean"),
        win_rate=("y", "mean")
    ).reset_index()
    return agg

def extract_ohe_feature_names(fitted_pre, cat_cols):
    names = []
    # numeric transformer
    if len(fitted_pre.transformers_) > 0 and fitted_pre.transformers_[0][0] == "num":
        num_sel = fitted_pre.transformers_[0][2]
        if isinstance(num_sel, (list, tuple)):
            names.extend(list(num_sel))
    # categorical transformer
    if len(fitted_pre.transformers_) > 1 and fitted_pre.transformers_[1][0] == "cat":
        ohe = fitted_pre.transformers_[1][1].named_steps.get("ohe", None)
        if ohe is not None:
            names.extend(ohe.get_feature_names_out(input_features=cat_cols).tolist())
    return names

def per_example_reason_codes_logreg(calibrated, X_val, cat_cols, topk=3):
    base = get_underlying_estimator(calibrated)
    pre  = base.named_steps["prep"]
    lr   = base.named_steps["clf"]

    Xtr = pre.transform(X_val)
    coef = lr.coef_.ravel()
    if sparse.issparse(Xtr):
        contrib = Xtr.multiply(coef).toarray()
    else:
        contrib = Xtr * coef

    feat_names = extract_ohe_feature_names(pre, cat_cols)
    # group by original feature (before OHE): split at last underscore
    groups = []
    num_names = list(pre.transformers_[0][2]) if len(pre.transformers_) > 0 else []
    for f in feat_names:
        if f in num_names:  # numeric original name
            groups.append(f)
        else:
            groups.append(f.rsplit("_", 1)[0])  # 'Account Region_US' -> 'Account Region'

    groups = np.array(groups)

    reasons = []
    for i in range(contrib.shape[0]):
        s = pd.Series(contrib[i, :], index=feat_names)
        gsum = s.groupby(groups).sum().sort_values(ascending=False)
        top_pos = gsum.head(topk)
        top_neg = gsum.tail(topk)
        reason = "; ".join([f"+{k}" for k in top_pos.index]) + " | " + "; ".join([f"-{k}" for k in top_neg.index])
        reasons.append(reason)
    return reasons

def build_loss_drivers(train_df, fields=None, min_count=30):
    if fields is None:
        fields = [
            "Opportunity Reason Lost", "Opportunity Reason Lost Detail",
            "Lead Source", "Lead Source Detail", "Opportunity Type",
            "Account Region", "Account Country", "Account Shipping Country",
        ]

    df = train_df.copy()
    df["is_lost"] = (df["y"] == 0).astype(int)

    rows = []
    for col in fields:
        if col not in df.columns:
            continue

        grp = (
            df.groupby(col, dropna=False)["is_lost"]
              .agg(count="size", loss_rate="mean")
              .reset_index()
        )

        grp.rename(columns={col: "value"}, inplace=True)
        grp.insert(0, "field", col)
        grp = grp.loc[grp["count"] >= min_count] \
                 .sort_values("loss_rate", ascending=False)

        rows.append(grp)

    if rows:
        out = pd.concat(rows, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["field", "value", "count", "loss_rate"])

    return out

def group_feature_importance(imp_df: pd.DataFrame) -> pd.DataFrame:
    bases = []
    for f in imp_df["feature"].astype(str):
        if "_" in f:
            base = f.rsplit("_", 1)[0]
        else:
            base = f
        bases.append(base)
    g = imp_df.copy()
    g["base_field"] = bases
    grouped = (
        g.groupby("base_field", as_index=False)
         .agg(importance_mean_sum=("importance_mean", "sum"),
              importance_mean_max=("importance_mean", "max"),
              features_count=("feature", "count"))
         .sort_values("importance_mean_sum", ascending=False)
    )
    return grouped

# -------- segment performance helper --------
def segment_performance(
    df_labeled: pd.DataFrame,
    probs: np.ndarray,
    thr: float,
    segment_cols: List[str],
    min_rows: int = 20,
    outfile: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute per-segment performance (precision/recall/F1/etc.) for labeled data.
    """
    df = df_labeled.copy()
    df["p_win"] = probs
    df["is_win"] = (df["y"] == 1).astype(int)

    rows = []
    for col in segment_cols:
        if col not in df.columns:
            print(f"[Seg] skip '{col}' (not in dataframe)")
            continue

        for val, sub in df.groupby(col, dropna=False):
            if len(sub) < min_rows:
                continue

            pred = (sub["p_win"] >= thr).astype(int)
            rows.append({
                "segment_field": col,
                "segment_value": val,
                "n": len(sub),
                "win_rate": sub["is_win"].mean(),
                "avg_p_win": sub["p_win"].mean(),
                "precision": precision_score(sub["is_win"], pred, zero_division=0),
                "recall": recall_score(sub["is_win"], pred, zero_division=0),
                "f1": f1_score(sub["is_win"], pred, zero_division=0),
            })

    if not rows:
        out = pd.DataFrame([{
            "info": f"No segments with at least min_rows={min_rows}"
        }])
    else:
        out = pd.DataFrame(rows).sort_values(
            ["segment_field", "win_rate"], ascending=[True, False]
        )

    if outfile is not None:
        out.to_csv(outfile, index=False)
        print(f"[Feedback] wrote -> {outfile}")

    return out

# -------- NEW: loss reasons by segment helper --------
def loss_reasons_by_segment(
    train_df: pd.DataFrame,
    segment_col: str,
    reason_cols: Optional[List[str]] = None,
    min_count: int = 10,
    outfile: Optional[Path] = None,
) -> pd.DataFrame:
    """
    For closed-lost opportunities (y=0), compute loss reason distribution
    within each segment (e.g., Region / Deal Size Bucket).
    """
    if reason_cols is None:
        reason_cols = [
            "Opportunity Reason Lost",
            "Opportunity Reason Lost Detail",
            "Opportunity Reason of Churn",   # NEW
        ]

    if segment_col not in train_df.columns:
        print(f"[LossReason] Skip '{segment_col}' (not in dataframe)")
        out = pd.DataFrame([{
            "info": f"segment_col '{segment_col}' not in dataframe"
        }])
        if outfile is not None:
            out.to_csv(outfile, index=False)
            print(f"[LossReason] wrote -> {outfile}")
        return out

    df = train_df.copy()
    df = df[df["y"] == 0].copy()  # only lost

    if df.empty:
        out = pd.DataFrame([{"info": "No lost opportunities (y=0) in train_df."}])
        if outfile is not None:
            out.to_csv(outfile, index=False)
            print(f"[LossReason] wrote -> {outfile}")
        return out

    rows = []
    for seg_val, seg_df in df.groupby(segment_col, dropna=False):
        seg_total = len(seg_df)
        if seg_total == 0:
            continue

        for rcol in reason_cols:
            if rcol not in seg_df.columns:
                continue

            for reason_val, reason_df in seg_df.groupby(rcol, dropna=False):
                if pd.isna(reason_val):
                    continue

                n = len(reason_df)
                if n < min_count:
                    continue

                rows.append({
                    "segment_field": segment_col,
                    "segment_value": seg_val,
                    "reason_field": rcol,
                    "reason_value": reason_val,
                    "n_lost": n,
                    "pct_of_lost_in_segment": n / seg_total,
                })

    if not rows:
        out = pd.DataFrame([{
            "info": f"No reason groups with count >= {min_count} for segment '{segment_col}'."
        }])
    else:
        out = pd.DataFrame(rows).sort_values(
            ["segment_field", "segment_value", "n_lost"],
            ascending=[True, True, False]
        )

    if outfile is not None:
        out.to_csv(outfile, index=False)
        print(f"[LossReason] wrote -> {outfile}")

    return out

# -------- NEW: overall loss reasons helper --------
def overall_loss_reasons(
    train_df: pd.DataFrame,
    reason_cols: Optional[List[str]] = None,
    min_count: int = 10,
    outfile: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Overall top loss reasons (no segment), just count how often each reason appears
    among y=0 (lost) opportunities.
    """
    if reason_cols is None:
        reason_cols = [
            "Opportunity Reason Lost",
            "Opportunity Reason Lost Detail",
            "Opportunity Reason of Churn",
        ]

    df = train_df.copy()
    df = df[df["y"] == 0].copy()
    if df.empty:
        out = pd.DataFrame([{"info": "No lost opportunities (y=0) in train_df."}])
        if outfile is not None:
            out.to_csv(outfile, index=False)
        return out

    rows = []
    for rcol in reason_cols:
        if rcol not in df.columns:
            continue
        for val, sub in df.groupby(rcol, dropna=False):
            if pd.isna(val):
                continue
            n = len(sub)
            if n < min_count:
                continue
            rows.append({
                "reason_field": rcol,
                "reason_value": val,
                "n_lost": n,
            })

    if not rows:
        out = pd.DataFrame([{
            "info": f"No loss reasons with count >= {min_count}."
        }])
    else:
        out = pd.DataFrame(rows).sort_values(
            ["n_lost"], ascending=False
        )

    if outfile is not None:
        out.to_csv(outfile, index=False)
        print(f"[LossReason] wrote -> {outfile}")

    return out

# -------- NEW: K-Fold model selection helper --------
def cross_validate_models(models: Dict[str, Pipeline],
                          X: pd.DataFrame,
                          y: np.ndarray,
                          n_splits: int = 5) -> Dict[str, Dict[str, float]]:
    """
    Run StratifiedKFold CV for each candidate model (without calibration),
    return mean AUC and PR-AUC per model.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    metrics = {name: {"cv_auc": [], "cv_prauc": []} for name in models}

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        for name, pipe in models.items():
            pipe.fit(Xtr, ytr)
            prob = pipe.predict_proba(Xva)[:, 1]
            auc = roc_auc_score(yva, prob)
            prauc = average_precision_score(yva, prob)
            metrics[name]["cv_auc"].append(auc)
            metrics[name]["cv_prauc"].append(prauc)

    out: Dict[str, Dict[str, float]] = {}
    for name, m in metrics.items():
        out[name] = {
            "cv_auc": float(np.mean(m["cv_auc"])) if m["cv_auc"] else np.nan,
            "cv_prauc": float(np.mean(m["cv_prauc"])) if m["cv_prauc"] else np.nan,
        }
    return out

# -------- NEW: temporal or random train/val split helper --------
def make_train_val_split(train_df: pd.DataFrame,
                         feat_cols: List[str]):

    if USE_TEMPORAL_SPLIT and (TEMP_SPLIT_COL in train_df.columns):
        print(f"[Split] Using temporal split on '{TEMP_SPLIT_COL}' "
              f"with fraction={TEMP_SPLIT_FRACTION}")
        df_tmp = train_df[feat_cols + ["y", TEMP_SPLIT_COL]].copy()
        df_tmp["_ts"] = pd.to_datetime(df_tmp[TEMP_SPLIT_COL], errors="coerce")
        df_tmp = df_tmp.dropna(subset=["_ts"]).sort_values("_ts")

        if len(df_tmp) < 10:
            print("[Split] Too few rows with valid timestamps; falling back to random split.")
        else:
            cut = int(len(df_tmp) * TEMP_SPLIT_FRACTION)
            cut = max(1, min(cut, len(df_tmp) - 1))
            train_idx = df_tmp.index[:cut]
            val_idx   = df_tmp.index[cut:]

            X_train = train_df.loc[train_idx, feat_cols].copy()
            X_val   = train_df.loc[val_idx, feat_cols].copy()
            y_train = train_df.loc[train_idx, "y"].astype(int).values
            y_val   = train_df.loc[val_idx, "y"].astype(int).values
            return X_train, X_val, y_train, y_val

    # default: stratified random split
    print("[Split] Using stratified random train/val split")
    X = train_df[feat_cols].copy()
    y = train_df["y"].astype(int).values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_val, y_train, y_val

# ----------------------- Main -----------------------
def train_and_score():
    # ---------- Load ----------
    raw = load_data()

    # Diagnostics: raw shape & target distribution before mapping
    print(f"[Diag] Raw shape: {raw.shape}")
    if TARGET_COL in raw.columns:
        print(f"[Diag] Raw {TARGET_COL} value counts (raw strings):")
        print(raw[TARGET_COL].astype(str).str.strip().value_counts(dropna=False).head(20))

    raw = map_target(raw)

    # Diagnostics: y distribution after mapping
    print("[Diag] y value counts after mapping (1=Won, 0=Lost, NaN=Other/Open):")
    print(raw["y"].value_counts(dropna=False))

    # Train / score split by target availability
    train_df = raw[raw["y"].isin([0, 1])].copy()
    score_df = raw[~raw["y"].isin([0, 1])].copy()  # unknown/NA goes here

    print(f"[Diag] Trainable rows: {len(train_df)} | To-score rows: {len(score_df)}")

    # 1) Pick candidate columns (leak-safe) on FULL train_df
    feature_space_df = train_df.drop(columns=[TARGET_COL, "y"], errors="ignore")
    cat_cols, num_cols = pick_cols(feature_space_df)
    feat_cols = cat_cols + num_cols

    # Save used features
    used = pd.DataFrame({
        "feature_type": (["categorical"] * len(cat_cols)) + (["numeric"] * len(num_cols)),
        "feature_name": feat_cols
    })
    used.to_csv(FEEDBACK_DIR / "used_features.csv", index=False)
    print(f"[Feedback] wrote -> {FEEDBACK_DIR / 'used_features.csv'}")
    print(f"[Diag] #categorical={len(cat_cols)}  #numeric={len(num_cols)}")

    # X/y on full train_df（用于 K-Fold / split）
    X_full = train_df[feat_cols].copy()
    y_full = train_df["y"].astype(int).values

    # 2) Build preprocessor & models
    pre = build_preprocessor(cat_cols, num_cols)
    models = get_models(pre)

    cv_metrics: Dict[str, Dict[str, float]] = {}
    if USE_KFOLD_MODEL_SELECTION:
        print(f"[CV] Running StratifiedKFold (n_splits={N_FOLDS}) for model selection...")
        cv_metrics = cross_validate_models(models, X_full, y_full, n_splits=N_FOLDS)
        for name, m in cv_metrics.items():
            print(f"[CV] {name}: cv_auc={m['cv_auc']:.4f}, cv_prauc={m['cv_prauc']:.4f}")
    else:
        print("[CV] K-Fold model selection disabled; using single holdout metrics only.")

    X_train, X_val, y_train, y_val = make_train_val_split(train_df, feat_cols)

    def nonempty_cols(df):
        return [c for c in df.columns if df[c].notna().any()]

    keep_cols = sorted(set(nonempty_cols(X_train)).intersection(nonempty_cols(X_val)))
    X_train = X_train[keep_cols].copy()
    X_val   = X_val[keep_cols].copy()
    cat_cols = [c for c in cat_cols if c in keep_cols]
    num_cols = [c for c in num_cols if c in keep_cols]
    feat_cols = cat_cols + num_cols  # finalized training feature order

    # 4) Rebuild preprocessor & models AFTER columns are finalized
    pre = build_preprocessor(cat_cols, num_cols)
    models = get_models(pre)

    rows = []
    best_name, best_cal, best_val_prob, best_thr_prec = None, None, None, None
    best_sel_metric = -1.0

    for name, pipe in models.items():
        cal = CalibratedClassifierCV(estimator=pipe, method="isotonic", cv=3)
        cal.fit(X_train, y_train)

        val_prob = cal.predict_proba(X_val)[:, 1]
        auc   = roc_auc_score(y_val, val_prob)
        prauc = average_precision_score(y_val, val_prob)
        brier = brier_score_loss(y_val, val_prob)

        # thresholds
        metrics_05 = evaluate_at_threshold(y_val, val_prob, 0.5)
        thr_prec, hit = pick_precision_first_threshold(y_val, val_prob, TARGET_PRECISION)
        metrics_prec = evaluate_at_threshold(y_val, val_prob, thr_prec)

        # attach CV metrics if available
        cv_auc   = cv_metrics.get(name, {}).get("cv_auc", np.nan)
        cv_prauc = cv_metrics.get(name, {}).get("cv_prauc", np.nan)

        row = {
            "model": name,
            "cv_auc": round(float(cv_auc), 4) if not np.isnan(cv_auc) else None,
            "cv_prauc": round(float(cv_prauc), 4) if not np.isnan(cv_prauc) else None,
            "val_auc": round(float(auc), 4),
            "val_prauc": round(float(prauc), 4),
            "brier": round(float(brier), 4),
            "thr_at_0p5": 0.5,
            "precision@0p5": round(metrics_05["precision"], 4),
            "recall@0p5": round(metrics_05["recall"], 4),
            "f1@0p5": round(metrics_05["f1"], 4),
            "balacc@0p5": round(metrics_05["balacc"], 4),
            "cm@0p5": json.dumps(metrics_05["cm"]),
            "target_precision": TARGET_PRECISION,
            "thr_at_target_precision": round(thr_prec, 4),
            "precision@target": round(metrics_prec["precision"], 4),
            "recall@target": round(metrics_prec["recall"], 4),
            "f1@target": round(metrics_prec["f1"], 4),
            "balacc@target": round(metrics_prec["balacc"], 4),
            "cm@target": json.dumps(metrics_prec["cm"]),
            "met_target_precision": bool(hit),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
        }

        rows.append(row)

        if USE_KFOLD_MODEL_SELECTION and not np.isnan(cv_auc):
            sel_metric = cv_auc
        else:
            sel_metric = auc

        if sel_metric > best_sel_metric:
            best_sel_metric = sel_metric
            best_name = name
            best_cal = cal
            best_val_prob = val_prob
            best_thr_prec = thr_prec

    # write metrics
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_METRICS, index=False)
    print(f"[Metrics] wrote -> {OUT_METRICS}")
    print(metrics_df)

    # ---------- Feedback pack on validation ----------
    # 1) error analysis: FPs and FNs at the precision-first threshold
    val_pred = (best_val_prob >= best_thr_prec).astype(int)
    err_mask = val_pred != y_val

    err_df = X_val.copy()
    err_df["y_true"] = y_val
    err_df["p_win"] = best_val_prob
    err_df["y_pred"] = val_pred
    err_df["error_type"] = np.where((val_pred == 1) & (y_val == 0), "FP", "FN")

    err_df = err_df.loc[err_mask].copy()

    if len(err_df) == 0:
        pd.DataFrame([{
            "info": "No misclassifications at the precision-first threshold on the validation split."
        }]).to_csv(FEEDBACK_DIR / "error_analysis_val.csv", index=False)
        print(f"[Feedback] wrote -> {FEEDBACK_DIR / 'error_analysis_val.csv'} (no errors)")
    else:
        reason_series = pd.Series([""] * len(err_df), index=err_df.index)
        if best_name == "LogisticRegression":
            try:
                reason_series = pd.Series(
                    per_example_reason_codes_logreg(best_cal, X_val.loc[err_mask], cat_cols, topk=3),
                    index=err_df.index
                )
            except Exception as e:
                print(f"[Warn] per-example reasons failed: {e}")

        err_df["reasons"] = reason_series
        err_df.to_csv(FEEDBACK_DIR / "error_analysis_val.csv", index=False)
        print(f"[Feedback] wrote -> {FEEDBACK_DIR / 'error_analysis_val.csv'}")

    # 2) global feature importance (permutation on validation)
    try:
        base = get_underlying_estimator(best_cal)
        base.fit(X_train, y_train)

        r = permutation_importance(
            base, X_val, y_val,
            n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
        )

        pre_fitted = base.named_steps["prep"]
        feat_names = extract_ohe_feature_names(pre_fitted, cat_cols)

        k = min(len(feat_names), r.importances_mean.shape[0])
        if k == 0:
            pd.DataFrame([{"info": "Feature importance unavailable (no aligned features)."}]) \
              .to_csv(FEEDBACK_DIR / "feature_importance.csv", index=False)
        else:
            imp = pd.DataFrame({
                "feature": feat_names[:k],
                "importance_mean": r.importances_mean[:k],
                "importance_std": r.importances_std[:k]
            }).sort_values("importance_mean", ascending=False)
            imp.to_csv(FEEDBACK_DIR / "feature_importance.csv", index=False)
            group_feature_importance(imp).to_csv(FEEDBACK_DIR / "feature_importance_grouped.csv", index=False)
        print(f"[Feedback] wrote -> {FEEDBACK_DIR / 'feature_importance.csv'}")
        print(f"[Feedback] wrote -> {FEEDBACK_DIR / 'feature_importance_grouped.csv'}")
    except Exception as e:
        print(f"[Warn] permutation importance failed: {e}")
        pd.DataFrame([{"info": f"Permutation importance failed: {e}"}]) \
          .to_csv(FEEDBACK_DIR / "feature_importance.csv", index=False)

    # 3) calibration table (deciles)
    calib = calibration_table(y_val, best_val_prob, bins=10)
    calib.to_csv(FEEDBACK_DIR / "calibration_table.csv", index=False)
    print(f"[Feedback] wrote -> {FEEDBACK_DIR / 'calibration_table.csv'}")

    # 4) loss drivers on training set (closed-lost categories with enough support)
    dynamic_min_count = 1000
    loss_drv = build_loss_drivers(train_df, min_count=dynamic_min_count)
    if loss_drv.empty:
        pd.DataFrame([{
            "info": f"No groups met support >= {dynamic_min_count}. Try lowering min_count or add data."
        }]).to_csv(FEEDBACK_DIR / "loss_drivers.csv", index=False)
    else:
        loss_drv.to_csv(FEEDBACK_DIR / "loss_drivers.csv", index=False)
    print(f"[Feedback] wrote -> {FEEDBACK_DIR / 'loss_drivers.csv'}")

    # NEW: overall top loss reasons (no segment)
    overall_loss_reasons(
        train_df,
        reason_cols=[
            "Opportunity Reason Lost",
            "Opportunity Reason Lost Detail",
            "Opportunity Reason of Churn",
        ],
        min_count=10,
        outfile=FEEDBACK_DIR / "loss_reasons_overall.csv",
    )

    # ---------- segment analysis (regions, deal types, combos) ----------
    if best_cal is not None and len(train_df) > 0:
        X_all = train_df[feat_cols].copy()
        X_all = X_all[[c for c in feat_cols if c in X_all.columns]]
        prob_all = best_cal.predict_proba(X_all)[:, 1]

        # Regions: EMEA / APAC / AMER etc.
        region_cols = ["Account Region", "Assigned to Region"]
        segment_performance(
            train_df,
            prob_all,
            best_thr_prec,
            segment_cols=region_cols,
            min_rows=10,
            outfile=FEEDBACK_DIR / "segment_perf_regions.csv",
        )

        # Deal types: New / Renewal / Expansion etc.
        dealtype_cols = ["Sale Type", "Opportunity Record Type"]
        segment_performance(
            train_df,
            prob_all,
            best_thr_prec,
            segment_cols=dealtype_cols,
            min_rows=10,
            outfile=FEEDBACK_DIR / "segment_perf_dealtypes.csv",
        )

        # Region × Sale Type combo
        if ("Account Region" in train_df.columns) and ("Sale Type" in train_df.columns):
            combo_df = train_df.copy()
            combo_df["Region x SaleType"] = (
                combo_df["Account Region"].astype(str)
                + " | "
                + combo_df["Sale Type"].astype(str)
            )
            segment_performance(
                combo_df,
                prob_all,
                best_thr_prec,
                segment_cols=["Region x SaleType"],
                min_rows=10,
                outfile=FEEDBACK_DIR / "segment_perf_region_x_saletype.csv",
            )
        else:
            print("[Seg] Skipping Region x SaleType combo (columns missing).")

        # ---------- NEW: loss reasons by Region ----------
        region_col = None
        for cand in ["Account Region", "Assigned to Region"]:
            if cand in train_df.columns:
                region_col = cand
                break

        if region_col is not None:
            loss_reasons_by_segment(
                train_df,
                segment_col=region_col,
                reason_cols=[
                    "Opportunity Reason Lost",
                    "Opportunity Reason Lost Detail",
                    "Opportunity Reason of Churn",
                ],
                min_count=5,
                outfile=FEEDBACK_DIR / "loss_reasons_by_region.csv",
            )
        else:
            print("[LossReason] No region column found for loss analysis.")

        # NEW: loss reasons by Country
        country_col = None
        for cand in ["Account Country", "Account Shipping Country"]:
            if cand in train_df.columns:
                country_col = cand
                break

        if country_col is not None:
            loss_reasons_by_segment(
                train_df,
                segment_col=country_col,
                reason_cols=[
                    "Opportunity Reason Lost",
                    "Opportunity Reason Lost Detail",
                    "Opportunity Reason of Churn",
                ],
                min_count=5,
                outfile=FEEDBACK_DIR / "loss_reasons_by_country.csv",
            )
        else:
            print("[LossReason] No country column found for loss analysis.")

        # ---------- NEW: loss reasons by Deal Size (bucketed) ----------
        amount_col = first_present(train_df, AMOUNT_CANDIDATES)
        if (amount_col is not None) and (amount_col in train_df.columns):
            tmp = train_df.copy()
            tmp = tmp[tmp[amount_col].notna()].copy()
            try:
                tmp["Deal Size Bucket"] = pd.qcut(
                    tmp[amount_col],
                    q=4,
                    labels=["Q1-Small", "Q2-Medium", "Q3-Large", "Q4-XL"],
                    duplicates="drop",
                )
                loss_reasons_by_segment(
                    tmp,
                    segment_col="Deal Size Bucket",
                    reason_cols=[
                        "Opportunity Reason Lost",
                        "Opportunity Reason Lost Detail",
                        "Opportunity Reason of Churn",
                    ],
                    min_count=5,
                    outfile=FEEDBACK_DIR / "loss_reasons_by_dealsize.csv",
                )
            except Exception as e:
                print(f"[LossReason] Deal size bucketing failed: {e}")
        else:
            print("[LossReason] No amount column found for deal size analysis.")

    # ---------- Score unknown/open opportunities ----------
    if best_cal is not None and not score_df.empty:
        needed = feat_cols  # finalized training feature order

        # Build X_score with same column set/order; create missing as NaN
        X_score = score_df[[c for c in needed if c in score_df.columns]].copy()
        for c in needed:
            if c not in X_score.columns:
                X_score[c] = np.nan
        X_score = X_score[needed]

        prob = best_cal.predict_proba(X_score)[:, 1]

        id_col    = first_present(score_df, ID_CANDIDATES)
        name_col  = first_present(score_df, NAME_CANDIDATES)
        owner_col = first_present(score_df, OWNER_CANDIDATES)
        amount_col= first_present(score_df, AMOUNT_CANDIDATES)

        keep_cols_raw = [id_col, name_col, STAGE_COL, owner_col, CLOSE_DATE_COL, amount_col]
        keep_cols = [c for c in keep_cols_raw if c and c in score_df.columns]

        out = score_df[keep_cols].copy() if keep_cols else pd.DataFrame(index=score_df.index)

        out.rename(columns={id_col: "Opportunity ID",
                            name_col: "Opportunity Name",
                            owner_col: "Owner",
                            amount_col: "Amount"}, inplace=True)
        out["win_prob"] = prob
        out["pred_at_prec_thr"] = (out["win_prob"] >= best_thr_prec).astype(int)
        out = out.sort_values("win_prob", ascending=False)
        out.to_csv(OUT_PROBS, index=False)
        print(f"[Scoring] wrote -> {OUT_PROBS}")
    else:
        print("[Info] No unknown/open opportunities to score.")

    # ---------- Model registry snapshot ----------
    try:
        registry_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "excel_path": str(EXCEL_PATH),
            "sheet": SHEET,
            "target_col": TARGET_COL,
            "best_model": best_name,
            "use_kfold_model_selection": USE_KFOLD_MODEL_SELECTION,
            "n_folds": N_FOLDS if USE_KFOLD_MODEL_SELECTION else None,
            "use_temporal_split": USE_TEMPORAL_SPLIT,
            "temp_split_col": TEMP_SPLIT_COL if USE_TEMPORAL_SPLIT else None,
            "metrics": metrics_df.to_dict(orient="records"),
            "features": feat_cols,
            "n_train_rows": int(len(train_df)),
            "n_score_rows": int(len(score_df)),
        }
        with MODEL_REGISTRY_PATH.open("w", encoding="utf-8") as f:
            json.dump(registry_entry, f, indent=2)
        print(f"[Registry] wrote -> {MODEL_REGISTRY_PATH}")
    except Exception as e:
        print(f"[Warn] failed to write model registry: {e}")

if __name__ == "__main__":
    train_and_score()
