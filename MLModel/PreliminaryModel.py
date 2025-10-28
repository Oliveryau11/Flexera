"""
Win/Loss Modeling Pipeline (Flexera dummy data)
------------------------------------------------
Reads Dummy Data.xlsx (sheet: "Opportunities"), uses "CRO Win" as label,
trains Logistic Regression and Random Forest with preprocessing, evaluates,
and scores open opportunities (Unknown).

Outputs:
- win_probabilities.csv
- Console metrics table
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

EXCEL_PATH = Path("RealDummyData.xlsx")
SHEET = "Opportunities"

def load_data():
    df_raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET)
    df = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()
    return df

def prepare_training(df):
    df["y"] = df["CRO Win"].map({"Yes": 1, "No": 0})
    train_df = df.loc[df["y"].isin([0, 1])].copy()
    
    candidate_cats = [
        "Stage", "Stage Number", "Opportunity Stage Prior to Close",
        "Primary Partner Sales Stage",
        "Forecast Category", "Rep Forecasted", "Opportunity Rep Forecasted",
        "Opportunity VP Forecasted", "Opportunity RVP Forecasted", "Opportunity CRO Forecasted",
        "Lead Source", "Lead Source Detail", "Opportunity Type",
        "Account Shipping Country", "Account Country", "Account Region",
        "Opportunity SE Process Stage",
        "Opportunity Reason Lost", "Opportunity Reason Lost Detail"
    ]
    cat_cols = [c for c in candidate_cats if c in train_df.columns]

    num_cols = [c for c in train_df.columns if train_df[c].dtype.kind in "if" and c not in ["y"]]
    leak_like = [c for c in num_cols if any(k in c.lower() for k in ["id", "win", "loss"])]
    num_cols = [c for c in num_cols if c not in leak_like]

    cat_cols = list(dict.fromkeys(cat_cols))
    num_cols = [c for c in dict.fromkeys(num_cols) if c not in cat_cols]

    X = train_df[cat_cols + num_cols].copy()
    y = train_df["y"].astype(int).copy()
    return X, y, cat_cols, num_cols

def build_pipelines(cat_cols, num_cols):
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )
    logreg = Pipeline([("prep", preprocess),
                       ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))])
    rf = Pipeline([("prep", preprocess),
                   ("clf", RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced"))])
    return logreg, rf

def evaluate(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return dict(
        AUC=float(roc_auc_score(y_test, probs)),
        Accuracy=float(accuracy_score(y_test, preds)),
        Precision=float(precision_score(y_test, preds, zero_division=0)),
        Recall=float(recall_score(y_test, preds, zero_division=0)),
        n_test=int(len(y_test)),
    )

def train_and_score():
    df = load_data()
    X, y, cat_cols, num_cols = prepare_training(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    logreg, rf = build_pipelines(cat_cols, num_cols)
    logreg.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    m_log = evaluate(logreg, X_test, y_test)
    m_rf = evaluate(rf, X_test, y_test)

    metrics = pd.DataFrame([
        dict(model="LogisticRegression", **m_log),
        dict(model="RandomForest", **m_rf),
    ])
    print("\n=== Preliminary Model Metrics ===")
    print(metrics.round(3))

    best = logreg if m_log["AUC"] >= m_rf["AUC"] else rf

    try:
        feature_names = best.named_steps["prep"].get_feature_names_out()
        if "LogisticRegression" in str(best.named_steps["clf"]):
            coefs = best.named_steps["clf"].coef_[0]
            importances = pd.Series(np.abs(coefs), index=feature_names).sort_values(ascending=False)
        else:
            fi = best.named_steps["clf"].feature_importances_
            importances = pd.Series(fi, index=feature_names).sort_values(ascending=False)
        print("\n=== Top 25 Predictive Features ===")
        print(importances.head(25))
    except Exception as e:
        print("\n[Warn] Could not extract feature importances:", e)

    df["y"] = df["CRO Win"].map({"Yes": 1, "No": 0})
    unlabeled = df.loc[~df["y"].isin([0, 1])].copy()
    if not unlabeled.empty:
        avail = [c for c in (cat_cols + num_cols) if c in unlabeled.columns]
        X_unl = unlabeled[avail]
        win_prob = best.predict_proba(X_unl)[:, 1]
        # Keep ID purely for output, not as a feature
        id_cols = [c for c in ["Id", "Opportunity Id", "Opportunity ID"] if c in unlabeled.columns]
        display_cols = id_cols + [c for c in ["Name", "Stage", "Amount", "Opportunity Line ACV USD"] if c in unlabeled.columns]

        scored = unlabeled[display_cols].copy() if display_cols else unlabeled.copy()
        scored["Win Probability"] = win_prob

        scored = scored.sort_values("Win Probability", ascending=False)
        out_path = Path("win_probabilities.csv")
        scored.to_csv(out_path, index=False)
        print(f"\n[Saved] Scored open opportunities → {out_path}")
    else:
        print("\n[Info] No unlabeled (Unknown) opportunities to score.")

if __name__ == "__main__":
   if __name__ == "__main__":
    train_and_score()

    # === Evaluate model accuracy on Closed Won/Lost deals ===
    try:
        df_eval = pd.read_csv("win_probabilities.csv")

        # Identify actual outcomes from Stage
        df_eval["actual"] = df_eval["Stage"].apply(
            lambda x: 1 if "Closed Won" in str(x) else (0 if "Closed Lost" in str(x) else None)
        )

        # Drop rows without known outcomes
        df_eval = df_eval.dropna(subset=["actual"])

        # Convert probabilities to binary predictions
        df_eval["predicted"] = (df_eval["Win Probability"] >= 0.5).astype(int)

        # Compute metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
        )

        acc = accuracy_score(df_eval["actual"], df_eval["predicted"])
        prec = precision_score(df_eval["actual"], df_eval["predicted"], zero_division=0)
        rec = recall_score(df_eval["actual"], df_eval["predicted"], zero_division=0)
        auc = roc_auc_score(df_eval["actual"], df_eval["Win Probability"])
        cm = confusion_matrix(df_eval["actual"], df_eval["predicted"])

        # Print to console
        print("\n=== Win/Loss Model Evaluation on Closed Deals ===")
        print(f"Accuracy:  {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall:    {rec:.3f}")
        print(f"AUC:       {auc:.3f}")
        print("\nConfusion Matrix:")
        print(cm)

        # Save metrics to CSV
        metrics_df = pd.DataFrame([{
            "Accuracy": round(acc, 3),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "AUC": round(auc, 3),
            "True Negatives": cm[0, 0],
            "False Positives": cm[0, 1],
            "False Negatives": cm[1, 0],
            "True Positives": cm[1, 1],
        }])

        metrics_df.to_csv("model_metrics.csv", index=False)
        print("\n[Saved] Model metrics → model_metrics.csv")

    except Exception as e:
        print(f"\n[Warn] Could not evaluate model automatically: {e}")
