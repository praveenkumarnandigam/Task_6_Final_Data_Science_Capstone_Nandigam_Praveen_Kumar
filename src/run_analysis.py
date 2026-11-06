"""
YuvaIntern Task 6 - Final Data Science Capstone
Student performance exploration and pass/fail classification.

Run from project root:
    python src/run_analysis.py
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42

def load_data():
    """Fetch official UCI Student Performance data and combine features/target."""
    repo = fetch_ucirepo(id=320)
    features = repo.data.features.copy()
    targets = repo.data.targets.copy()
    df = pd.concat([features.reset_index(drop=True),
                    targets.reset_index(drop=True)], axis=1)
    if "G3" not in df.columns:
        raise ValueError(
            "The UCI response did not contain the expected G3 target. "
            "Inspect repo.data.features/targets and update the target mapping."
        )
    return df

def main():
    print("Downloading Student Performance data from UCI...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print("\nFirst rows:\n", df.head())
    print("\nMissing values (top):\n", df.isna().sum().sort_values(ascending=False).head(10))
    print("Duplicate rows:", int(df.duplicated().sum()))

    # Remove exact duplicate records and create a binary target.
    df = df.drop_duplicates().copy()
    df["pass"] = (pd.to_numeric(df["G3"], errors="coerce") >= 10).astype(int)
    df = df.dropna(subset=["G3"])
    df.to_csv(OUT / "student_performance_cleaned.csv", index=False)

    # EDA charts
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="pass")
    plt.xticks([0, 1], ["Fail (<10)", "Pass (>=10)"])
    plt.title("Distribution of final pass/fail outcome")
    plt.tight_layout()
    plt.savefig(OUT / "pass_fail_distribution.png", dpi=160)
    plt.close()

    if "studytime" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x="studytime", y="G3", errorbar=None)
        plt.title("Average final grade by weekly study-time category")
        plt.xlabel("Study-time category")
        plt.ylabel("Average final grade (G3)")
        plt.tight_layout()
        plt.savefig(OUT / "average_grade_by_studytime.png", dpi=160)
        plt.close()

    if "absences" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="pass", y="absences")
        plt.xticks([0, 1], ["Fail", "Pass"])
        plt.title("Absences by final pass/fail outcome")
        plt.tight_layout()
        plt.savefig(OUT / "absences_by_outcome.png", dpi=160)
        plt.close()

    # Exclude grades G1/G2/G3 from predictors to reduce target leakage.
    excluded = {"G1", "G2", "G3", "pass"}
    X = df.drop(columns=[c for c in excluded if c in df.columns])
    y = df["pass"]
    if y.nunique() < 2:
        raise ValueError("The derived pass/fail target contains only one class.")

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE,
            class_weight="balanced", min_samples_leaf=2
        ),
    }
    rows = []
    reports = []
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba) if y_test.nunique() == 2 else np.nan,
        }
        rows.append(metrics)
        reports.append(f"\n{name}\n" + classification_report(y_test, pred, zero_division=0))
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion matrix - {name}")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(OUT / f"confusion_matrix_{safe_name}.png", dpi=160)
        plt.close()

    metrics_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    metrics_df.to_csv(OUT / "model_metrics.csv", index=False)

    # Compact descriptive statistics
    numeric_summary = df.select_dtypes(include="number").describe().round(2)
    numeric_summary.to_csv(OUT / "numeric_summary.csv")

    best = metrics_df.iloc[0]
    summary = [
        "STUDENT PERFORMANCE CAPSTONE - RUN SUMMARY",
        f"Rows after exact duplicate removal: {len(df)}",
        f"Number of input features (excluding G1, G2, G3): {X.shape[1]}",
        f"Pass rate: {y.mean():.3f}",
        "",
        "Model metrics:",
        metrics_df.to_string(index=False),
        "",
        f"Highest F1 in this run: {best['model']} (F1={best['f1']:.3f})",
        "Interpretation: compare precision and recall as well as accuracy. "
        "A higher score is not proof of causal relationships or real-world readiness.",
        "",
        "Leakage control: G1 and G2 are excluded from model inputs.",
        "Split: stratified 80/20 train/test split, random_state=42.",
        *reports,
    ]
    (OUT / "run_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n", "\n".join(summary[:15]))
    print(f"\nCompleted. Outputs saved in: {OUT}")

if __name__ == "__main__":
    main()
