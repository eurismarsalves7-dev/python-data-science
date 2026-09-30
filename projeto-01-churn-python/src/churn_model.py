"""
Análise Preditiva de Evasão de Clientes (Churn) - Telco Customer Churn
Autor: Eurismar Silveira Alves
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def explore_data(df: pd.DataFrame) -> None:
    print(df.info())
    print(df["Churn"].value_counts(normalize=True))

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churn")
    plt.title("Distribuição de Churn")
    plt.tight_layout()
    plt.savefig("churn_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="tenure")
    plt.title("Tempo de Permanência vs Churn")
    plt.tight_layout()
    plt.savefig("tenure_vs_churn.png")
    plt.close()


def preprocess(df: pd.DataFrame):
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    cat_cols = df.select_dtypes(include="object").columns
    cat_cols = [c for c in cat_cols if c != "customerID"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop(columns=["Churn", "customerID"], errors="ignore")
    y = df["Churn"]

    scaler = StandardScaler()
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y, encoders, scaler


def train_models(X_train, y_train):
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate_model(model, X_test, y_test, name: str):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    print(metrics)
    return metrics


def tune_gradient_boosting(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 4, 5],
    }
    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    grid = GridSearchCV(gb, param_grid, scoring="f1", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Melhores parâmetros:", grid.best_params_)
    return grid.best_estimator_


def feature_importance(model, X, top_n: int = 10):
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh")
    plt.title("Importância das Features - Churn")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

    return importances


def main():
    df = load_data("data/telco_churn.csv")
    explore_data(df)

    X, y, encoders, scaler = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = train_models(X_train, y_train)
    results = [
        evaluate_model(m, X_test, y_test, n)
        for n, m in models.items()
    ]

    best_model = tune_gradient_boosting(X_train, y_train)
    evaluate_model(best_model, X_test, y_test, "GradientBoosting (Tuned)")
    feature_importance(best_model, X)

    pd.DataFrame(results).to_csv("model_results.csv", index=False)
    print("\nAnálise concluída. Resultados salvos em model_results.csv")


if __name__ == "__main__":
    main()
