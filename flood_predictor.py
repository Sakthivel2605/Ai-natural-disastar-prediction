"""Flood risk prediction with logistic regression on SIMULATED weather data."""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FEATURES = ["rainfall_24h_mm", "river_level_m", "soil_moisture_pct"]


def make_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    rain = rng.gamma(2.0, 25.0, n)             # mm in 24 h
    river = rng.normal(3.0, 0.8, n).clip(0.5)  # metres
    soil = rng.uniform(10, 95, n)              # %
    # Made-up rule: heavy rain on wet soil with a high river means flood.
    score = 0.035 * rain + 0.9 * river + 0.03 * soil - 6.5
    prob = 1 / (1 + np.exp(-2.0 * score))
    y = rng.random(n) < prob
    return np.column_stack([rain, river, soil]), y.astype(int)


def train(seed=0):
    X, y = make_data(seed=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {"accuracy": accuracy_score(y_te, pred),
               "precision": precision_score(y_te, pred),
               "recall": recall_score(y_te, pred)}
    return model, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rain", type=float, default=120, help="rainfall in last 24 h (mm)")
    p.add_argument("--river", type=float, default=4.2, help="river level (m)")
    p.add_argument("--soil", type=float, default=80, help="soil moisture (%%)")
    a = p.parse_args()
    model, m = train()
    print("Test metrics on simulated data:", {k: round(v, 2) for k, v in m.items()})
    risk = model.predict_proba([[a.rain, a.river, a.soil]])[0][1]
    level = "HIGH" if risk > 0.7 else "MEDIUM" if risk > 0.4 else "LOW"
    print(f"Flood risk for rain={a.rain} mm, river={a.river} m, soil={a.soil}%: {risk:.0%} ({level})")


if __name__ == "__main__":
    main()
