"""Crop-yield estimation (hg/ha) with quantum and classical regressors.

Dataset: Kaggle "Crop Yield Prediction Dataset" (yield_df.csv; FAO + World Bank), 28,242 rows,
101 countries (Area), 10 crops (Item), 1990-2013, rainfall, pesticides, temperature.

Inputs for the quantum models (one per qubit):
    crop-type and country target encodings (mean log-yield in the training split), avg_temp, log rainfall,
    log pesticides, year  -> PCA to n_qubits -> [0, pi].
Target: log(yield), standardised. Metrics are reported on the original hg/ha scale.

    python crop_yield.py                       # QNNR, VQR, SVR, Linear, MLP
    python crop_yield.py --qubits 4 --vqr-train 2000
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

from qml_regressors import QNNR

ROOT = Path(__file__).resolve().parent.parent
TARGET = "hg/ha_yield"


def load(path=ROOT / "Dataset" / "yield_df.csv"):
    df = pd.read_csv(path).drop(columns=["Unnamed: 0"], errors="ignore")
    return df


def features(train, other):
    """Target-encode Area/Item with training-split means only (no leakage)."""
    logy = np.log(train[TARGET])
    enc = {c: logy.groupby(train[c]).mean() for c in ("Item", "Area")}
    prior = logy.mean()

    def f(df):
        return np.column_stack([
            df["Item"].map(enc["Item"]).fillna(prior), df["Area"].map(enc["Area"]).fillna(prior),
            df["avg_temp"], np.log1p(df["average_rain_fall_mm_per_year"]), np.log1p(df["pesticides_tonnes"]),
            df["Year"],
        ])

    return f(train), f(other)


class VQR:
    """Pure variational quantum regressor (qiskit-machine-learning): ZZ feature map + RealAmplitudes,
    prediction = <Z...Z> scaled to the target range, trained with COBYLA on a random training subset."""

    def __init__(self, n_qubits=4, reps=3, maxiter=150, n_train=2000, seed=42):
        self.n_qubits, self.reps, self.maxiter, self.n_train, self.seed = n_qubits, reps, maxiter, n_train, seed

    def fit(self, X, y):
        from qiskit.circuit.library import real_amplitudes, zz_feature_map
        from qiskit.primitives import StatevectorEstimator
        from qiskit_machine_learning.algorithms import VQR as _VQR
        from qiskit_machine_learning.optimizers import COBYLA

        idx = np.random.default_rng(self.seed).choice(len(X), min(self.n_train, len(X)), replace=False)
        self.scale_ = np.abs(y).max() * 1.05  # <Z...Z> lies in [-1, 1]
        self.losses = []
        self.model = _VQR(feature_map=zz_feature_map(self.n_qubits, reps=1, entanglement="linear"),
                          ansatz=real_amplitudes(self.n_qubits, reps=self.reps, entanglement="linear"),
                          optimizer=COBYLA(maxiter=self.maxiter), estimator=StatevectorEstimator(),
                          callback=lambda w, v: self.losses.append(float(v)))
        self.model.fit(X[idx], y[idx] / self.scale_)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X)).ravel() * self.scale_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=4)
    ap.add_argument("--qnnr-layers", type=int, default=4)
    ap.add_argument("--qnnr-epochs", type=int, default=30)
    ap.add_argument("--vqr-train", type=int, default=2000)
    ap.add_argument("--vqr-maxiter", type=int, default=150)
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load()
    train, test = train_test_split(df, test_size=args.test_size, random_state=args.seed)
    Ftr, Fte = features(train, test)
    ytr_log, yte = np.log(train[TARGET].to_numpy()), test[TARGET].to_numpy()
    ymu, ysd = ytr_log.mean(), ytr_log.std()
    ytr = (ytr_log - ymu) / ysd
    to_hg = lambda z: np.exp(z * ysd + ymu)

    reduce = Pipeline([("std", StandardScaler()), ("pca", PCA(n_components=args.qubits)),
                       ("angle", MinMaxScaler(feature_range=(0, np.pi)))]).fit(Ftr)
    Qtr, Qte = reduce.transform(Ftr), reduce.transform(Fte)
    std = StandardScaler().fit(Ftr)
    Str, Ste = std.transform(Ftr), std.transform(Fte)

    models = {
        "QNNR": (QNNR(args.qubits, args.qnnr_layers, args.qnnr_epochs, seed=args.seed), "raw"),
        "VQR": (VQR(args.qubits, maxiter=args.vqr_maxiter, n_train=args.vqr_train, seed=args.seed), "quantum"),
        "Classical SVR": (SVR(C=10, epsilon=0.05), "std"),
        "Linear Regressor": (LinearRegression(), "std"),
        "Neural Network Regressor": (MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=args.seed), "std"),
    }
    inputs = {"raw": (Ftr, Fte), "quantum": (Qtr, Qte), "std": (Str, Ste)}  # QNNR does its own PCA -> angles
    results, preds = {}, {}
    for name, (m, kind) in models.items():
        a, b = inputs[kind]
        t0 = time.time()
        m.fit(a, ytr.reshape(-1, 1) if name == "QNNR" else ytr)
        t_train = time.time() - t0
        t0 = time.time()
        p_tr, p_te = (to_hg(np.asarray(m.predict(x)).ravel()) for x in (a, b))
        t_test = time.time() - t0
        ytr_hg = train[TARGET].to_numpy()
        results[name] = {"train_seconds": round(t_train, 2), "test_seconds": round(t_test, 2),
                         "train_r2": float(r2_score(ytr_hg, p_tr)), "train_rmse": float(np.sqrt(mean_squared_error(ytr_hg, p_tr))),
                         "test_r2": float(r2_score(yte, p_te)), "test_rmse": float(np.sqrt(mean_squared_error(yte, p_te)))}
        preds[name] = p_te
        r = results[name]
        print(f"{name:26s} train R2 {r['train_r2']:.3f}  test R2 {r['test_r2']:.3f}  test RMSE {r['test_rmse']:.0f}  ({t_train:.1f}s)")

    out = ROOT / "Results"
    (out / "crop_yield_results.json").write_text(json.dumps({"config": vars(args), "n_train": len(train),
                                                             "n_test": len(test), "results": results}, indent=2))
    np.savez(out / "crop_yield_predictions.npz", y=yte, **{k.replace(" ", "_"): v for k, v in preds.items()})


if __name__ == "__main__":
    main()
