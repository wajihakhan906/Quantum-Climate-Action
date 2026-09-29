"""Figures: dataset overview, test predictions, training curves, VQC circuit."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import FEATURES, load
from quantum_layer import VQCLayer

ROOT = Path(__file__).resolve().parent.parent
UNITS = {"meantemp": "°C", "humidity": "%", "wind_speed": "km/h", "meanpressure": "hPa"}


def main():
    res = json.load(open(ROOT / "Results" / "results.json"))
    cfg = res["config"]
    d = dict(np.load(ROOT / "Results" / "predictions.npz"))
    dates = pd.to_datetime(d["dates"])
    all_results = dict(res["results"])
    qml_path = ROOT / "Results" / "results_qml.json"
    if qml_path.exists():
        all_results |= json.load(open(qml_path))["results"]
        q = np.load(ROOT / "Results" / "predictions_qml.npz")
        pca_name = next((k for k in all_results if "PCA" in k), None)
        d |= {"QSVR": q["QSVR"], "QNNR": q["QNNR"], "SVR (classical)": q["SVR"]}
        if pca_name:
            d[pca_name] = q["SVR_pca"]
    styles = {"CLSTM": "#4C72B0", "QLSTM": "#C44E52", "QSVR": "#55A868", "QNNR": "#8172B2", "SVR (classical)": "#999999"}
    styles |= {k: "#CCCCCC" for k in all_results if "PCA" in k}

    df = load("Train")
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    for ax, f in zip(axes, FEATURES):
        ax.plot(df["date"], df[f], lw=0.8)
        ax.set_ylabel(f"{f}\n({UNITS[f]})", fontsize=8)
    axes[0].set_title("Daily Delhi Climate, 2013-2017 (training period)")
    fig.tight_layout(); fig.savefig(ROOT / "Figures" / "dataset.png", dpi=200)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (j, f) in zip(axes.ravel(), enumerate(FEATURES)):
        ax.plot(dates, d["y"][:, j], "k-", lw=1.6, label="Actual")
        for name, c in styles.items():
            if name not in d or "PCA" in name:  # keep the time-series panels readable
                continue
            r2 = all_results[name]["test"][f]["r2"]
            ax.plot(dates, d[name][:, j], color=c, lw=1.1, alpha=0.9, label=f"{name} (R² {r2:.2f})")
        ax.set_title(f"{f} ({UNITS[f]})"); ax.legend(fontsize=8); ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    fig.suptitle(f"One-day-ahead forecasts on the 114-day test period (window = {cfg['seq_len']} days)")
    fig.tight_layout(); fig.savefig(ROOT / "Figures" / "test_predictions.png", dpi=200)

    names = [n for n in styles if n in all_results]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, f in zip(axes, FEATURES):
        vals = [all_results[n]["test"][f]["rmse"] for n in names]
        bars = ax.bar(range(len(names)), vals, color=[styles[n] for n in names])
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"{f} RMSE ({UNITS[f]})", fontsize=9)
    fig.tight_layout(); fig.savefig(ROOT / "Figures" / "rmse_comparison.png", dpi=200)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ("CLSTM", "QLSTM"):
        ax.plot(res["results"][name]["train_loss"], label=f"{name} ({res['results'][name]['parameters']} params)")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("train MSE (scaled)"); ax.legend()
    fig.tight_layout(); fig.savefig(ROOT / "Figures" / "training_loss.png", dpi=200)

    VQCLayer(cfg["qubits"], cfg["vqc_layers"]).to_qiskit().draw("mpl", filename=str(ROOT / "Figures" / "vqc_circuit.png"), fold=40)


if __name__ == "__main__":
    main()
