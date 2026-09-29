"""Train QSVR, QNNR and a classical SVR baseline on the same windows used by train.py.

    python train_qml.py --qubits 4
Writes Results/results_qml.json and adds predictions to Results/predictions_qml.npz.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from data import datasets
from qml_regressors import QNNR, QSVR, ClassicalSVR
from train import scores

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=7)
    ap.add_argument("--qubits", type=int, default=4)
    ap.add_argument("--qnnr-layers", type=int, default=4)
    ap.add_argument("--qnnr-epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_ds, test_ds, scaler, _, test_df = datasets(args.seq_len)
    Xtr, ytr = (t.numpy() for t in train_ds.tensors)
    Xte, yte = (t.numpy() for t in test_ds.tensors)
    Xtr, Xte = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    y_true = scaler.inverse(yte)

    models = {"QSVR": QSVR(args.qubits), "QNNR": QNNR(args.qubits, args.qnnr_layers, args.qnnr_epochs, seed=args.seed),
              "SVR (classical)": ClassicalSVR(),
              f"SVR ({args.qubits} PCA inputs)": ClassicalSVR(n_components=args.qubits)}
    results, preds = {}, {}
    for name, m in models.items():
        t0 = time.time()
        m.fit(Xtr, ytr)
        p = scaler.inverse(m.predict(Xte))
        preds[name] = p
        results[name] = {"train_seconds": round(time.time() - t0, 1), "test": scores(y_true, p)}
        if name == "QSVR":
            results[name] |= {"bandwidth": m.bandwidth_, "C": m.C_, "validation": m.validation_}
            print(f"  selected bandwidth {m.bandwidth_}, C {m.C_} on the chronological validation split")
        if name.startswith("SVR"):
            results[name] |= {"gamma": m.gamma_, "C": m.C_, "validation": m.validation_}
            print(f"  selected gamma {m.gamma_}, C {m.C_} on the chronological validation split")
        if name == "QNNR":
            results[name]["train_loss"] = m.losses
            results[name]["parameters"] = sum(q.numel() for q in m.model.parameters())
        print(name, f"({results[name]['train_seconds']}s)")
        for f, s in results[name]["test"].items():
            print(f"  {f:13s} RMSE {s['rmse']:.3f}  MAE {s['mae']:.3f}  R2 {s['r2']:.3f}")

    (ROOT / "Results" / "results_qml.json").write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
    np.savez(ROOT / "Results" / "predictions_qml.npz", dates=test_df["date"].astype(str).to_numpy(), y=y_true,
             **{("SVR_pca" if "PCA" in k else k.split(" ")[0]): v for k, v in preds.items()})


if __name__ == "__main__":
    main()
