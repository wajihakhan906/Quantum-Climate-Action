"""Train CLSTM and QLSTM on the Delhi climate series and evaluate on the 114-day test period.

    python train.py                       # both models, default settings
    python train.py --epochs 100 --qubits 4 --vqc-layers 2
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import FEATURES, datasets
from models import CLSTM, QLSTM, n_params

ROOT = Path(__file__).resolve().parent.parent


def fit(model, train_ds, epochs, lr, batch_size, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    losses = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in dl:
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        losses.append(total / len(train_ds))
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  epoch {epoch + 1:3d}  train MSE {losses[-1]:.5f}")
    return losses


@torch.no_grad()
def predict(model, ds):
    model.eval()
    x, y = ds.tensors
    return model(x).numpy(), y.numpy()


def scores(y, p):
    out = {}
    for j, f in enumerate(FEATURES):
        err = p[:, j] - y[:, j]
        ss_tot = ((y[:, j] - y[:, j].mean()) ** 2).sum()
        out[f] = {"rmse": float(np.sqrt((err**2).mean())), "mae": float(np.abs(err).mean()),
                  "r2": float(1 - (err**2).sum() / ss_tot)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--qubits", type=int, default=4)
    ap.add_argument("--vqc-layers", type=int, default=2)
    ap.add_argument("--qlstm-hidden", type=int, default=8)
    ap.add_argument("--clstm-hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_ds, test_ds, scaler, train_df, test_df = datasets(args.seq_len)
    torch.manual_seed(args.seed)
    models = {"CLSTM": CLSTM(hidden=args.clstm_hidden),
              "QLSTM": QLSTM(hidden=args.qlstm_hidden, n_qubits=args.qubits, n_layers=args.vqc_layers)}

    results, preds = {}, {}
    for name, model in models.items():
        print(f"{name}: {n_params(model)} trainable parameters")
        t0 = time.time()
        losses = fit(model, train_ds, args.epochs, 5e-3 if name == "QLSTM" else 1e-3, args.batch_size, args.seed)
        p, y = predict(model, test_ds)
        p, y = scaler.inverse(p), scaler.inverse(y)
        preds[name] = p
        results[name] = {"parameters": n_params(model), "train_seconds": round(time.time() - t0, 1),
                         "train_loss": losses, "test": scores(y, p)}
        for f, s in results[name]["test"].items():
            print(f"  {f:13s} RMSE {s['rmse']:.3f}  MAE {s['mae']:.3f}  R2 {s['r2']:.3f}")
        if name == "QLSTM":
            torch.save(model.state_dict(), ROOT / "Results" / "qlstm.pt")

    (ROOT / "Results").mkdir(exist_ok=True)
    (ROOT / "Results" / "results.json").write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
    np.savez(ROOT / "Results" / "predictions.npz", dates=test_df["date"].astype(str).to_numpy(), y=y, **preds)


if __name__ == "__main__":
    main()
