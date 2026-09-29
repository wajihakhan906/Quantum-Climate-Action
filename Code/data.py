"""Daily Delhi Climate dataset (2013-01-01 to 2017-04-24): meantemp, humidity, wind_speed, meanpressure."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

DATA = Path(__file__).resolve().parent.parent / "Dataset"
FEATURES = ["meantemp", "humidity", "wind_speed", "meanpressure"]


def load(split="Train"):
    df = pd.read_csv(DATA / f"DailyDelhiClimate{split}.csv", parse_dates=["date"])
    # meanpressure has sensor glitches (e.g. 59 hPa, 7679 hPa); treat implausible values as missing
    bad = (df["meanpressure"] < 950) | (df["meanpressure"] > 1050)
    df.loc[bad, "meanpressure"] = np.nan
    df[FEATURES] = df[FEATURES].interpolate(limit_direction="both")
    return df


class MinMax:
    def fit(self, x):
        self.lo, self.hi = x.min(0), x.max(0)
        return self

    def transform(self, x):
        return (x - self.lo) / (self.hi - self.lo)

    def inverse(self, x):
        return x * (self.hi - self.lo) + self.lo


def windows(values, seq_len):
    X = np.stack([values[i:i + seq_len] for i in range(len(values) - seq_len)])
    y = values[seq_len:]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def datasets(seq_len=7):
    """Train on the training file; test on the 114 days that follow it. The last `seq_len` training days are
    prepended to the test series so that every test day gets a prediction."""
    train, test = load("Train"), load("Test")
    train = train[train["date"] < test["date"].min()]  # the Kaggle train file overlaps the test start by 1 day
    scaler = MinMax().fit(train[FEATURES].to_numpy())
    tr = scaler.transform(train[FEATURES].to_numpy())
    te = scaler.transform(np.concatenate([train[FEATURES].to_numpy()[-seq_len:], test[FEATURES].to_numpy()]))
    return TensorDataset(*windows(tr, seq_len)), TensorDataset(*windows(te, seq_len)), scaler, train, test
