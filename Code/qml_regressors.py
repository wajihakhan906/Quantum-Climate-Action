"""Quantum regressors on flattened climate windows: QSVR (quantum kernel) and QNNR (variational QNN).

Both map a window of `seq_len` days x 4 variables to `n_qubits` inputs with PCA (fitted on training data),
scaled to [0, pi] for angle encoding, and predict all 4 variables for the next day.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

from quantum_layer import VQCLayer


def reducer(n_qubits):
    return Pipeline([("std", StandardScaler()), ("pca", PCA(n_components=n_qubits)),
                     ("angle", MinMaxScaler(feature_range=(0, np.pi)))])


# --------------------------------------------------------------------------- #
# QSVR: SVR with the fidelity kernel k(x, x') = |<phi(x)|phi(x')>|^2 of a ZZ feature map
# --------------------------------------------------------------------------- #
def zz_statevectors(X, reps=2):
    """Statevectors of the ZZ feature map (linear entanglement) for each row of X."""
    from qiskit.circuit.library import zz_feature_map
    from qiskit.quantum_info import Statevector

    fm = zz_feature_map(X.shape[1], reps=reps, entanglement="linear")
    return np.stack([Statevector(fm.assign_parameters(x)).data for x in X])


def fidelity_kernel(A, B):
    return np.abs(A.conj() @ B.T) ** 2


class QSVR:
    """Multi-output SVR on a precomputed ZZ fidelity kernel.

    `bandwidth` scales the encoded angles before the feature map (x -> bandwidth * x). Large angles make
    fidelity kernels vanishingly small off the diagonal (every point looks unlike every other), so the
    bandwidth matters a lot. With bandwidth="auto", bandwidth and C are chosen on a chronological
    validation split (the last `val_fraction` of the training data), never on test data.
    """

    GRID_BANDWIDTH = (0.1, 0.2, 0.35, 0.5, 1.0)
    GRID_C = (1.0, 10.0)

    def __init__(self, n_qubits=4, C=1.0, epsilon=0.01, reps=1, bandwidth="auto", val_fraction=0.2):
        self.n_qubits, self.C, self.epsilon, self.reps = n_qubits, C, epsilon, reps
        self.bandwidth, self.val_fraction = bandwidth, val_fraction

    def _states(self, enc, X, bw):
        return zz_statevectors(enc.transform(X) * bw, self.reps)

    def _svr(self, C):
        return MultiOutputRegressor(SVR(kernel="precomputed", C=C, epsilon=self.epsilon))

    def _select(self, X, y):
        cut = int(len(X) * (1 - self.val_fraction))
        enc = reducer(self.n_qubits).fit(X[:cut])
        self.validation_ = []
        for bw in self.GRID_BANDWIDTH:
            A, B = self._states(enc, X[:cut], bw), self._states(enc, X[cut:], bw)
            K_aa, K_ba = fidelity_kernel(A, A), fidelity_kernel(B, A)
            for C in self.GRID_C:
                p = self._svr(C).fit(K_aa, y[:cut]).predict(K_ba)
                r2 = 1 - ((p - y[cut:]) ** 2).sum(0) / ((y[cut:] - y[cut:].mean(0)) ** 2).sum(0)
                self.validation_.append({"bandwidth": bw, "C": C, "val_r2_mean": float(r2.mean())})
        best = max(self.validation_, key=lambda r: r["val_r2_mean"])
        return best["bandwidth"], best["C"]

    def fit(self, X, y):
        if self.bandwidth == "auto":
            self.bandwidth_, self.C_ = self._select(X, y)
        else:
            self.bandwidth_, self.C_ = self.bandwidth, self.C
        self.enc = reducer(self.n_qubits).fit(X)
        self.train_states_ = self._states(self.enc, X, self.bandwidth_)
        self.svr = self._svr(self.C_).fit(fidelity_kernel(self.train_states_, self.train_states_), y)
        return self

    def predict(self, X):
        return self.svr.predict(fidelity_kernel(self._states(self.enc, X, self.bandwidth_), self.train_states_))


# --------------------------------------------------------------------------- #
# QNNR: angle-encoded VQC with <Z_i> readout and a linear output layer
# --------------------------------------------------------------------------- #
class QNNRModel(nn.Module):
    def __init__(self, n_qubits=4, n_layers=4, n_outputs=4):
        super().__init__()
        self.vqc = VQCLayer(n_qubits, n_layers)
        self.head = nn.Linear(n_qubits, n_outputs)

    def forward(self, x):
        return self.head(self.vqc(x))


class QNNR:
    def __init__(self, n_qubits=4, n_layers=4, epochs=150, lr=0.02, seed=42):
        self.enc = reducer(n_qubits)
        self.n_qubits, self.n_layers, self.epochs, self.lr, self.seed = n_qubits, n_layers, epochs, lr, seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        x = torch.tensor(self.enc.fit_transform(X), dtype=torch.float32)
        t = torch.tensor(y, dtype=torch.float32)
        self.model = QNNRModel(self.n_qubits, self.n_layers, y.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.losses = []
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            total = 0.0
            for idx in torch.randperm(len(x), generator=g).split(64):
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.model(x[idx]), t[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            self.losses.append(total / len(x))
        return self

    @torch.no_grad()
    def predict(self, X):
        return self.model(torch.tensor(self.enc.transform(X), dtype=torch.float32)).numpy()


class ClassicalSVR:
    """RBF SVR on the full standardised window, with gamma and C tuned on the same chronological
    validation split as QSVR so the comparison is fair."""

    GRID_GAMMA = ("scale", 0.003, 0.01, 0.03, 0.1, 0.3)
    GRID_C = (1.0, 10.0)

    def __init__(self, epsilon=0.01, val_fraction=0.2, n_components=None):
        self.epsilon, self.val_fraction, self.n_components = epsilon, val_fraction, n_components

    def _model(self, gamma, C):
        """n_components=None: full window; an int: the same PCA inputs the quantum models receive."""
        pre = [("std", StandardScaler())]
        if self.n_components:
            pre += [("pca", PCA(n_components=self.n_components)), ("std2", StandardScaler())]
        return Pipeline(pre + [("svr", MultiOutputRegressor(SVR(gamma=gamma, C=C, epsilon=self.epsilon)))])

    def fit(self, X, y):
        cut = int(len(X) * (1 - self.val_fraction))
        self.validation_ = []
        for g in self.GRID_GAMMA:
            for C in self.GRID_C:
                p = self._model(g, C).fit(X[:cut], y[:cut]).predict(X[cut:])
                r2 = 1 - ((p - y[cut:]) ** 2).sum(0) / ((y[cut:] - y[cut:].mean(0)) ** 2).sum(0)
                self.validation_.append({"gamma": g, "C": C, "val_r2_mean": float(r2.mean())})
        best = max(self.validation_, key=lambda r: r["val_r2_mean"])
        self.gamma_, self.C_ = best["gamma"], best["C"]
        self.model = self._model(self.gamma_, self.C_).fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
