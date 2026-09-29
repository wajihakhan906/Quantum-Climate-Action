"""Differentiable variational quantum circuit (VQC) layer in pure PyTorch.

A batched statevector simulator (exact, no shot noise) so the QLSTM can be trained with
back-propagation. `to_qiskit()` exports the same circuit for running on IBM hardware.

Circuit for n qubits and L layers (as in Chen et al., "Quantum Long Short-Term Memory", 2020):
    encoding : H, RY(arctan(x_i)), RZ(arctan(x_i^2)) on each qubit
    layer    : CNOT ring (i -> i+1, i+r...), then trainable Rot(alpha, beta, gamma) = RZ RY RZ on each qubit
    readout  : <Z_i> for every qubit
"""
import math

import torch
import torch.nn as nn

H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / math.sqrt(2)


def _ry(theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex64)


def _rz(theta):
    e = torch.exp(-0.5j * theta.to(torch.complex64))
    z = torch.zeros_like(e)
    return torch.stack([torch.stack([e, z], -1), torch.stack([z, e.conj()], -1)], -2)


class StateVector:
    """Batched n-qubit state stored as a tensor of shape (batch, 2, 2, ..., 2); qubit 0 is the least significant."""

    def __init__(self, batch, n, device=None):
        self.n = n
        psi = torch.zeros(batch, 2**n, dtype=torch.complex64, device=device)
        psi[:, 0] = 1
        self.psi = psi.reshape(batch, *([2] * n))

    def _axis(self, q):
        return self.n - q  # tensor axis of qubit q (axis 0 is the batch)

    def apply_1q(self, U, q):
        """U: (2, 2) or batched (B, 2, 2)."""
        ax = self._axis(q)
        psi = self.psi.movedim(ax, -1)
        psi = torch.einsum("...j,ij->...i", psi, U) if U.dim() == 2 else \
            torch.einsum("b...j,bij->b...i", psi, U)
        self.psi = psi.movedim(-1, ax)

    def cnot(self, control, target):
        c, t = self._axis(control), self._axis(target)
        psi = self.psi.clone()
        idx1 = [slice(None)] * psi.dim()
        idx1[c] = 1
        sub = psi[tuple(idx1)]
        t_in_sub = t - 1 if t > c else t
        psi[tuple(idx1)] = sub.flip(t_in_sub)
        self.psi = psi

    def expval_z(self):
        probs = self.psi.abs() ** 2
        out = []
        for q in range(self.n):
            p = probs.movedim(self._axis(q), 1).reshape(probs.shape[0], 2, -1).sum(-1)
            out.append(p[:, 0] - p[:, 1])
        return torch.stack(out, -1)


class VQCLayer(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n, self.L = n_qubits, n_layers
        self.weights = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits, 3))

    def forward(self, x):
        """x: (batch, n_qubits) real -> (batch, n_qubits) expectation values in [-1, 1]."""
        sv = StateVector(x.shape[0], self.n, x.device)
        for q in range(self.n):
            sv.apply_1q(H.to(x.device), q)
            sv.apply_1q(_ry(torch.atan(x[:, q])), q)
            sv.apply_1q(_rz(torch.atan(x[:, q] ** 2)), q)
        for layer in range(self.L):
            for r in range(1, self.n if self.n > 2 else 2):
                for q in range(self.n):
                    if self.n > 1 and (q + r) % self.n != q:
                        sv.cnot(q, (q + r) % self.n)
                if self.n <= 2:
                    break
            w = self.weights[layer]
            for q in range(self.n):
                sv.apply_1q(_rz(w[q, 0]), q)
                sv.apply_1q(_ry(w[q, 1]), q)
                sv.apply_1q(_rz(w[q, 2]), q)
        return sv.expval_z()

    def to_qiskit(self, x=None):
        """Same circuit as a Qiskit QuantumCircuit (inputs as Parameters unless `x` is given)."""
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector

        xs = ParameterVector("x", self.n) if x is None else [float(v) for v in x]
        qc = QuantumCircuit(self.n)
        for q in range(self.n):
            qc.h(q)
            if x is None:
                qc.ry(xs[q], q)  # hardware runs pass arctan(x) and arctan(x^2) as parameter values
                qc.rz(xs[q], q)
            else:
                qc.ry(math.atan(xs[q]), q)
                qc.rz(math.atan(xs[q] ** 2), q)
        w = self.weights.detach().cpu().numpy()
        for layer in range(self.L):
            qc.barrier()
            for r in range(1, self.n if self.n > 2 else 2):
                for q in range(self.n):
                    if self.n > 1 and (q + r) % self.n != q:
                        qc.cx(q, (q + r) % self.n)
                if self.n <= 2:
                    break
            for q in range(self.n):
                qc.rz(w[layer, q, 0], q)
                qc.ry(w[layer, q, 1], q)
                qc.rz(w[layer, q, 2], q)
        return qc
