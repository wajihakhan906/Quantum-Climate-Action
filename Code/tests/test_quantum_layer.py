import numpy as np
import torch
from qiskit.quantum_info import SparsePauliOp, Statevector

from quantum_layer import VQCLayer


def test_matches_qiskit_statevector():
    torch.manual_seed(0)
    layer = VQCLayer(n_qubits=4, n_layers=2)
    x = torch.randn(3, 4)
    ours = layer(x).detach().numpy()
    for b in range(3):
        sv = Statevector(layer.to_qiskit(x[b].numpy()))
        ref = [sv.expectation_value(SparsePauliOp("I" * (3 - q) + "Z" + "I" * q)).real for q in range(4)]
        np.testing.assert_allclose(ours[b], ref, atol=1e-5)


def test_gradients_flow():
    layer = VQCLayer(4, 2)
    x = torch.randn(5, 4, requires_grad=True)
    layer(x).sum().backward()
    assert layer.weights.grad is not None and x.grad is not None
    assert layer.weights.grad.abs().sum() > 0


def test_qsvr_kernel_is_valid_and_matches_qiskit():
    from qiskit.circuit.library import zz_feature_map
    from qiskit.quantum_info import Statevector

    from qml_regressors import fidelity_kernel, zz_statevectors

    X = np.random.default_rng(0).uniform(0, np.pi, (5, 4))
    S = zz_statevectors(X)
    K = fidelity_kernel(S, S)
    np.testing.assert_allclose(np.diag(K), 1, atol=1e-9)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-9
    fm = zz_feature_map(4, reps=2, entanglement="linear")
    a, b = (Statevector(fm.assign_parameters(x)) for x in X[:2])
    assert abs(abs(a.inner(b)) ** 2 - K[0, 1]) < 1e-9


def test_qnnr_learns_simple_target():
    from qml_regressors import QNNR

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6))
    y = np.column_stack([np.tanh(X[:, 0]), np.tanh(X[:, 1])])
    m = QNNR(n_qubits=2, n_layers=2, epochs=40).fit(X, y)
    assert m.losses[-1] < m.losses[0]
