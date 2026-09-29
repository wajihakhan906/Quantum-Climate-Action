# Quantum Climate Action

Quantum and hybrid quantum–classical models for **climate time-series forecasting**, benchmarked against classical
models on the **Daily Delhi Climate** dataset (2013–2017): mean temperature, humidity, wind speed and mean pressure.

## Models
| Model | Type | Input | Idea |
|---|---|---|---|
| **CLSTM** | classical | 7-day window | standard LSTM, 32 hidden units |
| **QLSTM** | hybrid | 7-day window | LSTM cell whose forget, input, candidate and output gates are **VQC₁–VQC₄** |
| **QNNR** | quantum NN regressor | window → 4 PCA angles | angle-encoded variational circuit, ⟨Z<sub>i</sub>⟩ readout + linear layer |
| **QSVR** | quantum kernel | window → 4 PCA angles | SVR with the ZZ-feature-map fidelity kernel \|⟨φ(x)\|φ(x′)⟩\|² |
| SVR | classical baseline | full window, and the same 4 PCA inputs | RBF SVR |

All models predict all four variables one day ahead and are tested on the same **114 days** (Jan–Apr 2017).
Hyper-parameters (QSVR kernel bandwidth, SVR γ and C) are chosen on the last 20 % of the *training* period,
never on test data.

### QLSTM gate
Each gate g ∈ {f, i, C̃, o} computes `act(W_out · VQC_g(W_in · [h_{t−1}, x_t]))`, where VQC_g encodes 4 inputs with
H, RY(arctan x), RZ(arctan x²), applies 2 layers of CNOT entanglers and trainable RZ-RY-RZ rotations, and measures ⟨Z⟩
on every qubit (Chen et al., *Quantum Long Short-Term Memory*, 2020).

The VQC is simulated exactly in PyTorch (`quantum_layer.py`), so QLSTM trains with ordinary back-propagation. The
simulator is tested against Qiskit's statevector, and `VQCLayer.to_qiskit()` exports the same circuit for IBM hardware.

## Repository Structure
```
Quantum-Climate-Action/
├── Code/
│   ├── data.py              # loading, pressure-glitch cleaning, scaling, 7-day windows
│   ├── quantum_layer.py     # differentiable VQC (PyTorch statevector) + Qiskit export
│   ├── models.py            # CLSTM, QLSTM (VQC gates)
│   ├── qml_regressors.py    # QSVR, QNNR, classical SVR baselines
│   ├── train.py             # CLSTM vs QLSTM
│   ├── train_qml.py         # QSVR, QNNR, SVR
│   ├── plot_results.py      # all figures
│   ├── tests/               # simulator vs Qiskit, kernel validity, QNNR learning
│   └── requirements.txt
├── Dataset/                 # DailyDelhiClimateTrain.csv, DailyDelhiClimateTest.csv
├── Figures/                 # dataset, predictions, RMSE comparison, VQC circuit
├── Results/                 # metrics JSON, predictions
├── LICENSE
└── README.md
```

## Quick Start
```bash
cd Code
pip install -r requirements.txt
python -m pytest tests
python train.py          # CLSTM vs QLSTM
python train_qml.py      # QSVR, QNNR, SVR
python plot_results.py
```

## Results
See [Results/README.md](Results/README.md).

## Author
**Wajiha Rahim Khan**  
[Google Scholar](https://scholar.google.com/citations?user=ctvOkbYAAAAJ) · [Email](mailto:wajihakhan906@gmail.com)

## License
MIT. See [LICENSE](LICENSE).
