"""Classical LSTM and Quantum LSTM (each gate's affine map replaced by a VQC)."""
import torch
import torch.nn as nn

from quantum_layer import VQCLayer


class CLSTM(nn.Module):
    def __init__(self, n_features=4, hidden=32, n_outputs=4):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_outputs)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class QLSTMCell(nn.Module):
    """LSTM cell where the forget, input, candidate and output gates are VQC_1 ... VQC_4.

    For each gate g:  g = act( W_out · VQC_g( W_in · [h_{t-1}, x_t] ) )
    """

    def __init__(self, n_features, hidden, n_qubits=4, n_layers=2):
        super().__init__()
        self.hidden = hidden
        self.w_in = nn.Linear(n_features + hidden, n_qubits)
        self.vqc = nn.ModuleList(VQCLayer(n_qubits, n_layers) for _ in range(4))
        self.w_out = nn.ModuleList(nn.Linear(n_qubits, hidden) for _ in range(4))

    def forward(self, x, state):
        h, c = state
        v = self.w_in(torch.cat([h, x], dim=1))
        f, i, g, o = (w(q(v)) for q, w in zip(self.vqc, self.w_out))
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class QLSTM(nn.Module):
    def __init__(self, n_features=4, hidden=8, n_outputs=4, n_qubits=4, n_layers=2):
        super().__init__()
        self.cell = QLSTMCell(n_features, hidden, n_qubits, n_layers)
        self.head = nn.Linear(hidden, n_outputs)

    def forward(self, x):
        b = x.shape[0]
        h = x.new_zeros(b, self.cell.hidden)
        c = x.new_zeros(b, self.cell.hidden)
        for t in range(x.shape[1]):
            h, c = self.cell(x[:, t], (h, c))
        return self.head(h)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
