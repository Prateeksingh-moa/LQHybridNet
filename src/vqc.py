"""Variational quantum circuit (VQC) feature map used by LQ-HybridNet.

Circuit (per sample): angle encoding RY(x_i) on every qubit, then L layers of
[trainable RY(theta_{l,i}) on every qubit -> CNOT ring (i -> i+1 mod n)],
then Pauli-Z expectation on every qubit.

Two numerically equivalent implementations are provided:
  * TorchStatevectorVQC - exact batched statevector simulation in PyTorch
    (runs on GPU, used for training and for the "pure GPU simulation" profile);
  * PennyLaneVQC        - the same circuit on PennyLane default.qubit (CPU),
    i.e. the implementation described in the manuscript.
`check_equivalence()` verifies outputs and gradients agree.
"""
import math

import torch
import torch.nn as nn


def _ring_permutation(n):
    """Index permutation of the 2^n amplitude vector for the CNOT ring
    CNOT(0,1), CNOT(1,2), ..., CNOT(n-1,0) applied in that order.
    Qubit 0 is the most significant bit (PennyLane wire ordering)."""
    dim = 2 ** n
    perm = list(range(dim))  # perm[new_index] = old_index
    pairs = [(i, (i + 1) % n) for i in range(n)] if n > 1 else []
    for c, t in pairs:
        cb, tb = n - 1 - c, n - 1 - t
        # applying CNOT maps basis |k> -> |k ^ (bit_t if bit_c)>, i.e. new[k'] = old[k]
        new = [0] * dim
        for k in range(dim):
            k2 = k ^ (1 << tb) if (k >> cb) & 1 else k
            new[k2] = perm[k]
        perm = new
    return torch.tensor(perm, dtype=torch.long)


class TorchStatevectorVQC(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n, self.L = n_qubits, n_layers
        self.theta = nn.Parameter(torch.empty(n_layers, n_qubits).uniform_(0, 2 * math.pi))
        dim = 2 ** n_qubits
        self.register_buffer("perm", _ring_permutation(n_qubits), persistent=False)
        idx = torch.arange(dim)
        zsign = torch.stack([1 - 2 * ((idx >> (n_qubits - 1 - q)) & 1) for q in range(n_qubits)], 1)
        self.register_buffer("zsign", zsign.float(), persistent=False)  # (dim, n)

    def _ry(self, state, angles):
        # state: (B, 2, ..., 2) real; angles: (B, n) or (n,)
        B = state.shape[0]
        if angles.dim() == 1:
            angles = angles.unsqueeze(0).expand(B, -1)
        for q in range(self.n):
            c = torch.cos(angles[:, q] / 2).view(B, *([1] * self.n))
            s = torch.sin(angles[:, q] / 2).view(B, *([1] * self.n))
            a0 = state.narrow(q + 1, 0, 1)
            a1 = state.narrow(q + 1, 1, 1)
            state = torch.cat([c * a0 - s * a1, s * a0 + c * a1], dim=q + 1)
        return state

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            B = x.shape[0]
            state = torch.zeros(B, 2 ** self.n, device=x.device)
            state[:, 0] = 1.0
            state = state.view(B, *([2] * self.n))
            state = self._ry(state, x)  # encoding E(x_i) = RY(x_i)
            for l in range(self.L):
                state = self._ry(state, self.theta[l])
                state = state.reshape(B, -1)[:, self.perm].view(B, *([2] * self.n))
            probs = state.reshape(B, -1) ** 2
            return probs @ self.zsign  # (B, n) Pauli-Z expectations in [-1, 1]


class PennyLaneVQC(nn.Module):
    """Same circuit evaluated by PennyLane default.qubit on the CPU."""

    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        import pennylane as qml

        self.n, self.L = n_qubits, n_layers
        self.theta = nn.Parameter(torch.empty(n_layers, n_qubits).uniform_(0, 2 * math.pi))
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[:, i], wires=i)
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[l, i], wires=i)
                if n_qubits > 1:
                    for i in range(n_qubits):
                        qml.CNOT(wires=[i, (i + 1) % n_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x):
        dev = x.device
        out = self.circuit(x.cpu().double(), self.theta.cpu().double())
        out = torch.stack(list(out), dim=-1) if isinstance(out, (list, tuple)) else out
        return out.to(dev, torch.float32)


def check_equivalence(n=4, L=2, B=8, seed=0):
    torch.manual_seed(seed)
    a, b = TorchStatevectorVQC(n, L), PennyLaneVQC(n, L)
    b.theta.data.copy_(a.theta.data)
    x = torch.randn(B, n) * 2
    x1, x2 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    ya, yb = a(x1), b(x2)
    ya.sum().backward()
    yb.sum().backward()
    return {
        "max_abs_out_diff": (ya - yb).abs().max().item(),
        "max_abs_grad_x_diff": (x1.grad - x2.grad).abs().max().item(),
        "max_abs_grad_theta_diff": (a.theta.grad - b.theta.grad).abs().max().item(),
    }


if __name__ == "__main__":
    for n in (2, 3, 4, 5, 6):
        print(n, check_equivalence(n=n))
