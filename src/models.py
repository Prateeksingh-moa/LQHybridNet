"""LQ-HybridNet and all matched controls.

Every model = timm backbone (+ optional channel-gating modules inserted at fixed
stage outputs) -> global pooled feature -> shared projection head
(Dropout -> Linear(F, 32) -> ReLU -> BatchNorm1d) -> c (32-d)
-> optional feature-map branch b(c) -> fused [c ; b(c)] -> fusion MLP -> 7 logits.

So a control differs from LQ-HybridNet *only* in the gating module and/or the
branch; insertion locations, projection head, fusion MLP and training recipe are
identical. `describe(cfg)` states which proposed components each config keeps.
"""
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from vqc import TorchStatevectorVQC, PennyLaneVQC

NUM_CLASSES = 7

# timm model name, attribute path of stage containers, default gate insertion
# indices (manuscript Sec. 2.3.3) and feature layout of those stage outputs.
BACKBONES = {
    "mobilenetv2": dict(timm="mobilenetv2_100.ra_in1k", stages="blocks", gate_at=(4, 6), layout="nchw"),
    "resnet50": dict(timm="resnet50.a1_in1k", stages=None, gate_at=("layer2", "layer3"), layout="nchw"),
    "efficientnet_b0": dict(timm="efficientnet_b0.ra_in1k", stages="blocks", gate_at=(3, 5), layout="nchw"),
    "vit_base": dict(timm="vit_base_patch16_224.augreg2_in21k_ft_in1k", stages="blocks", gate_at=(5, 10), layout="nlc"),
    "swin_t": dict(timm="swin_tiny_patch4_window7_224.ms_in1k", stages="layers", gate_at=(1, 2), layout="nhwc"),
    # strong lightweight baselines requested by Reviewer 1 (no gate/branch)
    "mobilenetv3_large": dict(timm="mobilenetv3_large_100.ra_in1k", stages="blocks", gate_at=(), layout="nchw"),
    "mobilevitv2": dict(timm="mobilevitv2_100.cvnets_in1k", stages="stages", gate_at=(), layout="nchw"),
    "efficientnetv2_b0": dict(timm="tf_efficientnetv2_b0.in1k", stages="blocks", gate_at=(), layout="nchw"),
}


# ----------------------------------------------------------------------------
# Channel gating modules. All produce a per-channel scale s in (0, gate_scale)
# and return x * s, so they are interchangeable at the same insertion points.
# gate_scale = 2 makes every gate start near identity (sigma(0) * 2 = 1), which
# protects the pretrained features; it is applied identically to every gate.
# ----------------------------------------------------------------------------
def _pool(x, layout):
    if layout == "nchw":
        return x.mean((2, 3))
    if layout == "nhwc":
        return x.mean((1, 2))
    return x.mean(1)  # nlc tokens


def _apply(x, s, layout):
    if layout == "nchw":
        return x * s[:, :, None, None]
    if layout == "nhwc":
        return x * s[:, None, None, :]
    return x * s[:, None, :]


class QCG(nn.Module):
    """Quantum-Inspired Channel Gating, manuscript Eqs. (1)-(5)."""

    def __init__(self, C, layout, reduction=8, gate_scale=2.0):
        super().__init__()
        r = max(C // reduction, 4)
        self.layout, self.gate_scale = layout, gate_scale
        self.amp = nn.Linear(C, r)
        self.phase = nn.Linear(C, r)
        self.psi = nn.Parameter(torch.zeros(r))
        self.out = nn.Linear(r, C)

    def forward(self, x):
        z = _pool(x, self.layout).float()
        a = torch.sigmoid(self.amp(z))
        phi = math.pi * torch.tanh(self.phase(z))
        q = a * torch.cos(phi + self.psi)
        s = self.gate_scale * torch.sigmoid(self.out(q))
        return _apply(x, s.to(x.dtype), self.layout)


class SE(nn.Module):
    def __init__(self, C, layout, reduction=8, gate_scale=2.0):
        super().__init__()
        r = max(C // reduction, 4)
        self.layout, self.gate_scale = layout, gate_scale
        # hidden width 1.5r so params ~= QCG (QCG has two C->r maps + one r->C)
        h = int(1.5 * r)
        self.fc1 = nn.Linear(C, h)
        self.fc2 = nn.Linear(h, C)

    def forward(self, x):
        z = _pool(x, self.layout).float()
        s = self.gate_scale * torch.sigmoid(self.fc2(F.relu(self.fc1(z))))
        return _apply(x, s.to(x.dtype), self.layout)


class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al., CVPR 2020)."""

    def __init__(self, C, layout, gate_scale=2.0, **_):
        super().__init__()
        k = int(abs((math.log2(C) + 1) / 2))
        k = k if k % 2 else k + 1
        self.layout, self.gate_scale = layout, gate_scale
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        z = _pool(x, self.layout).float()
        s = self.gate_scale * torch.sigmoid(self.conv(z.unsqueeze(1)).squeeze(1))
        return _apply(x, s.to(x.dtype), self.layout)


class CBAM(nn.Module):
    """CBAM (Woo et al., ECCV 2018): channel (avg+max MLP) then 7x7 spatial gate."""

    def __init__(self, C, layout, reduction=8, gate_scale=2.0):
        super().__init__()
        assert layout == "nchw", "CBAM is only used on CNN feature maps"
        h = int(1.5 * max(C // reduction, 4))  # params ~= QCG, as for SE
        self.gate_scale = gate_scale
        self.mlp = nn.Sequential(nn.Linear(C, h), nn.ReLU(), nn.Linear(h, C))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        xf = x.float()
        s = self.mlp(xf.mean((2, 3))) + self.mlp(xf.amax((2, 3)))
        xf = xf * (self.gate_scale * torch.sigmoid(s))[:, :, None, None]
        m = torch.cat([xf.mean(1, keepdim=True), xf.amax(1, keepdim=True)], 1)
        xf = xf * (self.gate_scale * torch.sigmoid(self.spatial(m)))
        return xf.to(x.dtype)


GATES = {"qcg": QCG, "se": SE, "eca": ECA, "cbam": CBAM}


# ----------------------------------------------------------------------------
# Feature-map branches b(c): input c (B, 32). All except "trig" output n_out
# features that are concatenated with c. Each branch starts with the same
# angle projection a = pi * tanh(W c + b) where applicable.
# ----------------------------------------------------------------------------
class AngleProj(nn.Module):
    def __init__(self, d_in, n):
        super().__init__()
        self.lin = nn.Linear(d_in, n)

    def forward(self, c):
        return math.pi * torch.tanh(self.lin(c))


class VQCBranch(nn.Module):
    def __init__(self, d_in=32, n_qubits=4, n_layers=2, impl="torch"):
        super().__init__()
        self.proj = AngleProj(d_in, n_qubits)
        self.vqc = (TorchStatevectorVQC if impl == "torch" else PennyLaneVQC)(n_qubits, n_layers)
        self.out_dim = n_qubits

    def forward(self, c):
        return self.vqc(self.proj(c))


class TrigBranch(nn.Module):
    """Manuscript's trigonometric control: angles -> [sin, cos]."""

    def __init__(self, d_in=32, n=4):
        super().__init__()
        self.proj = AngleProj(d_in, n)
        self.out_dim = 2 * n

    def forward(self, c):
        a = self.proj(c)
        return torch.cat([torch.sin(a), torch.cos(a)], 1)


class FourierBranch(nn.Module):
    """Learned Fourier features: cos(W2 a + b2) on top of the angle projection
    (a trainable periodic map of the same size as the VQC)."""

    def __init__(self, d_in=32, n=4):
        super().__init__()
        self.proj = AngleProj(d_in, n)
        self.freq = nn.Linear(n, n)
        self.out_dim = n

    def forward(self, c):
        return torch.cos(self.freq(self.proj(c)))


class PolyBranch(nn.Module):
    """Degree-2 polynomial expansion of a 4-d projection, mixed back to 4-d."""

    def __init__(self, d_in=32, n=4):
        super().__init__()
        self.proj = nn.Linear(d_in, n)
        iu = torch.triu_indices(n, n)
        self.register_buffer("iu", iu, persistent=False)
        self.mix = nn.Linear(n + iu.shape[1], n)
        self.out_dim = n

    def forward(self, c):
        a = torch.tanh(self.proj(c))
        quad = (a[:, :, None] * a[:, None, :])[:, self.iu[0], self.iu[1]]
        return torch.tanh(self.mix(torch.cat([a, quad], 1)))


class MLPBranch(nn.Module):
    def __init__(self, d_in=32, hidden=16, n=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, n), nn.Tanh())
        self.out_dim = n

    def forward(self, c):
        return self.net(c)


class ResMLPBranch(nn.Module):
    def __init__(self, d_in=32, hidden=4, n=4):
        super().__init__()
        self.skip = nn.Linear(d_in, n)
        self.f = nn.Sequential(nn.Linear(n, hidden), nn.ReLU(), nn.Linear(hidden, n))
        self.out_dim = n

    def forward(self, c):
        a = self.skip(c)
        return torch.tanh(a + self.f(a))


def make_branch(name, n_qubits=4, n_layers=2, vqc_impl="torch"):
    if name in (None, "none"):
        return None
    if name == "vqc":
        return VQCBranch(32, n_qubits, n_layers, vqc_impl)
    if name == "trig":
        return TrigBranch(32, n_qubits)
    if name == "fourier":
        return FourierBranch(32, n_qubits)
    if name == "poly":
        return PolyBranch(32, n_qubits)
    if name == "mlp":  # parameter-matched to the VQC branch (~150 params)
        return MLPBranch(32, 4, n_qubits)
    if name == "mlp_paper":  # manuscript's Dense 32 -> 16 -> 4
        return MLPBranch(32, 16, n_qubits)
    if name == "resmlp":
        return ResMLPBranch(32, 4, n_qubits)
    raise ValueError(name)


# ----------------------------------------------------------------------------
class GatedStage(nn.Module):
    def __init__(self, stage, gate):
        super().__init__()
        self.stage, self.gate = stage, gate

    def forward(self, x):
        return self.gate(self.stage(x))


def _stage_channels(model, name, idx, layout):
    """Infer output channels of a stage with a dry run."""
    feats = {}

    def hook(_, __, out):
        feats["c"] = out.shape[1] if layout == "nchw" else out.shape[-1]

    h = _get_stage(model, name, idx).register_forward_hook(hook)
    with torch.no_grad():
        model.eval()
        model.forward_features(torch.zeros(1, 3, 224, 224))
    h.remove()
    return feats["c"]


def _get_stage(model, stages, idx):
    return getattr(model, idx) if stages is None else getattr(model, stages)[idx]


def _set_stage(model, stages, idx, mod):
    if stages is None:
        setattr(model, idx, mod)
    else:
        getattr(model, stages)[idx] = mod


class HybridNet(nn.Module):
    def __init__(self, backbone="mobilenetv2", gate="none", branch="none", gate_at=None,
                 n_qubits=4, n_layers=2, fusion_hidden=64, gate_reduction=8, gate_scale=2.0,
                 vqc_impl="torch", pretrained=True, drop=0.2):
        super().__init__()
        spec = BACKBONES[backbone]
        self.backbone = timm.create_model(spec["timm"], pretrained=pretrained, num_classes=0)
        gate_at = spec["gate_at"] if gate_at is None else gate_at
        self.gate_at = list(gate_at) if gate != "none" else []
        for idx in self.gate_at:
            C = _stage_channels(self.backbone, spec["stages"], idx, spec["layout"])
            g = GATES[gate](C, spec["layout"], reduction=gate_reduction, gate_scale=gate_scale)
            _set_stage(self.backbone, spec["stages"], idx, GatedStage(_get_stage(self.backbone, spec["stages"], idx), g))
        nf = getattr(self.backbone, "head_hidden_size", None) or self.backbone.num_features
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(nf, 32), nn.ReLU(), nn.BatchNorm1d(32))
        self.branch = make_branch(branch, n_qubits, n_layers, vqc_impl)
        d = 32 + (self.branch.out_dim if self.branch is not None else 0)
        self.fusion = nn.Sequential(nn.Linear(d, fusion_hidden), nn.ReLU(), nn.Dropout(drop),
                                    nn.Linear(fusion_hidden, NUM_CLASSES))

    def new_parameters(self):
        """Parameters not from the pretrained backbone (get the head LR)."""
        ids = {id(p) for p in self.backbone.parameters()}
        gate_ids = {id(p) for m in self.backbone.modules() if isinstance(m, GatedStage)
                    for p in m.gate.parameters()}
        return [p for p in self.parameters() if id(p) not in ids or id(p) in gate_ids]

    def features(self, x):
        return self.backbone.forward_features(x)

    def head(self, fmap):
        c = self.proj(self.backbone.forward_head(fmap, pre_logits=True))
        if self.branch is not None:
            c = torch.cat([c, self.branch(c).to(c.dtype)], 1)
        return self.fusion(c)

    def forward(self, x):
        return self.head(self.features(x))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def describe(cfg):
    g, b = cfg.get("gate", "none"), cfg.get("branch", "none")
    return f"gate={g} ({'proposed QCG kept' if g == 'qcg' else 'QCG removed'}); " \
           f"branch={b} ({'proposed VQC kept' if b == 'vqc' else 'VQC removed'}); " \
           f"shared 32-d projection + fusion MLP(hidden={cfg.get('fusion_hidden', 64)})"
