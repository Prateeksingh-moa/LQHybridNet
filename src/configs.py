"""Experiment registry. Every config is trained on the same 5 lesion-level folds
with the same recipe; only the fields listed here differ.

`tuned`: config gets its own LR search (equal budget: same grid, fold-0
validation loss). Configs with `inherit` reuse the selected LR of that config.
"""

D = dict(backbone="mobilenetv2", gate="none", branch="none", gate_at=None, n_qubits=4, n_layers=2,
         fusion_hidden=64, aug="standard", train_frac=1.0)


def C(**kw):
    c = dict(D)
    c.update(kw)
    return c


CONFIGS = {
    # ---- MobileNetV2 family: proposed model, its components, matched controls
    "base": C(),                                       # no QCG, no VQC
    "qcg": C(gate="qcg"),                              # QCG only
    "vqc": C(branch="vqc"),                            # VQC only
    "lq": C(gate="qcg", branch="vqc"),                 # full LQ-HybridNet
    # gating swapped at the same 2 locations, VQC branch kept
    "se_vqc": C(gate="se", branch="vqc"),
    "eca_vqc": C(gate="eca", branch="vqc"),
    "cbam_vqc": C(gate="cbam", branch="vqc"),
    # branch swapped, QCG kept
    "qcg_mlp": C(gate="qcg", branch="mlp"),            # parameter-matched MLP (152 vs 140 params)
    "qcg_mlp_paper": C(gate="qcg", branch="mlp_paper"),  # manuscript's Dense 32->16->4
    "qcg_resmlp": C(gate="qcg", branch="resmlp"),
    "qcg_fourier": C(gate="qcg", branch="fourier"),
    "qcg_poly": C(gate="qcg", branch="poly"),
    "qcg_trig": C(gate="qcg", branch="trig"),
    # fully classical attention baselines (no proposed component)
    "se": C(gate="se"),
    "eca": C(gate="eca"),
    "cbam": C(gate="cbam"),
    "eca_trig": C(gate="eca", branch="trig"),
    # ---- strong lightweight baselines (Reviewer 1, point 4), strong augmentation
    "effb0_strong": C(backbone="efficientnet_b0", aug="strong"),
    "effv2b0_strong": C(backbone="efficientnetv2_b0", aug="strong"),
    "mbv3_strong": C(backbone="mobilenetv3_large", aug="strong"),
    "mobilevitv2_strong": C(backbone="mobilevitv2", aug="strong"),
    "base_strong": C(aug="strong"),
    "lq_strong": C(gate="qcg", branch="vqc", aug="strong"),
    # ---- backbone generality (classical C vs hybrid H)
    **{f"{b}_C": C(backbone=b) for b in ["resnet50", "efficientnet_b0", "vit_base", "swin_t"]},
    **{f"{b}_H": C(backbone=b, gate="qcg", branch="vqc") for b in ["resnet50", "efficientnet_b0", "vit_base", "swin_t"]},
}
# Only Reviewer 1 point 1: LQ vs the Table 14 controls (+ residual-MLP,
# Fourier, polynomial), 5 folds each. Backbone C/H pairs already exist.
REQUIRED = ["lq", "qcg", "vqc", "qcg_mlp", "qcg_trig", "eca_vqc", "qcg_resmlp", "qcg_fourier", "qcg_poly"]
CONFIGS = {k: CONFIGS[k] for k in REQUIRED}
TUNED = list(CONFIGS)
# one fixed learning rate per backbone family, identical for every config on it
FIXED_LR = {"vit_base": 1e-4, "swin_t": 1e-4}  # transformers; all CNNs use 3e-4


def lr_for(cfg):
    return FIXED_LR.get(CONFIGS[cfg]["backbone"], 3e-4)

ABLATIONS, LEARNING, LC_MODELS, LC_FRACS = {}, {}, [], ()

ALL = {**CONFIGS, **ABLATIONS, **LEARNING}

# checkpoints kept only where later analyses need weights (Grad-CAM, latency)
SAVE_CKPT = set()

LR_GRID = (1e-4, 3e-4, 1e-3)
BATCH = {"vit_base": 32, "swin_t": 32}  # 12 GB GPUs; others use 64
