"""Quantitative Grad-CAM localisation against expert lesion masks
(Reviewer 1, point 5): every HAM10000 lesion-level test image (all have
Tschandl et al. segmentation masks), all 5 fold models of each config.

Per image (CAM for the predicted class at the final feature map, upsampled to
the 256x256 mask):
  pointing_hit   : CAM maximum lies inside the lesion mask (pointing game);
  energy_in_mask : fraction of CAM mass inside the mask (energy pointing game);
  iou_cam50      : IoU between {CAM >= 0.5 max} and the mask.
Per-image scores are averaged over the 5 fold models; LQ vs baseline
differences use the lesion-clustered paired bootstrap.
A seeded random (not hand-picked) sample of images is plotted for illustration.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.environ.get("LQH_RESULTS", f"{ROOT}/results")
sys.path.insert(0, f"{ROOT}/src")
import configs  # noqa: E402
import stats as S  # noqa: E402
from models import HybridNet  # noqa: E402
from train import ImgDS, transforms  # noqa: E402


def load(cfg_name, fold, dev):
    cfg = configs.ALL[cfg_name]
    kw = {k: cfg[k] for k in ["backbone", "gate", "branch", "gate_at", "n_qubits", "n_layers", "fusion_hidden"]}
    m = HybridNet(**kw, pretrained=False)
    m.load_state_dict(torch.load(f"{RES}/runs/{cfg_name}/fold{fold}/ckpt.pt", map_location="cpu"))
    return m.to(dev).eval()


def gradcam(model, x):
    fmap = model.features(x)
    logits = model.head(fmap)
    pred = logits.argmax(1)
    g = torch.autograd.grad(logits.gather(1, pred[:, None]).sum(), fmap)[0]
    cam = F.relu((g.mean((2, 3), keepdim=True) * fmap).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(256, 256), mode="bilinear", align_corners=False)[:, 0]
    cam = cam / cam.flatten(1).amax(1).clamp_min(1e-8)[:, None, None]
    return cam.detach(), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--configs", default="base,lq,qcg_mlp")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = f"{RES}/gradcam"
    os.makedirs(out, exist_ok=True)
    sp = pd.read_csv(f"{a.data}/splits.csv")
    te = sp[sp.split == "test"].reset_index(drop=True)
    masks = torch.tensor(np.stack([np.asarray(Image.open(f"{a.data}/mask/{i}.png")) > 127 for i in te.image_id]))
    per_img = {}
    rows = []
    for cfg in a.configs.split(","):
        if not all(os.path.exists(f"{RES}/runs/{cfg}/fold{k}/ckpt.pt") for k in range(5)):
            print("skip", cfg)
            continue
        scores = []
        for k in range(5):
            m = load(cfg, k, dev)
            _, tf = transforms("standard", m.backbone.pretrained_cfg["mean"], m.backbone.pretrained_cfg["std"])
            dl = torch.utils.data.DataLoader(ImgDS(a.data, te.image_id, te.label, tf), batch_size=64, num_workers=8)
            hit, en, iou, corr, j = [], [], [], [], 0
            for x, y in dl:
                cam, pred = gradcam(m, x.to(dev))
                cam = cam.cpu()
                mk = masks[j:j + len(x)]
                j += len(x)
                flat = cam.flatten(1).argmax(1)
                hit.append(mk.flatten(1).gather(1, flat[:, None])[:, 0].float())
                en.append((cam * mk).flatten(1).sum(1) / cam.flatten(1).sum(1).clamp_min(1e-8))
                b = cam >= 0.5
                iou.append((b & mk).flatten(1).sum(1) / (b | mk).flatten(1).sum(1).clamp_min(1))
                corr.append((pred.cpu() == y).float())
            sc = np.stack([torch.cat(v).numpy() for v in (hit, en, iou, corr)], 1)  # (N, 4)
            scores.append(sc)
            rows.append(dict(config=cfg, fold=k, pointing_game=sc[:, 0].mean(), energy_in_mask=sc[:, 1].mean(),
                             iou_cam50=sc[:, 2].mean(), pointing_game_correct_only=sc[sc[:, 3] == 1, 0].mean()))
            del m
            torch.cuda.empty_cache()
        per_img[cfg] = np.mean(scores, 0)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/gradcam_per_fold.csv", index=False)
    summ = df.groupby("config").agg(["mean", "std"])
    summ.to_csv(f"{out}/gradcam_summary.csv")
    print(summ)
    # LQ vs others: clustered paired bootstrap on per-image fold-averaged scores
    cl = te.lesion_id.values
    _, inv = np.unique(cl, return_inverse=True)
    L = inv.max() + 1
    rng = np.random.default_rng(2026)
    W = rng.multinomial(L, np.full(L, 1 / L), size=5000).astype(np.float32)
    n_img = W @ np.bincount(inv, minlength=L)
    comp = []
    if "lq" in per_img:
        for other in [c for c in per_img if c != "lq"]:
            for mi, name in enumerate(["pointing_game", "energy_in_mask", "iou_cam50"]):
                d = per_img["lq"][:, mi] - per_img[other][:, mi]
                dist = (W @ np.bincount(inv, weights=d, minlength=L)) / n_img
                comp.append(dict(model_A="lq", model_B=other, metric=name, delta=d.mean(),
                                 ci_lo=S.ci(dist)[0], ci_hi=S.ci(dist)[1]))
    pd.DataFrame(comp).to_csv(f"{out}/gradcam_lq_vs_others.csv", index=False)

    # illustration: seeded random sample (not hand-picked), fold-0 models
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [c for c in ["base", "lq"] if c in per_img]
    idx = np.random.default_rng(0).choice(len(te), 6, replace=False)
    fig, ax = plt.subplots(1 + len(show), 6, figsize=(13, 2.3 * (1 + len(show))))
    for r, cfg in enumerate(show):
        m = load(cfg, 0, dev)
        _, tf = transforms("standard", m.backbone.pretrained_cfg["mean"], m.backbone.pretrained_cfg["std"])
        x = torch.stack([tf(Image.open(f"{a.data}/img/{te.image_id[i]}.jpg").convert("RGB")) for i in idx]).to(dev)
        cam, pred = gradcam(m, x)
        for c, i in enumerate(idx):
            if r == 0:
                ax[0, c].imshow(Image.open(f"{a.data}/img/{te.image_id[i]}.jpg"))
                ax[0, c].contour(masks[i].numpy(), levels=[0.5], colors="lime", linewidths=0.8)
                ax[0, c].set_title(f"{te.image_id[i]} ({te.dx[i]})", fontsize=7)
            ax[r + 1, c].imshow(Image.open(f"{a.data}/img/{te.image_id[i]}.jpg"))
            ax[r + 1, c].imshow(cam[c].cpu(), alpha=0.5, cmap="jet")
            ax[r + 1, c].contour(masks[i].numpy(), levels=[0.5], colors="white", linewidths=0.6)
            ax[r + 1, c].set_title(f"{cfg}: pred {['nv','mel','bkl','bcc','akiec','vasc','df'][pred[c]]}", fontsize=7)
    for x_ in ax.ravel():
        x_.axis("off")
    plt.tight_layout()
    plt.savefig(f"{RES}/figures/F3_gradcam_random_sample.png", dpi=170)


if __name__ == "__main__":
    main()
