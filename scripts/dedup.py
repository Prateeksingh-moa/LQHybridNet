"""Overlap assessment between HAM10000 (training corpus) and the ISIC 2018 Task 3
test set (editor + Reviewer 2, point 4).

1. Metadata: ISIC Archive lesion identifiers (IL_*) of every HAM10000 image and
   every test image (fetched with fetch_isic_metadata.py) are compared.
2. Image content (catches re-photographed / re-uploaded lesions that metadata
   might miss):
   a. 64-bit DCT perceptual hash, minimum Hamming distance to any HAM image;
   b. ImageNet DINOv2-free baseline: cosine similarity of L2-normalised
      ImageNet-pretrained ResNet-50 global features, nearest HAM neighbour.
   Dermoscopic images share a global layout (central blob on skin), so the
   perceptual hash turned out non-discriminative (visually distinct lesions at
   Hamming 0-4); it is reported as a diagnostic only. A test image is flagged as a
   *candidate* near-duplicate if it shares an archive lesion id with HAM10000 or
   its embedding cosine to a HAM image is >= 0.95 (above the 90th percentile of
   cosine between different images of the same HAM lesion); the flagged pairs are
   saved as a montage and ISIC results are re-computed without them.
Outputs results/dedup/{isic_flags.csv, summary.json, top_pairs.png}.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.fft import dctn


def phash(path):
    im = np.asarray(Image.open(path).convert("L").resize((32, 32), Image.BILINEAR), float)
    d = dctn(im, norm="ortho")[:8, :8].ravel()
    return d[1:] > np.median(d[1:])  # 63 bits


@torch.no_grad()
def embed(paths, dev):
    import timm
    from torchvision import transforms as T
    m = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=0).to(dev).eval()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    out = []
    for i in range(0, len(paths), 128):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + 128]]).to(dev)
        out.append(torch.nn.functional.normalize(m(x).float(), dim=1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sp = pd.read_csv(f"{a.data}/splits.csv")
    te = pd.read_csv(f"{a.data}/isic_test.csv")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. metadata
    ham_les = set(sp.archive_lesion_id.dropna())
    te["lesion_id_overlap"] = te.archive_lesion_id.isin(ham_les)

    # 2a. perceptual hash
    hp = np.stack([phash(f"{a.data}/img/{i}.jpg") for i in sp.image_id])
    tp = np.stack([phash(f"{a.data}/img/{i}.jpg") for i in te.image_id])
    H = torch.tensor(hp, dtype=torch.float32, device=dev)
    Tt = torch.tensor(tp, dtype=torch.float32, device=dev)
    ham_d = (Tt @ (1 - H).T + (1 - Tt) @ H.T)  # Hamming distances (n_test, n_ham)
    dmin, darg = ham_d.min(1)
    te["phash_min_hamming"] = dmin.cpu().numpy().astype(int)
    te["phash_nn"] = sp.image_id.values[darg.cpu().numpy()]

    # 2b. CNN embedding cosine
    eh = embed([f"{a.data}/img/{i}.jpg" for i in sp.image_id], dev)
    et = embed([f"{a.data}/img/{i}.jpg" for i in te.image_id], dev)
    cos = (et.to(dev) @ eh.to(dev).T)
    cmax, carg = cos.max(1)
    te["emb_max_cosine"] = cmax.cpu().numpy()
    te["emb_nn"] = sp.image_id.values[carg.cpu().numpy()]
    # reference: within-HAM similarity of different images of the SAME lesion
    same = sp.groupby("lesion_id").image_id.apply(list)
    idx = {i: k for k, i in enumerate(sp.image_id)}
    pairs = [(idx[l[0]], idx[l[1]]) for l in same if len(l) > 1]
    sims = np.array([(eh[i] @ eh[j]).item() for i, j in pairs])
    hams = np.array([(hp[i] != hp[j]).sum() for i, j in pairs])

    te["flag_phash"] = te.phash_min_hamming <= 8
    te["flag_emb"] = te.emb_max_cosine >= 0.95
    te["any_flag"] = te.lesion_id_overlap | te.flag_emb
    te.to_csv(f"{a.out}/isic_flags.csv", index=False)
    summ = dict(n_test=len(te), n_test_lesions=int(te.archive_lesion_id.nunique()),
                lesion_id_overlap_images=int(te.lesion_id_overlap.sum()),
                phash_le8_diagnostic_only=int(te.flag_phash.sum()), emb_flagged=int(te.flag_emb.sum()),
                any_flagged=int(te.any_flag.sum()),
                test_emb_max_cosine_quantiles={q: float(te.emb_max_cosine.quantile(q)) for q in (0.5, 0.9, 0.99, 1.0)},
                test_phash_min_quantiles={q: float(te.phash_min_hamming.quantile(q)) for q in (0.0, 0.01, 0.1, 0.5)},
                ham_same_lesion_pairs=len(pairs),
                ham_same_lesion_cosine_quantiles={q: float(np.quantile(sims, q)) for q in (0.1, 0.5, 0.9)},
                ham_same_lesion_hamming_quantiles={q: float(np.quantile(hams, q)) for q in (0.1, 0.5, 0.9)},
                thresholds=dict(phash_hamming_le=8, emb_cosine_ge=0.95))
    json.dump(summ, open(f"{a.out}/summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))

    # montage of the 12 most similar test/HAM pairs for visual review
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = te.sort_values("emb_max_cosine", ascending=False).head(12)
    fig, ax = plt.subplots(4, 6, figsize=(12, 8.5))
    for k, r in enumerate(top.itertuples()):
        i, j = divmod(k, 3)
        ax[i, 2 * j].imshow(Image.open(f"{a.data}/img/{r.image_id}.jpg"))
        ax[i, 2 * j].set_title(f"test {r.image_id}\ncos {r.emb_max_cosine:.3f}", fontsize=7)
        ax[i, 2 * j + 1].imshow(Image.open(f"{a.data}/img/{r.emb_nn}.jpg"))
        ax[i, 2 * j + 1].set_title(f"HAM {r.emb_nn}", fontsize=7)
    for x in ax.ravel():
        x.axis("off")
    plt.tight_layout()
    plt.savefig(f"{a.out}/top_pairs.png", dpi=150)


if __name__ == "__main__":
    main()
