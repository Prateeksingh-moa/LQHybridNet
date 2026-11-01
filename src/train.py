"""Train one (config, fold) and save predictions for val / HAM test / ISIC test.

Recipe (identical for every config): AdamW (wd 1e-2), backbone LR = lr, new
modules (gates, projection, branch, fusion) LR = 10 x lr, 1 warm-up epoch then
cosine decay, fp16 autocast (the VQC runs in fp32), batch 64 (32 for ViT/Swin),
class-balanced oversampling of the training folds only (sample weight
n_c^-0.5), up to 30 epochs, early stopping (patience 8) and checkpoint
selection on validation-fold cross-entropy. Test sets are evaluated once, after
selection.
"""
import argparse
import copy
import json
import math
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(__file__))
import configs  # noqa: E402
from models import BACKBONES, HybridNet, count_params, describe  # noqa: E402


class ImgDS(Dataset):
    def __init__(self, root, ids, labels, tf):
        self.root, self.ids, self.labels, self.tf = root, list(ids), np.asarray(labels), tf

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        im = Image.open(f"{self.root}/img/{self.ids[i]}.jpg").convert("RGB")
        return self.tf(im), int(self.labels[i])


def transforms(aug, mean, std):
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    train = [T.RandomResizedCrop(224, scale=(0.8, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
             T.RandomRotation(20), T.ColorJitter(0.2, 0.2, 0.2)]
    if aug == "strong":
        train += [T.RandAugment(num_ops=2, magnitude=9)]
        norm = norm + [T.RandomErasing(p=0.25)]
    return T.Compose(train + norm), T.Compose([T.Resize((224, 224))] + norm[:2])


def subsample_lesions(df, frac, seed):
    """Stratified lesion-level subsample, identical for every model at a given (fold, frac)."""
    if frac >= 1.0:
        return df
    rng = np.random.RandomState(seed)
    les = df.groupby("lesion_id").label.first()
    keep = []
    for c, grp in les.groupby(les):
        n = max(1, int(math.ceil(frac * len(grp))))
        keep += list(rng.choice(grp.index.values, n, replace=False))
    return df[df.lesion_id.isin(set(keep))]


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            out.append(model(x.to(dev, non_blocking=True)).float().softmax(1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--hparams", default=None, help="json with selected lr per config")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.5, help="oversampling exponent")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no_test", action="store_true", help="tuning: validation only")
    a = ap.parse_args()

    cfg = dict(configs.ALL[a.config])
    lr = a.lr
    if lr is None:
        hp = json.load(open(a.hparams))
        lr = hp[cfg.get("inherit", a.config)]["lr"]
    out = a.out
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/metrics.json"):
        print("done already", out)
        return

    seed = 42 + a.fold
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sp = pd.read_csv(f"{a.data}/splits.csv")
    isic = pd.read_csv(f"{a.data}/isic_test.csv")
    trn = sp[(sp.split == "dev") & (sp.fold != a.fold)]
    trn = subsample_lesions(trn, cfg["train_frac"], seed=1000 + a.fold)
    val = sp[(sp.split == "dev") & (sp.fold == a.fold)]
    tst = sp[sp.split == "test"]

    model_kw = {k: cfg[k] for k in ["backbone", "gate", "branch", "gate_at", "n_qubits", "n_layers", "fusion_hidden"]}
    model = HybridNet(**model_kw).to(dev)
    pc = model.backbone.pretrained_cfg
    tf_train, tf_eval = transforms(cfg["aug"], pc["mean"], pc["std"])
    bs = configs.BATCH.get(cfg["backbone"], 64)

    counts = trn.label.value_counts().reindex(range(7), fill_value=0).values
    w = (counts.astype(float) ** -a.alpha)[trn.label.values]
    sampler = WeightedRandomSampler(torch.tensor(w), num_samples=len(trn), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    mk = lambda df, tf, **kw: DataLoader(ImgDS(a.data, df.image_id, df.label, tf), batch_size=bs,
                                         num_workers=a.workers, pin_memory=True, persistent_workers=False, **kw)
    dl_tr = mk(trn, tf_train, sampler=sampler, drop_last=True)
    dl_val = mk(val, tf_eval)

    new = model.new_parameters()
    new_ids = {id(p) for p in new}
    old = [p for p in model.parameters() if id(p) not in new_ids]
    opt = torch.optim.AdamW([{"params": old, "lr": lr}, {"params": new, "lr": 10 * lr}], weight_decay=1e-2)
    steps = a.epochs * len(dl_tr)
    warm = len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    best, best_state, best_ep, bad, log = float("inf"), None, -1, 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tl, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = F.cross_entropy(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(y)
            n += len(y)
        pv = predict(model, dl_val, dev)
        yv = val.label.values
        vloss = float(-np.log(np.clip(pv[np.arange(len(yv)), yv], 1e-12, 1)).mean())
        vacc = float((pv.argmax(1) == yv).mean())
        vf1 = float(f1_score(yv, pv.argmax(1), average="macro"))
        log.append(dict(epoch=ep, train_loss=tl / n, val_loss=vloss, val_acc=vacc, val_macro_f1=vf1,
                        time=time.time() - t0))
        print(json.dumps(log[-1]), flush=True)
        if vloss < best:
            best, best_ep, bad = vloss, ep, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= a.patience:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    pd.DataFrame(log).to_csv(f"{out}/log.csv", index=False)

    res = dict(val_ids=val.image_id.values, val_y=val.label.values, val_p=predict(model, dl_val, dev))
    if not a.no_test:
        res.update(test_ids=tst.image_id.values, test_y=tst.label.values,
                   test_p=predict(model, mk(tst, tf_eval), dev),
                   isic_ids=isic.image_id.values, isic_y=isic.label.values,
                   isic_p=predict(model, mk(isic, tf_eval), dev))
    np.savez_compressed(f"{out}/preds.npz", **res)
    if a.config in configs.SAVE_CKPT and not a.no_test:
        torch.save(best_state, f"{out}/ckpt.pt")

    meta = dict(config=a.config, fold=a.fold, cfg=cfg, description=describe(cfg), lr=lr, head_lr=10 * lr,
                batch=bs, epochs_max=a.epochs, patience=a.patience, alpha=a.alpha, best_epoch=best_ep,
                epochs_run=len(log), best_val_loss=best, n_train_images=len(trn),
                n_train_lesions=int(trn.lesion_id.nunique()), params=count_params(model),
                train_time_s=train_time, timm_name=BACKBONES[cfg["backbone"]]["timm"],
                versions=dict(python=platform.python_version(), torch=torch.__version__,
                              cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                              gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
                              timm=__import__("timm").__version__, numpy=np.__version__),
                host=platform.node())
    json.dump(meta, open(f"{out}/metrics.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
