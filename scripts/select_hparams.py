"""Pick each config's learning rate by fold-0 *validation* cross-entropy (equal
grid for every config; the test sets are never touched during tuning).
Writes results/hparams.json and results/tables/T10_lr_tuning.csv."""
import glob
import json
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
rows = []
for f in glob.glob(f"{ROOT}/results/tune/*/lr*/fold0/metrics.json"):
    m = json.load(open(f))
    log = pd.read_csv(os.path.join(os.path.dirname(f), "log.csv"))
    rows.append(dict(config=m["config"], lr=m["lr"], best_val_loss=m["best_val_loss"],
                     val_macro_f1_at_best=log.loc[log.val_loss.idxmin(), "val_macro_f1"],
                     best_epoch=m["best_epoch"], epochs_run=m["epochs_run"]))
df = pd.DataFrame(rows).sort_values(["config", "lr"])
sel = df.loc[df.groupby("config").best_val_loss.idxmin()]
df["selected"] = df.index.isin(sel.index)
os.makedirs(f"{ROOT}/results/tables", exist_ok=True)
df.to_csv(f"{ROOT}/results/tables/T10_lr_tuning.csv", index=False)
hp = {r.config: {"lr": r.lr, "val_loss": r.best_val_loss} for r in sel.itertuples()}
json.dump(hp, open(f"{ROOT}/results/hparams.json", "w"), indent=1)
counts = df.groupby("config").size()
print(sel[["config", "lr", "best_val_loss", "val_macro_f1_at_best"]].to_string(index=False))
incomplete = counts[counts < 3]
if len(incomplete):
    print("WARNING incomplete grids:", incomplete.to_dict(), file=sys.stderr)
