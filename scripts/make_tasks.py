"""Write task lists (one training run per line) for the SLURM array workers.

  tune : every config in configs.TUNED x LR grid, fold 0, validation only
  main : every config in configs.CONFIGS x 5 folds (needs results/hparams.json)
  extra: ablations + learning curves x 5 folds (inherit LRs from hparams.json)
Lines are ordered most-expensive first so array workers finish together.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/src")
import configs  # noqa: E402

COST = {"vit_base": 12, "swin_t": 5, "resnet50": 3, "efficientnetv2_b0": 1.5, "efficientnet_b0": 1.5,
        "mobilevitv2": 2.5, "mobilenetv3_large": 1, "mobilenetv2": 1}


def cost(cfg):
    c = configs.ALL[cfg]
    return COST[c["backbone"]] * c["train_frac"] * (1.5 if c.get("gate_at") and len(c["gate_at"]) > 3 else 1)


def main(phase):
    R = f"{ROOT}/results"
    lines = []
    if phase == "tune":
        for cfg in configs.TUNED:
            for lr in configs.LR_GRID:
                out = f"{R}/tune/{cfg}/lr{lr:g}/fold0"
                lines.append((cost(cfg), f"--config {cfg} --fold 0 --lr {lr:g} --epochs 20 --patience 6 --no_test --out {out}"))
    else:
        names = configs.CONFIGS if phase == "main" else {**configs.ABLATIONS, **configs.LEARNING}
        for cfg in names:
            for k in range(5):
                lines.append((cost(cfg), f"--config {cfg} --fold {k} --lr {configs.lr_for(cfg):g} --out {R}/runs/{cfg}/fold{k}"))
    lines.sort(key=lambda x: -x[0])
    os.makedirs(f"{ROOT}/slurm/tasks", exist_ok=True)
    with open(f"{ROOT}/slurm/tasks/{phase}.txt", "w") as f:
        f.write("\n".join(l for _, l in lines) + "\n")
    print(phase, len(lines), "tasks; relative GPU cost", round(sum(c for c, _ in lines), 1))


if __name__ == "__main__":
    main(sys.argv[1])
