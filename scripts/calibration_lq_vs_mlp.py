"""Calibration of LQ-HybridNet vs the parameter-matched MLP control (Reviewer 2,
point 2). Both models: same QCG, same head, same recipe; only the 4-qubit VQC is
replaced by a 32->4->4 MLP. Uses the saved test-set probabilities of all 5 folds.

Outputs (results/calibration/):
  T5_calibration_lq_vs_mlp.csv   ECE (15 bins, top-label), NLL, Brier per model:
                                 fold mean +- SD, 5-model ensemble with
                                 lesion-clustered 95% CI, and LQ - MLP paired
                                 differences (clustered bootstrap CI, fold t-test)
  per_fold_calibration.csv       the same metrics for every fold
  reliability_bins.csv           numbers behind the reliability diagrams
  F1_reliability_lq_vs_mlp.png/.pdf
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/src")
import analyze as A  # noqa: E402
import stats as S  # noqa: E402

MODELS = {"lq": "LQ-HybridNet", "qcg_mlp": "MLP control"}
COLORS = {"lq": "#2a78d6", "qcg_mlp": "#eb6834"}
MARKERS = {"lq": "o", "qcg_mlp": "s"}
LINES = {"lq": "-", "qcg_mlp": "--"}
NB = 15
METRICS = ["ece", "nll", "brier"]


def bins_table(p, y):
    conf, corr, b = S.calib_stats(p, y, NB)
    rows = []
    for k in range(NB):
        m = b == k
        rows.append(dict(bin=k, lo=k / NB, hi=(k + 1) / NB, n=int(m.sum()),
                         mean_conf=conf[m].mean() if m.any() else np.nan,
                         accuracy=corr[m].mean() if m.any() else np.nan))
    return pd.DataFrame(rows)


def main():
    out = f"{ROOT}/results/calibration"
    os.makedirs(out, exist_ok=True)
    st = A.Store(f"{ROOT}/results", f"{ROOT}/data", S.B_DEFAULT)
    for c in MODELS:
        assert st.complete(c), f"{c}: not all 5 folds finished"

    summ, folds, rel = [], [], []
    for ds in A.DSETS:
        y = st.ids(ds)[1]
        bt = st.bootstrap(ds)
        dist, fm = {}, {}
        for c in MODELS:
            pe = st.ens(c, ds)
            dist[c] = bt.metric_dist(y, pe)
            fm[c] = [S.all_metrics(p, y) for p in st.fold_probs(c, ds)]
            em = S.all_metrics(pe, y)
            r = dict(dataset=A.DSETS[ds], model=MODELS[c], n_images=len(y))
            for m in METRICS:
                v = np.array([f[m] for f in fm[c]])
                r[f"{m}_fold_mean"], r[f"{m}_fold_sd"] = v.mean(), v.std(ddof=1)
                lo, hi = S.ci(dist[c][m])
                r[f"{m}_ensemble"], r[f"{m}_ensemble_ci_lo"], r[f"{m}_ensemble_ci_hi"] = em[m], lo, hi
            summ.append(r)
            for k, f in enumerate(fm[c]):
                folds.append(dict(dataset=A.DSETS[ds], model=MODELS[c], fold=k, **{m: f[m] for m in METRICS}))
            t = bins_table(pe, y)
            t.insert(0, "model", MODELS[c])
            t.insert(0, "dataset", A.DSETS[ds])
            rel.append(t)
        # paired difference LQ - MLP (negative = LQ better calibrated)
        r = dict(dataset=A.DSETS[ds], model="Difference LQ - MLP (negative favours LQ)", n_images=len(y))
        for m in METRICS:
            fp = S.fold_paired([f[m] for f in fm["lq"]], [f[m] for f in fm["qcg_mlp"]])
            r[f"{m}_fold_mean"], r[f"{m}_fold_sd"] = fp["fold_diff_mean"], fp["fold_diff_sd"]
            r[f"{m}_fold_diff_ci_lo"], r[f"{m}_fold_diff_ci_hi"] = fp["fold_diff_ci_lo"], fp["fold_diff_ci_hi"]
            r[f"{m}_fold_paired_t_p"] = fp["p_fold_paired_t"]
            d = dist["lq"][m] - dist["qcg_mlp"][m]
            lo, hi = S.ci(d)
            r[f"{m}_ensemble"] = S.all_metrics(st.ens("lq", ds), y)[m] - S.all_metrics(st.ens("qcg_mlp", ds), y)[m]
            r[f"{m}_ensemble_ci_lo"], r[f"{m}_ensemble_ci_hi"] = lo, hi
            r[f"{m}_P_boot_LQ_better"] = float(np.mean(d < 0))
        summ.append(r)

    pd.DataFrame(summ).to_csv(f"{out}/T5_calibration_lq_vs_mlp.csv", index=False)
    pd.DataFrame(folds).to_csv(f"{out}/per_fold_calibration.csv", index=False)
    rel = pd.concat(rel)
    rel.to_csv(f"{out}/reliability_bins.csv", index=False)
    plot(st, rel, pd.DataFrame(summ), pd.DataFrame(folds), out)


def plot(st, rel, summ, folds, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888888", "axes.labelcolor": "#222222",
                         "xtick.color": "#555555", "ytick.color": "#555555"})
    fig = plt.figure(figsize=(10.5, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.1], width_ratios=[1, 1, 0.8], hspace=0.08, wspace=0.32)
    for j, ds in enumerate(A.DSETS):
        ax = fig.add_subplot(gs[0, j])
        axh = fig.add_subplot(gs[1, j], sharex=ax)
        ax.plot([0, 1], [0, 1], ls=":", c="#999999", lw=1, label="Perfect calibration")
        y = st.ids(ds)[1]
        for c, name in MODELS.items():
            # thin per-fold curves show fold-to-fold variability
            for p in st.fold_probs(c, ds):
                t = bins_table(p, y)
                t = t[t.n >= 10]
                ax.plot(t.mean_conf, t.accuracy, c=COLORS[c], lw=0.6, alpha=0.25, ls=LINES[c])
            t = rel[(rel.dataset == A.DSETS[ds]) & (rel.model == name) & (rel.n >= 10)]
            s = summ[(summ.dataset == A.DSETS[ds]) & (summ.model == name)].iloc[0]
            ax.plot(t.mean_conf, t.accuracy, c=COLORS[c], lw=2, ls=LINES[c], marker=MARKERS[c], ms=5,
                    mec="white", mew=0.8,
                    label=f"{name}\nECE {s.ece_ensemble:.3f} [{s.ece_ensemble_ci_lo:.3f}, {s.ece_ensemble_ci_hi:.3f}]")
            hb = rel[(rel.dataset == A.DSETS[ds]) & (rel.model == name)]
            off = -0.012 if c == "lq" else 0.012
            axh.bar((hb.lo + hb.hi) / 2 + off, hb.n, width=1 / NB / 2 - 0.004, color=COLORS[c],
                    hatch=None if c == "lq" else "////", edgecolor="white", lw=0.5)
        ax.set_xlim(0.15, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("Observed accuracy")
        ax.set_title(f"{A.DSETS[ds]} (n={len(y)})", fontsize=10, loc="left")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7, loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.2)
        plt.setp(ax.get_xticklabels(), visible=False)
        axh.set_xlabel("Predicted confidence (top-label), 15 bins")
        axh.set_ylabel("Images")
        axh.set_yscale("log")
        axh.grid(axis="y", alpha=0.25, lw=0.5)
    # per-fold ECE, paired by fold
    ax = fig.add_subplot(gs[:, 2])
    for j, ds in enumerate(A.DSETS):
        f = folds[folds.dataset == A.DSETS[ds]]
        a = f[f.model == MODELS["lq"]].sort_values("fold").ece.values
        b = f[f.model == MODELS["qcg_mlp"]].sort_values("fold").ece.values
        x0, x1 = 2 * j, 2 * j + 0.8
        for u, v in zip(a, b):
            ax.plot([x0, x1], [u, v], c="#bbbbbb", lw=0.8, zorder=1)
        ax.scatter([x0] * 5, a, c=COLORS["lq"], marker="o", s=30, zorder=2, edgecolor="white", lw=0.8)
        ax.scatter([x1] * 5, b, c=COLORS["qcg_mlp"], marker="s", s=30, zorder=2, edgecolor="white", lw=0.8)
        d = summ[(summ.dataset == A.DSETS[ds]) & summ.model.str.startswith("Difference")].iloc[0]
        ax.text(x0 + 0.4, 0.004,
                f"Δ ens. {d.ece_ensemble:+.3f}\n[{d.ece_ensemble_ci_lo:+.3f}, {d.ece_ensemble_ci_hi:+.3f}]",
                ha="center", va="bottom", fontsize=7, color="#333333")
    ax.text(1.4, 0.0, "", fontsize=1)
    ax.set_xticks([0, 0.8, 2, 2.8])
    ax.set_xticklabels(["LQ", "MLP", "LQ", "MLP"])
    ax.text(0.4, -0.07, "HAM test", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
    ax.text(2.4, -0.07, "ISIC test", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
    ax.set_xlim(-0.5, 3.3)
    ax.set_ylim(0, None)
    ax.set_ylabel("ECE per fold model (lower is better)")
    ax.set_title("Fold-level ECE", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.suptitle("Reliability: LQ-HybridNet vs parameter-matched MLP control "
                 "(bold = 5-model ensemble, thin = individual folds)", fontsize=10, x=0.02, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/F1_reliability_lq_vs_mlp.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
