"""Assemble results/REPORT.md from the CSV tables (all numbers come from the
saved predictions; nothing is typed in by hand)."""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DS = {"test": "HAM10000 lesion-level test (1002 images / 747 lesions)",
      "isic": "ISIC 2018 Task 3 test (1512 images)"}


def pct(x):
    return f"{100 * x:.2f}"


def ci(r, k, scale=100, d=2):
    return f"{scale * r[k]:.{d}f} [{scale * r[k + '_ci_lo']:.{d}f}, {scale * r[k + '_ci_hi']:.{d}f}]"


def pv(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def md(df):
    return df.to_markdown(index=False) if len(df) else "_not available yet_"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"{ROOT}/results")
    a = ap.parse_args()
    R, T = a.results, f"{a.results}/tables"
    rd = lambda n: pd.read_csv(f"{T}/{n}") if os.path.exists(f"{T}/{n}") and os.path.getsize(f"{T}/{n}") > 5 else pd.DataFrame()
    L = ["# LQ-HybridNet revision: results generated from saved predictions\n",
         "Reporting units: **fold mean ± SD** = 5 fold-specific models each evaluated on the test set; "
         "**ensemble** = average of the 5 models' probabilities (unit for CIs and McNemar tests); "
         "**selected fold** = single model with the lowest validation-fold cross-entropy. "
         "95% CIs: lesion-clustered percentile bootstrap (5000 lesion resamples; paired for differences). "
         "McNemar: exact binomial on discordant images and clustered McNemar (Durkalski 2003) on lesions; "
         "Holm-Bonferroni within each family.\n"]

    st = rd("T0_run_status.csv")
    if len(st):
        st.columns = ["config", "folds"]
        L.append(f"Runs complete: {(st.folds == 5).sum()} / {len(st)} configs.\n")

    m = rd("T1_main_results.csv")
    if len(m):
        L.append("## 1. Main results (MobileNetV2 family, strong lightweight baselines, other backbones)\n")
        for ds in DS:
            t = pd.DataFrame({
                "config": m.config, "params": (m.params / 1e6).round(3).astype(str) + "M",
                "acc fold mean±SD": [f"{pct(x)} ± {pct(s)}" for x, s in zip(m[f"{ds}_acc_fold_mean"], m[f"{ds}_acc_fold_sd"])],
                "macro-F1 fold mean±SD": [f"{x:.4f} ± {s:.4f}" for x, s in zip(m[f"{ds}_macro_f1_fold_mean"], m[f"{ds}_macro_f1_fold_sd"])],
                "bal-acc fold mean": [pct(x) for x in m[f"{ds}_bal_acc_fold_mean"]],
                "ensemble acc [95% CI]": [ci(r, f"{ds}_acc_ens") for _, r in m.iterrows()],
                "ensemble macro-F1 [95% CI]": [ci(r, f"{ds}_macro_f1_ens", 1, 4) for _, r in m.iterrows()],
                "ensemble ECE": [f"{x:.4f}" for x in m[f"{ds}_ece_ens"]],
                "selected-fold acc": [pct(x) for x in m[f"{ds}_acc_selected"]],
            })
            L.append(f"### {DS[ds]}\n\n{md(t)}\n")

    c = rd("T2_lq_vs_controls.csv")
    if len(c):
        L.append("## 2. LQ-HybridNet (A) vs every control (B): differences, discordant counts, tests\n")
        for ds in DS:
            s = c[c.dataset == ds]
            t = pd.DataFrame({
                "B": s.model_B,
                "Δacc pp [95% CI]": [ci(r, "delta_acc") for _, r in s.iterrows()],
                "Δmacro-F1 [95% CI]": [ci(r, "delta_macro_f1", 1, 4) for _, r in s.iterrows()],
                "Δbal-acc pp [95% CI]": [ci(r, "delta_bal_acc") for _, r in s.iterrows()],
                "b (A✓B✗)": s.b_A_right_B_wrong, "c (A✗B✓)": s.c_A_wrong_B_right,
                "p exact": s.p_mcnemar_exact.map(pv), "p clustered": s.p_mcnemar_clustered.map(pv),
                "p Holm (clustered)": s.p_holm_clustered.map(pv),
                "fold Δacc mean±SD": [f"{100 * x:.2f} ± {100 * y:.2f}" for x, y in zip(s.fold_diff_mean_acc, s.fold_diff_sd_acc)],
            })
            L.append(f"### {DS[ds]}\n\n{md(t)}\n")

    b = rd("T3_backbones_C_vs_H.csv")
    if len(b):
        L.append(f"## 3. Classical (C) vs hybrid (H) per backbone — Holm across all {len(b)} backbone×dataset tests\n")
        t = pd.DataFrame({
            "backbone": b.backbone, "dataset": b.dataset,
            "C acc": b.B_acc.map(pct), "H acc": b.A_acc.map(pct),
            "Δacc pp [95% CI]": [ci(r, "delta_acc") for _, r in b.iterrows()],
            "Δmacro-F1 [95% CI]": [ci(r, "delta_macro_f1", 1, 4) for _, r in b.iterrows()],
            "b": b.b_A_right_B_wrong, "c": b.c_A_wrong_B_right,
            "p exact": b.p_mcnemar_exact.map(pv), "p clustered": b.p_mcnemar_clustered.map(pv),
            "p Holm": b.p_holm_clustered.map(pv), "ΔParams": (b.params_H - b.params_C).map(lambda x: f"+{x / 1e3:.1f}K"),
        })
        L.append(md(t) + "\n")

    pc = rd("T4_per_class_with_CI.csv")
    if len(pc):
        L.append("## 4. Per-class metrics with lesion-clustered 95% CIs (recomputed from the confusion matrix)\n")
        for (cfg, ds, unit), s in pc[pc.cls.isin(["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"])].groupby(
                ["config", "dataset", "unit"], sort=False):
            if cfg != "lq":
                continue
            t = pd.DataFrame({"class": s.cls, "support": s.support, "TP": s.tp, "FP": s.fp, "FN": s.fn,
                              "precision [CI]": [ci(r, "precision", 1, 3) for _, r in s.iterrows()],
                              "recall/sensitivity [CI]": [ci(r, "recall", 1, 3) for _, r in s.iterrows()],
                              "F1 [CI]": [ci(r, "f1", 1, 3) for _, r in s.iterrows()],
                              "specificity [CI]": [ci(r, "specificity", 1, 3) for _, r in s.iterrows()],
                              "recall Wilson CI": [f"[{x:.3f}, {y:.3f}]" for x, y in zip(s.recall_wilson_lo, s.recall_wilson_hi)]})
            L.append(f"### LQ-HybridNet, {DS[ds]}, {unit}\n\n{md(t)}\n")

    cal = rd("T5_calibration.csv")
    if len(cal):
        L.append("## 5. Calibration (top-label ECE, 15 bins; NLL; Brier), before / after temperature scaling fitted on each validation fold\n")
        s = cal[cal.config.isin(["base", "lq", "qcg_mlp", "qcg_mlp_paper", "qcg_trig", "eca", "cbam"])]
        t = pd.DataFrame({"config": s.config, "dataset": s.dataset, "TS": s.temperature_scaled,
                          "ECE fold mean±SD": [f"{x:.4f} ± {y:.4f}" for x, y in zip(s.ece_fold_mean, s.ece_fold_sd)],
                          "ECE ensemble [CI]": [ci(r, "ece_ens", 1, 4) for _, r in s.iterrows()],
                          "NLL ensemble [CI]": [ci(r, "nll_ens", 1, 4) for _, r in s.iterrows()],
                          "Brier ensemble [CI]": [ci(r, "brier_ens", 1, 4) for _, r in s.iterrows()]})
        L.append(md(t) + "\n\nReliability diagrams: `figures/F1_reliability_diagrams.png`.\n")
        if len(c):
            s = c[c.model_B.isin(["qcg_mlp", "qcg_mlp_paper", "base"])]
            t = pd.DataFrame({"B": s.model_B, "dataset": s.dataset,
                              "ΔECE (A−B) [CI]": [ci(r, "delta_ece", 1, 4) for _, r in s.iterrows()],
                              "ΔNLL [CI]": [ci(r, "delta_nll", 1, 4) for _, r in s.iterrows()],
                              "ΔBrier [CI]": [ci(r, "delta_brier", 1, 4) for _, r in s.iterrows()]})
            L.append("ECE/NLL/Brier differences, LQ-HybridNet (A) minus control (B), uncalibrated ensembles (negative = LQ better calibrated):\n\n" + md(t) + "\n")

    lc = rd("T6_learning_curves.csv")
    if len(lc):
        L.append("## 6. Learning curves (stratified lesion-level subsamples of each training split, identical across models)\n")
        for ds in DS:
            s = lc[lc.dataset == ds]
            t = s.pivot_table(index="model", columns="frac", values="macro_f1_mean").round(4)
            t2 = s.pivot_table(index="model", columns="frac", values="acc_mean").mul(100).round(2)
            L.append(f"### {DS[ds]} — macro-F1 (fold mean)\n\n{t.reset_index().to_markdown(index=False)}\n\n"
                     f"accuracy (fold mean, %)\n\n{t2.reset_index().to_markdown(index=False)}\n")
        d = rd("T6b_learning_curve_differences.csv")
        if len(d):
            t = pd.DataFrame({"frac": d.frac, "dataset": d.dataset, "B": d.model_B,
                              "Δmacro-F1 (LQ−B) [CI]": [ci(r, "delta_macro_f1", 1, 4) for _, r in d.iterrows()],
                              "Δacc pp [CI]": [ci(r, "delta_acc") for _, r in d.iterrows()],
                              "p Holm (clustered)": d.p_holm_clustered.map(pv)})
            L.append(md(t) + "\n\nFigure: `figures/F2_learning_curves.png`.\n")

    ab = rd("T7_ablations_validation_selection.csv")
    if len(ab):
        L.append("## 7. Architecture ablations — selection by mean validation-fold loss only (test shown for information)\n")
        t = ab[["group", "config", "gate_at", "params", "val_loss_mean", "val_macro_f1_mean", "selected_by_val_loss",
                "test_acc_mean", "test_macro_f1_mean", "isic_acc_mean", "isic_macro_f1_mean"]].round(4)
        L.append(md(t) + "\n")

    sg = rd("T8_subgroups.csv")
    if len(sg):
        L.append("## 8. Subgroups (ensembles; groups with ≥20 images) — LQ-HybridNet vs MobileNetV2\n")
        t = pd.DataFrame({"dataset": sg.dataset, "attribute": sg.attribute, "group": sg.group, "n img": sg.n_images,
                          "n lesions": sg.n_lesions, "LQ acc [CI]": [ci(r, "lq_acc") for _, r in sg.iterrows()],
                          "LQ bal-acc [CI]": [ci(r, "lq_bal_acc") for _, r in sg.iterrows()],
                          "Δacc vs base [CI]": [ci(r, "delta_acc") for _, r in sg.iterrows()],
                          "LQ ECE": sg.lq_ece.round(4)})
        L.append(md(t) + "\n")

    if os.path.exists(f"{R}/dedup/summary.json"):
        dd = json.load(open(f"{R}/dedup/summary.json"))
        L.append("## 9. HAM10000 ↔ ISIC 2018 test overlap\n\n```\n" + json.dumps(dd, indent=1) + "\n```\n")
        t9 = rd("T9_isic_deduplicated.csv")
        if len(t9):
            L.append(md(t9) + "\n")

    if os.path.exists(f"{R}/latency/latency.csv"):
        lt = pd.read_csv(f"{R}/latency/latency.csv").round(3)
        L.append("## 10. Latency (end-to-end incl. host↔device transfer)\n\n" + md(lt) + "\n")
        if os.path.exists(f"{R}/latency/env.json"):
            L.append("```\n" + open(f"{R}/latency/env.json").read() + "\n```\n")

    if os.path.exists(f"{R}/gradcam/gradcam_summary.csv"):
        g = pd.read_csv(f"{R}/gradcam/gradcam_per_fold.csv").groupby("config").agg(["mean", "std"]).round(4)
        L.append("## 11. Grad-CAM localisation vs expert masks (all 1002 HAM test images, 5 folds)\n\n"
                 + g.reset_index().to_markdown(index=False) + "\n")
        gc = pd.read_csv(f"{R}/gradcam/gradcam_lq_vs_others.csv").round(4)
        L.append(md(gc) + "\n")

    tu = rd("T10_lr_tuning.csv")
    if len(tu):
        L.append("## 12. Learning-rate selection (same 3-value grid for every config; fold-0 validation loss)\n\n"
                 + md(tu.round(4)) + "\n")

    if os.path.exists(f"{R}/software_versions.txt"):
        L.append("## 13. Software\n\n```\n" + open(f"{R}/software_versions.txt").read() + "```\n")

    # honest automatic verdict
    if len(c):
        L.append("## Verdict (automatic): LQ-HybridNet vs controls, ensemble predictions, Holm-adjusted clustered McNemar\n")
        for ds in DS:
            s = c[c.dataset == ds]
            win = s[(s.delta_acc_ci_lo > 0) & (s.p_holm_clustered < 0.05)].model_B.tolist()
            lose = s[(s.delta_acc_ci_hi < 0) & (s.p_holm_clustered < 0.05)].model_B.tolist()
            tie = [x for x in s.model_B if x not in win + lose]
            L.append(f"- **{DS[ds]}** — significantly better than: {win or 'none'}; "
                     f"significantly worse than: {lose or 'none'}; no significant difference: {tie or 'none'}.")
        L.append("")
    open(f"{R}/REPORT.md", "w").write("\n".join(L))
    print("wrote", f"{R}/REPORT.md")


if __name__ == "__main__":
    main()
