"""Build every table/figure requested by the editor and reviewers from saved
per-image predictions (results/runs/<config>/fold<k>/preds.npz).

Three clearly separated reporting units are used throughout:
  * fold mean +- SD : each of the 5 fold-specific models evaluated on the test set;
  * ensemble        : probabilities of the 5 fold models averaged (primary unit
                      for CIs and McNemar tests);
  * selected fold   : the single fold model with the lowest validation-fold
                      cross-entropy (the manuscript's "selected fold").
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import configs  # noqa: E402
import stats as S  # noqa: E402

CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
DSETS = {"test": "HAM10000 lesion-level test", "isic": "ISIC 2018 Task 3 test"}
MAIN_FAMILY = ["base", "qcg", "vqc", "lq", "se_vqc", "eca_vqc", "cbam_vqc", "qcg_mlp", "qcg_mlp_paper",
               "qcg_resmlp", "qcg_fourier", "qcg_poly", "qcg_trig", "se", "eca", "cbam", "eca_trig",
               "base_strong", "lq_strong", "effb0_strong", "effv2b0_strong", "mbv3_strong", "mobilevitv2_strong"]
# Table 14 controls (+ residual-MLP / Fourier / polynomial); base is tested in T3
CONTROLS = [c for c in MAIN_FAMILY if c not in ("lq", "base") and c in configs.CONFIGS]
BACKBONE_PAIRS = [("mobilenetv2", "base", "lq")] + [(b, f"{b}_C", f"{b}_H") for b in
                                                     ["resnet50", "efficientnet_b0", "vit_base", "swin_t"]]
CI_METRICS = ["acc", "bal_acc", "macro_f1", "macro_precision", "ece", "nll", "brier"]


def fmt_ci(x, lo, hi, d=4):
    return f"{x:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


class Store:
    def __init__(self, root, data, B):
        self.root = root
        sp = pd.read_csv(f"{data}/splits.csv")
        isic = pd.read_csv(f"{data}/isic_test.csv")
        self.lesion = {"test": dict(zip(sp.image_id, sp.lesion_id)),
                       "isic": dict(zip(isic.image_id, isic.archive_lesion_id))}
        self.cache = {}
        self.boot = {}
        self.B = B
        self.sp, self.isic = sp, isic

    def runs(self, cfg):
        if cfg in self.cache:
            return self.cache[cfg]
        out = []
        for k in range(5):
            d = f"{self.root}/runs/{cfg}/fold{k}"
            if os.path.exists(f"{d}/metrics.json"):
                z = dict(np.load(f"{d}/preds.npz", allow_pickle=True))
                z["meta"] = json.load(open(f"{d}/metrics.json"))
                out.append(z)
        self.cache[cfg] = out
        return out

    def complete(self, cfg):
        return len(self.runs(cfg)) == 5

    def ids(self, ds):
        r = self.runs("base") or next(self.runs(c) for c in configs.ALL if self.runs(c))
        return r[0][f"{ds}_ids"], r[0][f"{ds}_y"]

    def clusters(self, ds, ids=None):
        ids = self.ids(ds)[0] if ids is None else ids
        return np.array([self.lesion[ds][i] for i in ids])

    def bootstrap(self, ds):
        if ds not in self.boot:
            self.boot[ds] = S.ClusterBoot(self.clusters(ds), B=self.B)
        return self.boot[ds]

    def fold_probs(self, cfg, ds, calibrated=False):
        ps = []
        for r in self.runs(cfg):
            assert (r[f"{ds}_ids"] == self.ids(ds)[0]).all()
            p = r[f"{ds}_p"]
            if calibrated:
                p = S.apply_temperature(p, S.fit_temperature(r["val_p"], r["val_y"]))
            ps.append(p)
        return ps

    def ens(self, cfg, ds, calibrated=False):
        return np.mean(self.fold_probs(cfg, ds, calibrated), 0)

    def selected(self, cfg):
        rs = self.runs(cfg)
        return int(np.argmin([r["meta"]["best_val_loss"] for r in rs]))


# --------------------------------------------------------------------------
def table_main(st, out):
    rows = []
    base_params = st.runs("base")[0]["meta"]["params"] if st.runs("base") else np.nan
    for cfg in MAIN_FAMILY + [c for _, c0, c1 in BACKBONE_PAIRS[1:] for c in (c0, c1)]:
        if not st.complete(cfg):
            continue
        rs = st.runs(cfg)
        row = dict(config=cfg, description=rs[0]["meta"]["description"], backbone=rs[0]["meta"]["cfg"]["backbone"],
                   aug=rs[0]["meta"]["cfg"]["aug"], params=rs[0]["meta"]["params"],
                   lr=rs[0]["meta"]["lr"], mean_epochs=np.mean([r["meta"]["epochs_run"] for r in rs]))
        if row["backbone"] == "mobilenetv2":
            row["delta_params_vs_base"] = row["params"] - base_params
        y_val_m = [S.all_metrics(r["val_p"], r["val_y"]) for r in rs]
        row["val_loss_mean"] = np.mean([m["nll"] for m in y_val_m])
        row["val_macro_f1_mean"] = np.mean([m["macro_f1"] for m in y_val_m])
        for ds in DSETS:
            y = st.ids(ds)[1]
            fm = [S.all_metrics(p, y) for p in st.fold_probs(cfg, ds)]
            for m in ["acc", "bal_acc", "macro_f1", "macro_precision", "weighted_f1", "auroc_macro", "ece"]:
                v = np.array([f[m] for f in fm])
                row[f"{ds}_{m}_fold_mean"] = v.mean()
                row[f"{ds}_{m}_fold_sd"] = v.std(ddof=1)
            pe = st.ens(cfg, ds)
            em = S.all_metrics(pe, y)
            dist = st.bootstrap(ds).metric_dist(y, pe)
            for m in ["acc", "bal_acc", "macro_f1", "ece"]:
                lo, hi = S.ci(dist[m])
                row[f"{ds}_{m}_ens"] = em[m]
                row[f"{ds}_{m}_ens_ci_lo"], row[f"{ds}_{m}_ens_ci_hi"] = lo, hi
            row[f"{ds}_auroc_ens"] = em["auroc_macro"]
            sel = st.selected(cfg)
            sm = S.all_metrics(st.fold_probs(cfg, ds)[sel], y)
            row["selected_fold"] = sel
            row[f"{ds}_acc_selected"], row[f"{ds}_macro_f1_selected"] = sm["acc"], sm["macro_f1"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/T1_main_results.csv", index=False)
    return df


def compare(st, ref, comp, ds):
    y = st.ids(ds)[1]
    cl = st.clusters(ds)
    pa, pb = st.ens(ref, ds), st.ens(comp, ds)
    bt = st.bootstrap(ds)
    da, db = bt.metric_dist(y, pa), bt.metric_dist(y, pb)
    ma, mb = S.all_metrics(pa, y), S.all_metrics(pb, y)
    row = dict(dataset=ds, model_A=ref, model_B=comp)
    for m in ["acc", "bal_acc", "macro_f1", "ece", "nll", "brier"]:
        lo, hi = S.ci(da[m] - db[m])
        row[f"A_{m}"], row[f"B_{m}"] = ma[m], mb[m]
        row[f"delta_{m}"], row[f"delta_{m}_ci_lo"], row[f"delta_{m}_ci_hi"] = ma[m] - mb[m], lo, hi
        row[f"P_boot_A_better_{m}"] = float(np.mean((da[m] - db[m]) > 0)) if m not in ("ece", "nll", "brier") \
            else float(np.mean((da[m] - db[m]) < 0))
    row.update(S.mcnemar(y, pa, pb, cl))
    fa = [S.all_metrics(p, y) for p in st.fold_probs(ref, ds)]
    fb = [S.all_metrics(p, y) for p in st.fold_probs(comp, ds)]
    for m in ["acc", "macro_f1"]:
        fp = S.fold_paired([f[m] for f in fa], [f[m] for f in fb])
        row.update({f"{k}_{m}": v for k, v in fp.items()})
    # per-fold McNemar discordant counts (fold k of A vs fold k of B)
    pf = [S.mcnemar(y, a, b, cl) for a, b in zip(st.fold_probs(ref, ds), st.fold_probs(comp, ds))]
    row["per_fold_b"] = ";".join(str(x["b_A_right_B_wrong"]) for x in pf)
    row["per_fold_c"] = ";".join(str(x["c_A_wrong_B_right"]) for x in pf)
    row["per_fold_p_exact"] = ";".join(f"{x['p_mcnemar_exact']:.3g}" for x in pf)
    return row


def table_controls(st, out):
    rows = []
    for ds in DSETS:
        sub = [compare(st, "lq", c, ds) for c in CONTROLS if st.complete(c) and st.complete("lq")]
        if not sub:
            continue
        d = pd.DataFrame(sub)
        d["p_holm_exact"] = S.holm(d.p_mcnemar_exact)
        d["p_holm_clustered"] = S.holm(d.p_mcnemar_clustered)
        d["p_bh_clustered"] = S.bh(d.p_mcnemar_clustered)
        rows.append(d)
    df = pd.concat(rows) if rows else pd.DataFrame()
    df.to_csv(f"{out}/T2_lq_vs_controls.csv", index=False)
    return df


def table_backbones(st, out):
    rows = []
    for bb, c, h in BACKBONE_PAIRS:
        if st.complete(c) and st.complete(h):
            for ds in DSETS:
                r = compare(st, h, c, ds)
                r["backbone"] = bb
                r["params_C"] = st.runs(c)[0]["meta"]["params"]
                r["params_H"] = st.runs(h)[0]["meta"]["params"]
                rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        # Holm-Bonferroni across the backbone x dataset comparisons (10 when complete)
        df["n_tests_in_family"] = len(df)
        df["p_holm_exact"] = S.holm(df.p_mcnemar_exact)
        df["p_holm_clustered"] = S.holm(df.p_mcnemar_clustered)
        df["p_bh_clustered"] = S.bh(df.p_mcnemar_clustered)
    df.to_csv(f"{out}/T3_backbones_C_vs_H.csv", index=False)
    return df


def table_per_class(st, out, cfgs=("lq", "base")):
    rows, cms = [], []
    for cfg in cfgs:
        if not st.complete(cfg):
            continue
        for ds in DSETS:
            y = st.ids(ds)[1]
            bt = st.bootstrap(ds)
            sel = st.selected(cfg)
            for unit, p in [("ensemble", st.ens(cfg, ds)), (f"selected_fold_{sel}", st.fold_probs(cfg, ds)[sel])]:
                cm = S.confusion(y, p.argmax(1))
                m = S.metrics_from_cm(cm)
                dist = S.metrics_from_cm(bt.cm_dist(y, p.argmax(1)))
                for k, cname in enumerate(CLASSES):
                    tp, sup = cm[k, k], cm[k].sum()
                    neg = cm.sum() - sup
                    tn = neg - (cm[:, k].sum() - tp)
                    r = dict(config=cfg, dataset=ds, unit=unit, cls=cname, support=int(sup),
                             tp=int(tp), fp=int(cm[:, k].sum() - tp), fn=int(sup - tp), tn=int(tn))
                    for met in ["precision", "recall", "f1", "specificity"]:
                        lo, hi = S.ci(dist[met][:, k])
                        r[met], r[f"{met}_ci_lo"], r[f"{met}_ci_hi"] = float(m[met][k]), lo, hi
                    r["recall_wilson_lo"], r["recall_wilson_hi"] = S.wilson(tp, sup)
                    r["specificity_wilson_lo"], r["specificity_wilson_hi"] = S.wilson(tn, neg)
                    rows.append(r)
                for met in ["macro_precision", "macro_recall", "macro_f1", "weighted_f1", "acc"]:
                    lo, hi = S.ci(dist[met])
                    rows.append(dict(config=cfg, dataset=ds, unit=unit, cls=met, support=int(cm.sum()),
                                     f1=float(m[met]), f1_ci_lo=lo, f1_ci_hi=hi))
                c = pd.DataFrame(cm, index=[f"true_{x}" for x in CLASSES], columns=[f"pred_{x}" for x in CLASSES])
                c.insert(0, "config", cfg)
                c.insert(1, "dataset", ds)
                c.insert(2, "unit", unit)
                cms.append(c)
    pd.DataFrame(rows).to_csv(f"{out}/T4_per_class_with_CI.csv", index=False)
    if cms:
        pd.concat(cms).to_csv(f"{out}/T4b_confusion_matrices.csv")


def table_calibration(st, out, figdir):
    rows = []
    cfgs = [c for c in MAIN_FAMILY if st.complete(c)]
    for cfg in cfgs:
        for ds in DSETS:
            y = st.ids(ds)[1]
            for cal in (False, True):
                fm = [S.all_metrics(p, y) for p in st.fold_probs(cfg, ds, cal)]
                pe = st.ens(cfg, ds, cal)
                dist = st.bootstrap(ds).metric_dist(y, pe)
                r = dict(config=cfg, dataset=ds, temperature_scaled=cal)
                for m in ["ece", "nll", "brier"]:
                    v = [f[m] for f in fm]
                    r[f"{m}_fold_mean"], r[f"{m}_fold_sd"] = np.mean(v), np.std(v, ddof=1)
                    lo, hi = S.ci(dist[m])
                    r[f"{m}_ens"], r[f"{m}_ens_ci_lo"], r[f"{m}_ens_ci_hi"] = S.all_metrics(pe, y)[m], lo, hi
                if cal:
                    r["fold_temperatures"] = ";".join(f"{S.fit_temperature(rr['val_p'], rr['val_y']):.3f}"
                                                      for rr in st.runs(cfg))
                rows.append(r)
    pd.DataFrame(rows).to_csv(f"{out}/T5_calibration.csv", index=False)
    # reliability diagrams: LQ vs parameter-matched MLP control vs baseline
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [c for c in ["base", "qcg_mlp", "qcg_mlp_paper", "lq"] if st.complete(c)]
    if not show:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    colors = {"base": "#7f7f7f", "qcg_mlp": "#1f77b4", "qcg_mlp_paper": "#2ca02c", "lq": "#d62728"}
    for ax, ds in zip(axes, DSETS):
        y = st.ids(ds)[1]
        ax.plot([0, 1], [0, 1], ls="--", c="k", lw=0.8)
        for cfg in show:
            p = st.ens(cfg, ds)
            conf, corr, bins = S.calib_stats(p, y)
            xs, ys = [], []
            for b in range(15):
                mk = bins == b
                if mk.sum() >= 5:
                    xs.append(conf[mk].mean())
                    ys.append(corr[mk].mean())
            ax.plot(xs, ys, marker="o", ms=3, c=colors[cfg], label=f"{cfg} (ECE {S.ece(p, y):.3f})")
        ax.set_title(DSETS[ds])
        ax.set_xlabel("Confidence (top-label)")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{figdir}/F1_reliability_diagrams.png", dpi=200)
    plt.close()


def table_learning_curves(st, out, figdir):
    rows = []
    for m in configs.LC_MODELS:
        for f in list(configs.LC_FRACS) + [1.0]:
            cfg = m if f == 1.0 else f"{m}_frac{int(f * 100)}"
            if not st.complete(cfg):
                continue
            for ds in DSETS:
                y = st.ids(ds)[1]
                fm = [S.all_metrics(p, y) for p in st.fold_probs(cfg, ds)]
                r = dict(model=m, frac=f, config=cfg, dataset=ds,
                         n_train_images_mean=np.mean([x["meta"]["n_train_images"] for x in st.runs(cfg)]))
                for met in ["acc", "bal_acc", "macro_f1", "ece"]:
                    v = [x[met] for x in fm]
                    r[f"{met}_mean"], r[f"{met}_sd"] = np.mean(v), np.std(v, ddof=1)
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/T6_learning_curves.csv", index=False)
    # LQ minus each comparator at every fraction
    drows = []
    for comp in [c for c in configs.LC_MODELS if c != "lq"]:
        for f in list(configs.LC_FRACS) + [1.0]:
            a = "lq" if f == 1.0 else f"lq_frac{int(f * 100)}"
            b = comp if f == 1.0 else f"{comp}_frac{int(f * 100)}"
            if st.complete(a) and st.complete(b):
                for ds in DSETS:
                    r = compare(st, a, b, ds)
                    r["frac"] = f
                    drows.append(r)
    dd = pd.DataFrame(drows)
    if len(dd):
        for ds in DSETS:
            mk = dd.dataset == ds
            dd.loc[mk, "p_holm_clustered"] = S.holm(dd.loc[mk, "p_mcnemar_clustered"])
    dd.to_csv(f"{out}/T6b_learning_curve_differences.csv", index=False)
    if len(df):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax, ds in zip(axes, DSETS):
            for m in configs.LC_MODELS:
                s = df[(df.model == m) & (df.dataset == ds)].sort_values("frac")
                if len(s):
                    ax.errorbar(s.frac * 100, s.macro_f1_mean, yerr=s.macro_f1_sd, marker="o", capsize=3, label=m)
            ax.set_xscale("log")
            ax.set_xticks([10, 25, 50, 100])
            ax.set_xticklabels(["10", "25", "50", "100"])
            ax.set_xlabel("% of training lesions")
            ax.set_ylabel("Macro F1 (mean ± SD over 5 folds)")
            ax.set_title(DSETS[ds])
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{figdir}/F2_learning_curves.png", dpi=200)
        plt.close()


def table_ablations(st, out):
    groups = {"fusion_hidden": ["lq_f16", "lq_f32", "lq", "lq_f128", "lq_f256"],
              "qcg_placement": ["lq_p4", "lq_p6", "lq", "lq_p56", "lq_p3456"]}
    rows = []
    for g, cfgs in groups.items():
        for cfg in cfgs:
            if not st.complete(cfg):
                continue
            rs = st.runs(cfg)
            vm = [S.all_metrics(r["val_p"], r["val_y"]) for r in rs]
            r = dict(group=g, config=cfg, params=rs[0]["meta"]["params"],
                     gate_at=str(rs[0]["meta"]["cfg"]["gate_at"] or "(4, 6) default"),
                     val_loss_mean=np.mean([m["nll"] for m in vm]), val_loss_sd=np.std([m["nll"] for m in vm], ddof=1),
                     val_macro_f1_mean=np.mean([m["macro_f1"] for m in vm]), val_acc_mean=np.mean([m["acc"] for m in vm]))
            for ds in DSETS:
                y = st.ids(ds)[1]
                fm = [S.all_metrics(p, y) for p in st.fold_probs(cfg, ds)]
                for met in ["acc", "macro_f1"]:
                    v = [x[met] for x in fm]
                    r[f"{ds}_{met}_mean"], r[f"{ds}_{met}_sd"] = np.mean(v), np.std(v, ddof=1)
            rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df["selected_by_val_loss"] = df.groupby("group").val_loss_mean.transform("min") == df.val_loss_mean
    df.to_csv(f"{out}/T7_ablations_validation_selection.csv", index=False)


def table_subgroups(st, out, data):
    if not (st.complete("lq") and st.complete("base")):
        return
    sp = st.sp.set_index("image_id")
    ita_path = f"{data}/ita.csv"
    ita = pd.read_csv(ita_path).set_index("image_id") if os.path.exists(ita_path) else None
    rows = []
    for ds in DSETS:
        ids, y = st.ids(ds)
        if ds == "test":
            meta = sp.loc[ids]
            groups = {"sex": meta.sex.fillna("unknown").values,
                      "age": pd.cut(meta.age, [0, 30, 50, 70, 120], right=False,
                                    labels=["<30", "30-49", "50-69", ">=70"]).astype(str).values,
                      "site": meta.localization.fillna("unknown").values,
                      "dx_type": meta.dx_type.values}
            if ita is not None:
                cat = ita.reindex(ids).ita_category.fillna("undetermined")
                groups["ITA_skin_tone_proxy"] = cat.replace({"tan": "tan_or_darker", "brown": "tan_or_darker",
                                                             "dark": "tan_or_darker"}).values
        else:
            meta = st.isic.set_index("image_id").loc[ids]
            groups = {"sex": meta.sex.fillna("unknown").values,
                      "age": pd.cut(meta.age_approx, [0, 30, 50, 70, 120], right=False,
                                    labels=["<30", "30-49", "50-69", ">=70"]).astype(str).values,
                      "site": meta.anatom_site.fillna("unknown").values}
        cl = st.clusters(ds)
        pa, pb = st.ens("lq", ds), st.ens("base", ds)
        for g, vals in groups.items():
            for v in pd.unique(vals):
                mk = vals == v
                if mk.sum() < 20:
                    continue
                bt = S.ClusterBoot(cl[mk], B=2000)
                da, db = bt.metric_dist(y[mk], pa[mk]), bt.metric_dist(y[mk], pb[mk])
                r = dict(dataset=ds, attribute=g, group=v, n_images=int(mk.sum()), n_lesions=len(np.unique(cl[mk])))
                for met in ["acc", "bal_acc"]:
                    a_, b_ = S.metrics_from_cm(S.confusion(y[mk], pa[mk].argmax(1)))[met], \
                             S.metrics_from_cm(S.confusion(y[mk], pb[mk].argmax(1)))[met]
                    r[f"lq_{met}"], (r[f"lq_{met}_ci_lo"], r[f"lq_{met}_ci_hi"]) = float(a_), S.ci(da[met])
                    r[f"base_{met}"] = float(b_)
                    r[f"delta_{met}"], (r[f"delta_{met}_ci_lo"], r[f"delta_{met}_ci_hi"]) = float(a_ - b_), S.ci(da[met] - db[met])
                r["lq_ece"] = S.ece(pa[mk], y[mk])
                rows.append(r)
    pd.DataFrame(rows).to_csv(f"{out}/T8_subgroups.csv", index=False)


def table_dedup(st, out, data):
    path = f"{data}/../results/dedup/isic_flags.csv"
    if not os.path.exists(path) or not (st.complete("lq") and st.complete("base")):
        return
    fl = pd.read_csv(path).set_index("image_id")
    ids, y = st.ids("isic")
    keep = ~fl.reindex(ids).any_flag.fillna(False).values.astype(bool)
    cl = st.clusters("isic")
    rows = []
    for cfg in [c for c in MAIN_FAMILY + [c for _, c0, c1 in BACKBONE_PAIRS[1:] for c in (c0, c1)] if st.complete(c)]:
        p = st.ens(cfg, "isic")
        bt = S.ClusterBoot(cl[keep], B=2000)
        d = bt.metric_dist(y[keep], p[keep])
        m = S.all_metrics(p[keep], y[keep])
        rows.append(dict(config=cfg, n_kept=int(keep.sum()), n_removed=int((~keep).sum()),
                         acc=m["acc"], acc_ci=S.ci(d["acc"]), macro_f1=m["macro_f1"], macro_f1_ci=S.ci(d["macro_f1"]),
                         acc_full=S.all_metrics(p, y)["acc"]))
    pd.DataFrame(rows).to_csv(f"{out}/T9_isic_deduplicated.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--B", type=int, default=S.B_DEFAULT)
    a = ap.parse_args()
    out, fig = f"{a.results}/tables", f"{a.results}/figures"
    os.makedirs(out, exist_ok=True)
    os.makedirs(fig, exist_ok=True)
    st = Store(a.results, a.data, a.B)
    done = {c: len(st.runs(c)) for c in configs.ALL}
    pd.Series(done, name="n_folds_complete").to_csv(f"{out}/T0_run_status.csv")
    print("complete configs:", sum(v == 5 for v in done.values()), "/", len(done))
    table_main(st, out)
    table_controls(st, out)
    table_backbones(st, out)
    table_per_class(st, out)
    print("tables written to", out)


if __name__ == "__main__":
    main()
