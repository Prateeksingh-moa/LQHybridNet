"""Statistics used for the revision.

Confidence intervals: non-parametric *lesion-clustered* percentile bootstrap.
Lesions (not images) are resampled with replacement, B = 5000 resamples, and the
metric is recomputed on all images of the drawn lesions. For a difference
between two models the same lesion resample is used for both (paired).
Implemented with per-lesion sufficient statistics (confusion counts, calibration
bin counts) and multinomial resample weights, so it is exact and fast.

McNemar: discordant counts b (A right, B wrong) and c (A wrong, B right);
exact two-sided binomial test on the image level, and the clustered McNemar
statistic of Durkalski et al. (Stat Med 2003; 22:2417) that accounts for
several images per lesion:  X2 = (sum_k (b_k - c_k))^2 / sum_k (b_k - c_k)^2 ~ chi2(1).
Multiplicity: Holm-Bonferroni (and Benjamini-Hochberg reported alongside).
"""
import numpy as np
from scipy import stats as st
from scipy.optimize import minimize_scalar

K = 7
B_DEFAULT = 5000


# ---------------------------------------------------------------- metrics
def metrics_from_cm(cm):
    """cm: (..., K, K) rows = true, cols = predicted. Returns dict of arrays."""
    cm = cm.astype(float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    sup = cm.sum(-1)
    pred = cm.sum(-2)
    n = cm.sum((-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(sup > 0, tp / sup, np.nan)
        prec = np.where(pred > 0, tp / pred, 0.0)
        f1 = np.where(prec + np.nan_to_num(rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
        spec = (n[..., None] - sup - pred + tp) / (n[..., None] - sup)
    present = sup > 0
    wf1 = (f1 * sup).sum(-1) / n
    return dict(acc=tp.sum(-1) / n, bal_acc=np.nanmean(rec, -1), macro_precision=(prec * present).sum(-1) / present.sum(-1),
                macro_recall=np.nanmean(rec, -1), macro_f1=(np.nan_to_num(f1) * present).sum(-1) / present.sum(-1),
                weighted_f1=wf1, precision=prec, recall=rec, f1=f1, specificity=spec)


def confusion(y, yhat):
    cm = np.zeros((K, K), int)
    np.add.at(cm, (y, yhat), 1)
    return cm


def calib_stats(p, y, n_bins=15):
    conf = p.max(1)
    pred = p.argmax(1)
    corr = (pred == y).astype(float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return conf, corr, bins


def ece(p, y, n_bins=15):
    conf, corr, bins = calib_stats(p, y, n_bins)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return e


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def brier(p, y):
    oh = np.eye(K)[y]
    return float(((p - oh) ** 2).sum(1).mean())


def auroc_macro(p, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(K))))


def all_metrics(p, y):
    m = metrics_from_cm(confusion(y, p.argmax(1)))
    out = {k: float(v) for k, v in m.items() if np.ndim(v) == 0}
    out.update(ece=ece(p, y), nll=nll(p, y), brier=brier(p, y), auroc_macro=auroc_macro(p, y))
    return out


# ---------------------------------------------------------- temperature scaling
def fit_temperature(p_val, y_val):
    logp = np.log(np.clip(p_val, 1e-12, 1))

    def f(logT):
        z = logp / np.exp(logT)
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y_val)), y_val].mean()

    return float(np.exp(minimize_scalar(f, bounds=(-3, 3), method="bounded").x))


def apply_temperature(p, T):
    z = np.log(np.clip(p, 1e-12, 1)) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ------------------------------------------------------------ clustered bootstrap
class ClusterBoot:
    """Holds a fixed set of lesion resamples so that every model / metric /
    difference is evaluated on the same resamples (paired)."""

    def __init__(self, clusters, B=B_DEFAULT, seed=2026):
        self.uc, self.inv = np.unique(np.asarray(clusters), return_inverse=True)
        L = len(self.uc)
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(L, np.full(L, 1.0 / L), size=B).astype(np.float32)  # (B, L)
        self.B, self.L = B, L

    def _lesion_cm(self, y, yhat):
        a = np.zeros((self.L, K * K), np.float32)
        np.add.at(a, (self.inv, y * K + yhat), 1)
        return a

    def cm_dist(self, y, yhat):
        return (self.W @ self._lesion_cm(y, yhat)).reshape(self.B, K, K)

    def metric_dist(self, y, p):
        """Bootstrap distributions of CM metrics + ECE/NLL/Brier."""
        out = metrics_from_cm(self.cm_dist(y, p.argmax(1)))
        conf, corr, bins = calib_stats(p, y)
        nb = 15
        cnt = np.zeros((self.L, nb), np.float32)
        sc = np.zeros((self.L, nb), np.float32)
        sa = np.zeros((self.L, nb), np.float32)
        np.add.at(cnt, (self.inv, bins), 1)
        np.add.at(sc, (self.inv, bins), conf)
        np.add.at(sa, (self.inv, bins), corr)
        C, SC, SA = self.W @ cnt, self.W @ sc, self.W @ sa
        N = C.sum(1, keepdims=True)
        out["ece"] = (np.abs(SA - SC) / N).sum(1)
        per = np.zeros((self.L, 2), np.float32)
        np.add.at(per[:, 0], self.inv, -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)))
        np.add.at(per[:, 1], self.inv, ((p - np.eye(K)[y]) ** 2).sum(1))
        n_img = self.W @ np.bincount(self.inv, minlength=self.L).astype(np.float32)
        out["nll"] = (self.W @ per[:, 0]) / n_img
        out["brier"] = (self.W @ per[:, 1]) / n_img
        return out


def ci(dist, alpha=0.05):
    d = np.asarray(dist, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(d, 100 * alpha / 2)), float(np.percentile(d, 100 * (1 - alpha / 2))))


# ---------------------------------------------------------------- McNemar
def mcnemar(y, pa, pb, clusters):
    ca, cb = pa.argmax(1) == y, pb.argmax(1) == y
    b = int((ca & ~cb).sum())
    c = int((~ca & cb).sum())
    p_exact = float(st.binomtest(b, b + c, 0.5).pvalue) if b + c > 0 else 1.0
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    d = np.bincount(inv, weights=(ca & ~cb).astype(float)) - np.bincount(inv, weights=(~ca & cb).astype(float))
    denom = (d ** 2).sum()
    x2 = float(d.sum() ** 2 / denom) if denom > 0 else 0.0
    p_clu = float(st.chi2.sf(x2, 1)) if denom > 0 else 1.0
    return dict(b_A_right_B_wrong=b, c_A_wrong_B_right=c, n_discordant=b + c,
                p_mcnemar_exact=p_exact, chi2_clustered=x2, p_mcnemar_clustered=p_clu)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def bh(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    prev = 1.0
    for rank, idx in zip(range(m, 0, -1), order):
        prev = min(prev, p[idx] * m / rank)
        adj[idx] = prev
    return adj


def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def fold_paired(a, b):
    """Fold-level paired differences (a - b) across the 5 CV models."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    m, s = d.mean(), d.std(ddof=1) if n > 1 else 0.0
    half = st.t.ppf(0.975, n - 1) * s / np.sqrt(n) if n > 1 else np.nan
    p = float(st.ttest_rel(a, b).pvalue) if n > 1 and s > 0 else np.nan
    return dict(fold_diff_mean=float(m), fold_diff_sd=float(s), fold_diff_ci_lo=float(m - half),
                fold_diff_ci_hi=float(m + half), p_fold_paired_t=p, n_folds=n)
