"""Latency profiling (Reviewer 1, point 3).

End-to-end inference time per call, measured from a CPU (pinned) image tensor
to CPU logits, so host<->device transfers are included. Batch 1 (single-image
latency) and batch 32 (manuscript setting). Variants:
  base_gpu             MobileNetV2 baseline (classical head), GPU
  lq_pennylane_cpu     LQ-HybridNet, CNN on GPU, VQC on PennyLane default.qubit
                       CPU (manuscript implementation, incl. GPU->CPU->GPU copies)
  lq_torch_gpu         LQ-HybridNet, VQC as exact batched statevector on GPU
                       (numerically identical, see src/vqc.py)
  surrogate_trig_gpu   classical surrogate: VQC replaced by sin/cos (trigonometric) map
  surrogate_mlp_gpu    classical surrogate: VQC replaced by parameter-matched MLP
  base_cpu / lq_torch_cpu / lq_pennylane_cpuonly   everything on CPU (8 threads)
Also reports the VQC branch alone. fp32, eval mode, weights random (latency
does not depend on weight values). 30 warm-up calls, then 200 (batch 1) or 50
(batch 32) timed calls; median, p95, mean and images/s are reported.
"""
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.environ.get("LQH_RESULTS", f"{ROOT}/results")
sys.path.insert(0, f"{ROOT}/src")
from models import HybridNet, VQCBranch  # noqa: E402


def timeit(fn, n_warm, n_iter, cuda):
    for _ in range(n_warm):
        fn()
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return np.array(ts)


def main():
    torch.set_num_threads(8)
    gpu = torch.cuda.is_available()
    rows = []
    variants = [("base_gpu", dict(gate="none", branch="none"), "cuda"),
                ("lq_pennylane_cpu", dict(gate="qcg", branch="vqc", vqc_impl="pennylane"), "cuda"),
                ("lq_torch_gpu", dict(gate="qcg", branch="vqc", vqc_impl="torch"), "cuda"),
                ("surrogate_trig_gpu", dict(gate="qcg", branch="trig"), "cuda"),
                ("surrogate_mlp_gpu", dict(gate="qcg", branch="mlp"), "cuda"),
                ("base_cpu", dict(gate="none", branch="none"), "cpu"),
                ("lq_torch_cpu", dict(gate="qcg", branch="vqc", vqc_impl="torch"), "cpu"),
                ("lq_pennylane_cpuonly", dict(gate="qcg", branch="vqc", vqc_impl="pennylane"), "cpu")]
    for name, kw, devname in variants:
        if devname == "cuda" and not gpu:
            continue
        dev = torch.device(devname)
        m = HybridNet("mobilenetv2", pretrained=False, **kw).to(dev).eval()
        if kw.get("vqc_impl") == "pennylane":
            # keep the VQC parameters on CPU as in the manuscript pipeline
            m.branch.vqc.to("cpu")
        for bs in (1, 32):
            x = torch.randn(bs, 3, 224, 224).pin_memory() if gpu else torch.randn(bs, 3, 224, 224)

            @torch.no_grad()
            def run():
                return m(x.to(dev, non_blocking=True)).cpu()

            ts = timeit(run, 30, 200 if bs == 1 else 50, devname == "cuda")
            rows.append(dict(variant=name, device=devname, batch=bs, median_ms=np.median(ts),
                             p95_ms=np.percentile(ts, 95), mean_ms=ts.mean(), sd_ms=ts.std(),
                             ms_per_image_amortised=np.median(ts) / bs, images_per_s=1000 * bs / np.median(ts)))
            print(rows[-1], flush=True)
    # VQC branch alone (input = 32-d projected feature)
    for impl, devname in [("torch", "cuda"), ("torch", "cpu"), ("pennylane", "cpu")]:
        if devname == "cuda" and not gpu:
            continue
        br = VQCBranch(32, 4, 2, impl).to(devname).eval()
        for bs in (1, 32):
            c = torch.randn(bs, 32, device=devname)

            @torch.no_grad()
            def runb():
                return br(c)

            ts = timeit(runb, 30, 200, devname == "cuda")
            rows.append(dict(variant=f"vqc_branch_only_{impl}", device=devname, batch=bs, median_ms=np.median(ts),
                             p95_ms=np.percentile(ts, 95), mean_ms=ts.mean(), sd_ms=ts.std(),
                             ms_per_image_amortised=np.median(ts) / bs, images_per_s=1000 * bs / np.median(ts)))
            print(rows[-1], flush=True)
    os.makedirs(f"{RES}/latency", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{RES}/latency/latency.csv", index=False)
    import pennylane
    json.dump(dict(gpu=torch.cuda.get_device_name(0) if gpu else None, cpu=platform.processor(),
                   host=platform.node(), torch_threads=torch.get_num_threads(), torch=torch.__version__,
                   pennylane=pennylane.__version__), open(f"{RES}/latency/env.json", "w"), indent=1)


if __name__ == "__main__":
    main()
