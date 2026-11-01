# LQ-HybridNet

Code for *LQ-HybridNet*: MobileNetV2 with Quantum-Inspired Channel Gating (QCG)
and a 4-qubit variational quantum circuit (VQC) branch for dermoscopic
skin-lesion classification (HAM10000, ISIC 2018 Task 3).

## Contents
```
src/vqc.py        4-qubit VQC (RY encoding, 2 layers of trainable RY + CNOT ring, <Z> readout):
                  exact batched statevector in PyTorch + PennyLane reference implementation
src/models.py     LQ-HybridNet, QCG / SE / ECA / CBAM gates, VQC branch and classical
                  replacements (parameter-matched MLP, residual MLP, Fourier features,
                  polynomial expansion, trigonometric mapping)
src/configs.py    experiment configurations
src/train.py      training of one configuration on one cross-validation fold
src/stats.py      metrics, lesion-clustered bootstrap CIs, McNemar (exact and clustered),
                  Holm / Benjamini-Hochberg, ECE, temperature scaling
src/analyze.py    result tables from saved predictions
scripts/          data preparation, ISIC metadata retrieval, de-duplication, calibration,
                  latency profiling, Grad-CAM evaluation
data_manifest/    split identifiers and ISIC Archive metadata
```

## Data
Images are not redistributed. The images are HAM10000 (Tschandl et al.,
doi:10.7910/DVN/DBW86T) and the ISIC 2018 Task 3 test set. Labels and lesion
groupings come from the official ISIC 2018 Task 3 files at
`https://isic-archive.s3.amazonaws.com/challenges/2018/`. The ISIC Archive
lesion identifiers come from `https://api.isic-archive.com/api/v2/images/<isic_id>/`
(`scripts/fetch_isic_metadata.py`).

`data_manifest/splits.csv` gives every HAM10000 image's lesion ID, its split
(`dev` / `test`) and its cross-validation fold (0–4). The split is lesion-level:
`StratifiedGroupKFold` with seed 42 and groups = `lesion_id`. 10% of lesions
(747 lesions, 1002 images) are held out for testing. The remaining 9013 images
are divided into 5 lesion-disjoint folds. `data_manifest/isic_test.csv` lists
the 1512 labelled ISIC 2018 Task 3 test images.

## Usage
```bash
pip install -r requirements.txt
python scripts/fetch_isic_metadata.py data_manifest/ids_all.txt data_manifest/isic_archive_metadata.csv
python scripts/prepare_data.py --raw RAW_DIR --out data \
       --archive_meta data_manifest/isic_archive_metadata.csv
python src/vqc.py                                # checks PyTorch VQC == PennyLane VQC
python src/train.py --config lq --fold 0 --lr 3e-4 --data data --out results/runs/lq/fold0
python src/analyze.py --results results --data data
```
`--config` takes any key of `CONFIGS` in `src/configs.py` (e.g. `lq`, `qcg_mlp`,
`qcg_trig`, `qcg_fourier`, `qcg_poly`, `qcg_resmlp`, `eca_vqc`).
