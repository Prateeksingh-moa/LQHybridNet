"""Build the lesion-level split and the image cache.

Inputs (all public):
  raw/HAM10000_images_part_{1,2}.zip, raw/HAM10000_segmentations_lesion_tschandl.zip
      (HAM10000 v4 files, mirror: huggingface.co/datasets/oscar2525mv/ham10000-segmentation)
  raw/HAM10000_metadata.csv (Dataverse metadata, mirror: marmal88/skin_cancer)
  raw/official/ISIC2018_Task3_Training_LesionGroupings.csv,
  raw/official/ISIC2018_Task3_Training_GroundTruth/*.csv,
  raw/official/ISIC2018_Task3_Test_GroundTruth/*.csv
      (official ISIC files, https://isic-archive.s3.amazonaws.com/challenges/2018/)
  raw/zs389_test.parquet (ISIC 2018 Task 3 test images; byte sizes are checked
      against the ISIC Archive API "full" file size)

Outputs (in --out):
  splits.csv         image_id, lesion_id, dx, label, split (dev/test), fold (0-4 for dev)
  isic_test.csv      image_id, dx, label (official ground truth)
  img/<id>.jpg       256x256 cache for HAM and ISIC test
  mask/<id>.png      256x256 HAM lesion masks (for quantitative Grad-CAM)
"""
import argparse
import hashlib
import io
import os
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]  # label index 0..6 (manuscript order)
SIZE = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--archive_meta", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/img", exist_ok=True)
    os.makedirs(f"{a.out}/mask", exist_ok=True)
    off = f"{a.raw}/official"

    # ---- labels + lesion ids (official ISIC 2018 Task 3 training files = HAM10000)
    groups = pd.read_csv(f"{off}/ISIC2018_Task3_Training_LesionGroupings.csv")
    gt = pd.read_csv(f"{off}/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")
    onehot = gt.drop(columns="image").values
    gt["dx"] = [c.lower() for c in gt.columns[1:][onehot.argmax(1)]]
    df = groups.merge(gt[["image", "dx"]], on="image").rename(columns={"image": "image_id"})
    meta = pd.read_csv(f"{a.raw}/HAM10000_metadata.csv")
    df = df.merge(meta[["image_id", "lesion_id", "dx", "dx_type", "age", "sex", "localization"]]
                  .rename(columns={"lesion_id": "lesion_id_dv", "dx": "dx_dv"}), on="image_id", how="left")
    assert (df.lesion_id == df.lesion_id_dv).all(), "Dataverse and ISIC lesion ids disagree"
    assert (df.dx == df.dx_dv).all(), "Dataverse and ISIC labels disagree"
    df = df.drop(columns=["lesion_id_dv", "dx_dv"])
    am_all = pd.read_csv(a.archive_meta).set_index("isic_id")
    df["center"] = df.image_id.map(am_all.attribution)  # contributing centre (ISIC Archive)
    df["archive_lesion_id"] = df.image_id.map(am_all.archive_lesion_id)
    df["label"] = df.dx.map(CLASSES.index)
    print("images", len(df), "lesions", df.lesion_id.nunique())

    # ---- lesion-level stratified split: 10% of lesions -> test, rest -> 5 CV folds
    les = df.groupby("lesion_id").agg(dx=("label", "first")).reset_index()
    assert (df.groupby("lesion_id").label.nunique() == 1).all()
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=a.seed)
    _, test_idx = next(sgkf.split(df, df.label, groups=df.lesion_id))
    test_lesions = set(df.lesion_id.iloc[test_idx])
    df["split"] = np.where(df.lesion_id.isin(test_lesions), "test", "dev")
    df["fold"] = -1
    dev = df[df.split == "dev"]
    sgkf5 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=a.seed)
    for k, (_, vi) in enumerate(sgkf5.split(dev, dev.label, groups=dev.lesion_id)):
        df.loc[dev.index[vi], "fold"] = k
    # leakage checks
    assert not (set(df[df.split == "test"].lesion_id) & set(df[df.split == "dev"].lesion_id))
    for k in range(5):
        assert not (set(df[df.fold == k].lesion_id) & set(df[(df.fold != k) & (df.split == "dev")].lesion_id))
    df.to_csv(f"{a.out}/splits.csv", index=False)
    print(df.groupby(["split", "dx"]).size().unstack(0))
    print("test lesions", df[df.split == "test"].lesion_id.nunique(), "dev lesions", dev.lesion_id.nunique())
    print(df[df.split == "dev"].groupby(["fold", "dx"]).size().unstack(0))

    # ---- HAM images + masks
    for zname in ["HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"]:
        with zipfile.ZipFile(f"{a.raw}/{zname}") as z:
            for n in z.namelist():
                if n.endswith(".jpg") and "__MACOSX" not in n and not os.path.basename(n).startswith("._"):
                    iid = os.path.basename(n)[:-4]
                    im = Image.open(io.BytesIO(z.read(n))).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
                    im.save(f"{a.out}/img/{iid}.jpg", quality=95)
    with zipfile.ZipFile(f"{a.raw}/HAM10000_segmentations_lesion_tschandl.zip") as z:
        for n in z.namelist():
            if n.endswith(".png") and "__MACOSX" not in n and not os.path.basename(n).startswith("._"):
                iid = os.path.basename(n).replace("_segmentation.png", "")
                m = Image.open(io.BytesIO(z.read(n))).convert("L").resize((SIZE, SIZE), Image.NEAREST)
                m.save(f"{a.out}/mask/{iid}.png")
    missing = [i for i in df.image_id if not os.path.exists(f"{a.out}/img/{i}.jpg")]
    assert not missing, missing[:5]

    # ---- ISIC 2018 Task 3 test
    tgt = pd.read_csv(f"{off}/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv")
    oh = tgt.drop(columns="image").values
    tgt["dx"] = [c.lower() for c in tgt.columns[1:][oh.argmax(1)]]
    tgt = tgt.rename(columns={"image": "image_id"})[["image_id", "dx"]]
    tgt["label"] = tgt.dx.map(CLASSES.index)
    am = pd.read_csv(a.archive_meta).set_index("isic_id")
    rows = pq.read_table(f"{a.raw}/zs389_test.parquet").to_pylist()
    sizes_ok, md5 = 0, {}
    for r in rows:
        iid = os.path.basename(r["filename"]["path"])[:-4]
        b = r["filename"]["bytes"]
        if iid in am.index and int(am.loc[iid, "full_size"]) == len(b):
            sizes_ok += 1
        md5[iid] = hashlib.md5(b).hexdigest()
        Image.open(io.BytesIO(b)).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC).save(f"{a.out}/img/{iid}.jpg", quality=95)
    print(f"ISIC test: {len(tgt)} labelled, {len(rows)} images in mirror, {sizes_ok} byte-size matches with ISIC Archive")
    tgt["md5_mirror"] = tgt.image_id.map(md5)
    assert tgt.md5_mirror.notna().all()
    tgt = tgt.merge(am[["archive_lesion_id", "sex", "age_approx", "anatom_site", "attribution"]],
                    left_on="image_id", right_index=True, how="left")
    tgt.to_csv(f"{a.out}/isic_test.csv", index=False)
    print(tgt.dx.value_counts())


if __name__ == "__main__":
    main()
