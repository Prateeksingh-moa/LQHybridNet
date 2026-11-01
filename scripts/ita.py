"""Skin-tone proxy for subgroup analysis (Reviewer 1, point 4).

HAM10000 has no Fitzpatrick labels. Following Kinyanjui et al. (MICCAI 2020) we
estimate the Individual Typology Angle ITA = atan((L* - 50) / b*) * 180/pi from
healthy-skin pixels: outside the dilated expert lesion mask, excluding the dark
dermoscope vignette (L* < 25) and specular highlights (L* > 95). Categories
(Del Bino & Bernerd 2013): very light > 55, light 41-55, intermediate 28-41,
tan 10-28, brown -30-10, dark < -30. Dermoscopic ITA is affected by illumination
and is only a coarse proxy, which the manuscript must state.
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.color import rgb2lab


def category(t):
    if np.isnan(t):
        return "undetermined"
    for lo, name in [(55, "very_light"), (41, "light"), (28, "intermediate"), (10, "tan"), (-30, "brown")]:
        if t > lo:
            return name
    return "dark"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    sp = pd.read_csv(f"{a.data}/splits.csv")
    rows = []
    for iid in sp.image_id:
        lab = rgb2lab(np.asarray(Image.open(f"{a.data}/img/{iid}.jpg").convert("RGB")) / 255.0)
        m = np.asarray(Image.open(f"{a.data}/mask/{iid}.png")) > 127
        skin = ~binary_dilation(m, iterations=10) & (lab[..., 0] > 25) & (lab[..., 0] < 95)
        if skin.sum() < 500:
            rows.append((iid, np.nan, int(skin.sum())))
            continue
        L, b = np.median(lab[..., 0][skin]), np.median(lab[..., 2][skin])
        rows.append((iid, float(np.degrees(np.arctan2(L - 50, b))), int(skin.sum())))
    df = pd.DataFrame(rows, columns=["image_id", "ita", "n_skin_pixels"])
    df["ita_category"] = df.ita.map(category)
    df.to_csv(f"{a.data}/ita.csv", index=False)
    print(df.ita_category.value_counts())


if __name__ == "__main__":
    main()
