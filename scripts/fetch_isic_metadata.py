"""Fetch ISIC Archive metadata (archive lesion_id, sex, age, site, file size) for
every HAM10000 image and every ISIC 2018 Task 3 test image.

The archive assigns its own lesion identifiers (IL_*) across all collections, so
matching IL_* ids between the HAM10000 training corpus and the Task 3 test set is
the metadata-based lesion-level overlap check requested by the editor/Reviewer 2.

Usage: python -I fetch_isic_metadata.py <ids.txt> <out.csv>
"""
import csv
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://api.isic-archive.com/api/v2/images/{}/"


def fetch(isic_id):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(API.format(isic_id), timeout=60) as r:
                d = json.load(r)
            c = d.get("metadata", {}).get("clinical", {})
            return {
                "isic_id": isic_id,
                "archive_lesion_id": c.get("lesion_id", ""),
                "sex": c.get("sex", ""),
                "age_approx": c.get("age_approx", ""),
                "anatom_site": c.get("anatom_site_general", c.get("anatom_site_1", "")),
                "diagnosis_3": c.get("diagnosis_3", ""),
                "diagnosis_confirm_type": c.get("diagnosis_confirm_type", ""),
                "attribution": d.get("attribution", ""),
                "license": d.get("copyright_license", ""),
                "full_size": d.get("files", {}).get("full", {}).get("size", ""),
                "status": "ok",
            }
        except Exception as e:  # retry on transient network errors
            err = repr(e)
            time.sleep(2 * (attempt + 1))
    return {"isic_id": isic_id, "status": "error:" + err}


def main():
    ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    fields = ["isic_id", "archive_lesion_id", "sex", "age_approx", "anatom_site",
              "diagnosis_3", "diagnosis_confirm_type", "attribution", "license",
              "full_size", "status"]
    with ThreadPoolExecutor(24) as ex, open(sys.argv[2], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(ex.map(fetch, ids)):
            w.writerow(row)
            if i % 1000 == 0:
                print(i, flush=True)


if __name__ == "__main__":
    main()
