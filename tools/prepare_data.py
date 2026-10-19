"""Minimal image/label preparation matching the research converter; no depth/contact data."""

import argparse, json, re, shutil
from pathlib import Path
import numpy as np
from PIL import Image
from rolehos import ROOT
from rolehos.runtime import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Extracted InterFormer datasets/ directory",
    )
    p.add_argument("--output", type=Path)
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    if not a.verify_only and a.output is None:
        p.error("--output is required unless --verify-only")
    manifest = json.loads((ROOT / "docs/data_manifest.json").read_text())
    normalize = lambda s: re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")
    for split, rows in manifest.items():
        labels = {
            normalize(x.stem): x for x in (a.source / split / "label").glob("*.png")
        }
        for r in rows:
            image = a.source / split / "image" / (r["id"] + ".jpg")
            label = labels.get(normalize(r["id"]))
            if label is None or not image.is_file():
                raise FileNotFoundError(split + "/" + r["id"])
            if sha256(image) != r["image_sha256"]:
                raise ValueError("Input image differs: " + image.name)
            mask = np.asarray(Image.open(label), dtype=np.uint8).copy()
            mask[~np.isin(mask, [0, 1, 2, 3, 4, 5, 255])] = 255
            import hashlib

            if hashlib.sha256(mask.tobytes()).hexdigest() != r["label_pixels_sha256"]:
                raise ValueError("Converted label differs: " + label.name)
            if not a.verify_only:
                im = a.output / split / "image" / image.name
                gt = a.output / split / "label" / (r["id"] + ".png")
                if im.exists() or gt.exists():
                    raise FileExistsError("Use a new empty output directory")
                im.parent.mkdir(parents=True, exist_ok=True)
                gt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, im)
                Image.fromarray(mask).save(gt)
        print(
            split,
            len(rows),
            "verified" + (" and prepared" if not a.verify_only else ""),
            flush=True,
        )


if __name__ == "__main__":
    main()
