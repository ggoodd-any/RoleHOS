import argparse, json, hashlib
import numpy as np
from PIL import Image
from pathlib import Path
from rolehos import ROOT
from rolehos.runtime import sha256


def check(root):
    manifest = json.loads((ROOT / "docs/data_manifest.json").read_text())
    for split, rows in manifest.items():
        for kind, suffix in [("image", ".jpg"), ("label", ".png")]:
            found = {p.stem for p in (root / split / kind).glob("*" + suffix)}
            if found != {r["id"] for r in rows}:
                raise ValueError(f"{split}/{kind}: sample identities differ")
        for r in rows:
            for kind, suffix in [("image", ".jpg"), ("label", ".png")]:
                path = root / split / kind / (r["id"] + suffix)
                actual = (
                    sha256(path)
                    if kind == "image"
                    else hashlib.sha256(
                        np.asarray(Image.open(path), dtype=np.uint8).tobytes()
                    ).hexdigest()
                )
                expected = (
                    r["image_sha256"] if kind == "image" else r["label_pixels_sha256"]
                )
                if actual != expected:
                    raise ValueError(f'{split}/{r["id"]}: {kind} differs')
        print(split, len(rows), "PASS", flush=True)
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    a = p.parse_args()
    check(a.data_root)
