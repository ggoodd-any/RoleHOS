import argparse, json, shutil
from pathlib import Path
from rolehos import ROOT
from rolehos.runtime import sha256

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    a = p.parse_args()
    for r in json.loads((ROOT / "examples/manifest.json").read_text()):
        for kind, suffix in [("image", ".jpg"), ("label", ".png")]:
            source = a.data_root / r["split"] / kind / (r["sample_id"] + suffix)
            if sha256(source) != r[kind + "_sha256"]:
                raise ValueError("Wrong sample " + str(source))
            dest = ROOT / r[kind]
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(source, dest)
    print("Imported six exact curated examples from your licensed dataset")
