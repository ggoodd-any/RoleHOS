"""Prepare local archives only; never upload or infer redistribution permission."""

import argparse, zipfile, json
from pathlib import Path
from rolehos import ROOT
from rolehos.runtime import read_manifest, sha256


def pack(path, files):
    if path.exists():
        raise FileExistsError(path)
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as z:
        for p in sorted(set(files)):
            z.write(p, p.relative_to(ROOT))
    return {"archive": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    excluded = {"outputs", "data", "__pycache__", ".venv", ".git"}
    files = [
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and not excluded.intersection(p.relative_to(ROOT).parts)
        and p.suffix not in (".pth", ".pyc", ".zip")
        and not (
            p.relative_to(ROOT).parts[:2]
            in [("examples", "images"), ("examples", "labels")]
        )
    ]
    receipt = [pack(a.output / "RoleHOS_code_review.zip", files)]
    for v in dict.fromkeys(r["variant"] for r in read_manifest()):
        files = [ROOT / r["checkpoint"] for r in read_manifest() if r["variant"] == v]
        files += [
            ROOT / "weights/manifest.json",
            ROOT / "weights/README.md",
            ROOT / "NOTICE.md",
            *list((ROOT / "licenses").glob("*")),
        ]
        receipt.append(pack(a.output / f"RoleHOS_weights_{v}.zip", files))
    (a.output / "SHA256SUMS.json").write_text(json.dumps(receipt, indent=2))
    print(a.output)


if __name__ == "__main__":
    main()
