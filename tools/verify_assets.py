import json
import argparse
from rolehos import ROOT
from rolehos.runtime import read_manifest, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    count = 0
    skipped = 0
    for r in read_manifest():
        if not (ROOT / r["checkpoint"]).is_file() and not args.require_all:
            skipped += 1
            continue
        for file, expected in [
            (r["checkpoint"], r["sha256"]),
            (r["config"], r["config_sha256"]),
        ]:
            if sha256(ROOT / file) != expected:
                raise ValueError("Hash mismatch: " + file)
        if sha256(ROOT / r["evaluation_log"]) != r["evaluation_sha256"]:
            raise ValueError("Evaluation summary hash mismatch")
        for log in r["logs"]:
            if sha256(ROOT / log["file"]) != log["sha256"]:
                raise ValueError(log["file"])
        count += 1
    for r in json.loads((ROOT / "examples/manifest.json").read_text()):
        for k in ["image", "label"]:
            if (ROOT / r[k]).exists() and sha256(ROOT / r[k]) != r[k + "_sha256"]:
                raise ValueError(r[k])
    if not count:
        raise FileNotFoundError("No supplied weights installed")
    print(f"Uninstalled configurations skipped: {skipped}")
    print(
        f"PASS: {count} checkpoint/config pairs and their archived logs; present examples verified"
    )


if __name__ == "__main__":
    main()
