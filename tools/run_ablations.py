import argparse, subprocess, sys
from pathlib import Path
from rolehos.runtime import read_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--mode", choices=["test", "train"], default="test")
    p.add_argument("--pretrained")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--variants", nargs="+")
    p.add_argument("--output", default="outputs/ablations")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    for r in read_manifest():
        if r["seed"] not in a.seeds or (a.variants and r["variant"] not in a.variants):
            continue
        dest = str(Path(a.output) / r["variant"] / f"seed{r['seed']}")
        cmd = [
            sys.executable,
            "-m",
            "tools.evaluate" if a.mode == "test" else "tools.train",
            "--data-root",
            a.data_root,
            "--variant",
            r["variant"],
            "--seed",
            str(r["seed"]),
            "--device",
            a.device,
        ]
        if a.mode == "test":
            cmd += ["--output", dest, "--limit", str(a.limit)]
        else:
            if not a.pretrained:
                raise ValueError("--pretrained is required for training")
            cmd += ["--pretrained", a.pretrained, "--work-dir", dest]
        print("Running", r["variant"], r["seed"], flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
