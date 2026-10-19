import argparse, json, time, platform
from pathlib import Path
import cv2, numpy as np, torch
from rolehos.runtime import load_model, predict, confusion, metrics, sha256


def main():
    p = argparse.ArgumentParser(
        description="Exclusive foreground mIoU on ID/OOD/Mini; validation is separate."
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--variant", default="full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint")
    p.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    p.add_argument(
        "--splits", nargs="+", default=["test_indomain", "test_outdomain", "minihoi4d"]
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Smoke subset only; results explicitly marked partial",
    )
    p.add_argument("--save-masks", action="store_true")
    a = p.parse_args()
    expected = {
        "test_indomain": 1126,
        "test_outdomain": 500,
        "minihoi4d": 1095,
        "val": 1124,
    }
    a.output.mkdir(parents=True, exist_ok=True)
    model, record = load_model(a.variant, a.seed, a.device, a.checkpoint)
    result = {
        "variant": a.variant,
        "seed": a.seed,
        "partial": bool(a.limit),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "checkpoint_sha256": sha256(a.checkpoint) if a.checkpoint else record["sha256"],
        "splits": {},
    }
    for split in a.splits:
        if split not in expected:
            raise ValueError("Unsupported split " + split)
        images = sorted((a.data_root / split / "image").glob("*.jpg"))
        labels = {x.stem: x for x in (a.data_root / split / "label").glob("*.png")}
        if len(images) != expected[split] or set(x.stem for x in images) != set(labels):
            raise ValueError(
                f"{split}: expected {expected[split]} matched images/labels, got {len(images)}/{len(labels)}"
            )
        pairs = images[: a.limit] if a.limit else images
        cm = np.zeros((6, 6), dtype=np.int64)
        start = time.time()
        out = a.output / split
        out.mkdir(exist_ok=True)
        with open(out / "per_image.jsonl", "w") as log:
            for i, im in enumerate(pairs):
                bgr = cv2.imread(str(im))
                gt = cv2.imread(str(labels[im.stem]), cv2.IMREAD_GRAYSCALE)
                if bgr is None or gt is None:
                    raise ValueError("Unreadable sample " + im.name)
                gt = cv2.resize(gt, (448, 448), interpolation=cv2.INTER_NEAREST)
                pred = predict(model, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                c = confusion(pred, gt)
                cm += c
                log.write(
                    json.dumps(
                        {
                            "id": im.stem,
                            "confusion": c.tolist(),
                            "image_sha256": sha256(im),
                            "label_sha256": sha256(labels[im.stem]),
                        }
                    )
                    + "\n"
                )
                if a.save_masks:
                    cv2.imwrite(str(out / (im.stem + ".png")), pred)
                if (i + 1) % 50 == 0:
                    print(split, i + 1, len(pairs), flush=True)
        result["splits"][split] = dict(
            count=len(pairs), seconds=time.time() - start, **metrics(cm, split)
        )
    main_splits = ["test_indomain", "test_outdomain", "minihoi4d"]
    if all(x in result["splits"] for x in main_splits) and not a.limit:
        result["Avg"] = float(
            np.mean([result["splits"][x]["mIoU"] for x in main_splits])
        )
    (a.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
