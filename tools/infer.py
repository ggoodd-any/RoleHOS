import argparse
from pathlib import Path
import cv2
from rolehos.runtime import load_model, predict, overlay


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("--output", type=Path, default=Path("outputs/inference"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--variant", default="full")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    model, _ = load_model(a.variant, a.seed, a.device)
    bgr = cv2.imread(str(a.image))
    if bgr is None:
        raise ValueError("Unreadable image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = predict(model, rgb)
    ov, col = overlay(rgb, mask)
    a.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.output / "mask.png"), mask)
    cv2.imwrite(str(a.output / "overlay.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(a.output / "color_mask.png"), cv2.cvtColor(col, cv2.COLOR_RGB2BGR))
    print(a.output)


if __name__ == "__main__":
    main()
