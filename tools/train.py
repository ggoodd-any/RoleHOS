import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, json
from pathlib import Path
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from rolehos import ROOT
from rolehos.runtime import deterministic, record_for, sha256


def main():
    p = argparse.ArgumentParser(
        description="Train final-iteration RoleHOS or a paper ablation; no automatic testing."
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--variant", default="full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--work-dir", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument(
        "--smoke-iters", type=int, default=0, help="Development check, not a paper run"
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    a = p.parse_args()
    if not a.pretrained.is_file():
        raise FileNotFoundError(
            "Obtain the official DINOv3 ConvNeXt-Tiny LVD-1689M checkpoint first"
        )
    if (
        sha256(a.pretrained)
        != json.loads((ROOT / "docs/initialization.json").read_text())["sha256"]
    ):
        raise ValueError(
            "Wrong DINOv3 initialization hash; see docs/initialization.json"
        )
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    record = record_for(a.variant, a.seed)
    cfg = Config.fromfile(str(ROOT / record["config"]))
    deterministic(a.seed, strict=False)
    torch.backends.cudnn.allow_tf32 = (
        True  # Original PyTorch training default; inference disables TF32.
    )
    # Historical training enabled cuDNN deterministic mode (not global strict deterministic algorithms).
    cfg.model.backbone.checkpoint = str(a.pretrained.resolve())
    cfg.train_dataloader.dataset.data_root = str(a.data_root.resolve())
    cfg.work_dir = str(
        (a.work_dir or Path("outputs") / a.variant / f"seed{a.seed}").resolve()
    )
    if (
        Path(cfg.work_dir).exists()
        and list(Path(cfg.work_dir).glob("*.pth"))
        and not a.resume
    ):
        raise FileExistsError(
            "Existing checkpoints: choose another work-dir or explicitly --resume"
        )
    cfg.randomness = dict(seed=a.seed, deterministic=True)
    cfg.env_cfg = dict(
        cudnn_benchmark=False, mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0)
    )
    if a.smoke_iters:
        cfg.train_cfg.max_iters = a.smoke_iters
        cfg.train_dataloader.batch_size = 1
        cfg.train_dataloader.num_workers = 0
        cfg.train_dataloader.persistent_workers = False
        cfg.default_hooks.checkpoint.interval = a.smoke_iters
    if a.resume:
        cfg.resume = True
        cfg.load_from = str(a.resume.resolve())
    # MMEngine 0.10 checkpoints contain optimizer metadata; only locally trusted training files.
    original_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = trusted_load
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.work_dir) / "run_provenance.json").write_text(
        json.dumps(
            {
                "variant": a.variant,
                "seed": a.seed,
                "pretrained_sha256": sha256(a.pretrained),
                "smoke": bool(a.smoke_iters),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "strict_deterministic_algorithms": False,
                "cudnn_deterministic": True,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            },
            indent=2,
        )
    )
    Runner.from_cfg(cfg).train()


if __name__ == "__main__":
    main()
