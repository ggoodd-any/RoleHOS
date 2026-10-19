# RoleHOS

Reproducible research code for **RoleHOS: Efficient Egocentric Hand–Object Segmentation**.

[中文说明](README_zh.md) · [Ablations](docs/ABLATIONS.md) · [Data](docs/DATA.md) · [Licenses](NOTICE.md)

This package contains the frozen ConvNeXt-Tiny / DINOv3 RoleHOS model, original training losses and optimizer schedule, inference/evaluation tools, 36 checkpoint/config pairs, archived logs, and a local Gradio demo. The code is hosted on GitHub; pretrained checkpoints are distributed separately through Google Drive.

## Code and pretrained weights

- [GitHub repository](https://github.com/ggoodd-any/RoleHOS)
- [Google Drive: all 12 checkpoint archives](https://drive.google.com/drive/folders/1qylFVDv40EpEyo6JSRhAhpcej5M-VG0h)
- [Archive SHA-256 checksums](weights/downloads.json)

For inference or the Gradio demo, download **[RoleHOS_weights_full.zip](https://drive.google.com/file/d/1idTg-AdrRZ0OBhg0znCXJMIF6TI-zfQS/view)** (seeds 0, 1 and 2). For ablations, download the corresponding named ZIPs listed in [the ablation guide](docs/ABLATIONS.md). Extract each ZIP into the repository root, preserving its `weights/<configuration>/` layout:

```bash
git clone https://github.com/ggoodd-any/RoleHOS.git
cd RoleHOS
unzip /path/to/RoleHOS_weights_full.zip 'weights/*/*.pth' -d .
```

The archives include frozen checkpoint manifests and required third-party license notices. The command above extracts model files only, preserving the repository's current documentation and manifests. Run `python -m tools.verify_assets` after environment setup; `--require-all` checks all 36 checkpoints. The public repository does not redistribute dataset images: upload your own RGB image in Gradio, or restore the six curated examples from your licensed dataset using `python -m tools.import_examples --data-root /path/to/egohos`.

## Environment

Reference research runs used Linux, a single NVIDIA RTX 5090, Python 3.10.20, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, CUDA 12.8, NumPy 1.26.4, MMEngine 0.10.7 and MMCV 2.2.0. The lightweight release uses MMCV-lite 2.2.0; it does not require custom CUDA extensions, MMDetection, a separate MMSegmentation install or access to the authors' repository. The Gradio environment uses Pillow 11.3.0; original training used 9.5.0, but train/test RGB resizing here uses OpenCV 4.11.0.86. The new release smoke checks run on CPU to avoid interfering with other GPU jobs; they are not new GPU throughput measurements.

```bash
conda create -n rolehos python=3.10.20 -y
conda activate rolehos
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-demo.txt -c constraints-tested.txt
python -m unittest discover -s tests
python -m tools.verify_assets
```

Run commands from this directory. `scripts/*.sh` also locate this directory automatically; `PYTHON=/path/to/python` can select a specific environment. A CUDA 12.8-compatible driver is needed for the GPU wheel; CPU is supported for demonstration/testing, but full training is intended for a GPU. Pinned versions are in `requirements*.txt`; no root installation is needed.

## Quick reviewer demo

```bash
bash scripts/demo.sh --device cpu
# Or, on an available GPU:
bash scripts/demo.sh --device cuda
```

Open **http://127.0.0.1:7860**. Choose a curated test example or upload a picture, select seed 0/1/2, run segmentation and download a class-ID PNG. The interface shows an RGB overlay, a color mask and the role legend. The server is local-only (`share=False`) and does not upload images. Inference is FP32 PyTorch, not Jetson INT8; browser latency is not a paper speed measurement. Download and extract the full-model weight archive before running the demo. Example image redistribution is separate from code licensing.

```bash
bash scripts/infer.sh path/to/image.jpg --device cpu --output outputs/demo
```

## Train and test with one command

First obtain the prescribed dataset and run the data check; see [data instructions](docs/DATA.md). For training, download the official `dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth` after accepting the [DINOv3 terms](https://github.com/facebookresearch/dinov3). Final RoleHOS weights are not a substitute for this initialization.

```bash
python -m tools.check_data --data-root /path/to/egohos
bash scripts/train.sh --data-root /path/to/egohos --pretrained /path/to/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth --seed 0
bash scripts/test.sh --data-root /path/to/egohos --seed 0 --save-masks --output outputs/full_seed0
# Explicitly evaluate validation separately:
bash scripts/test.sh --data-root /path/to/egohos --splits val --output outputs/full_seed0_val
```

Default training is FP32, 20,000 iterations, batch 8, AdamW lr 6e-5, backbone LR multiplier 0.2, weight decay 0.01, gradient clipping 5; linear warmup 1,000 iterations followed by polynomial decay. Checkpoints are final-20k, not test-selected. Data augmentation is horizontal flipping with LH↔RH and LO↔RO swaps, followed by resize to 448²; TO is unchanged. Input RGB mean/std are `[106.01075,95.40013,87.42854]` / `[64.356636,60.888744,61.41911]`. Exact losses and remaining settings are in each config. A `--smoke-iters 2 --device cpu` training run is a development check with batch 1, not the paper protocol. Training writes text/scalar logs and provenance; testing writes a summary, per-image confusion JSONL and optional masks.

## Reproduce the ablations

```bash
bash scripts/test_ablations.sh --data-root /path/to/egohos
# All 36 runs serially, seeds 0/1/2. To select a subset:
bash scripts/test_ablations.sh --data-root /path/to/egohos --variants full without_sfsc_zero --seeds 0
# One-command ablation training (long-running):
bash scripts/test_ablations.sh --mode train --data-root /path/to/egohos --pretrained /path/to/dinov3_initialization.pth
```

[The mapping table](docs/ABLATIONS.md) distinguishes final zero-context controls from earlier experiments. `--limit 2` is only a smoke evaluation and never reports the full-test Avg. Model loading is strict and supplied checkpoint hashes are verified. Query-to-role readout is preserved for anonymous-query controls.

## Reproducibility and evidence

Seeds are 0, 1 and 2. Python, NumPy, PyTorch and CUDA seeds are set; cuDNN benchmarking is disabled. Inference disables TF32 and enables `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; unsupported deterministic operations fail instead of silently proceeding. Historical training used cuDNN deterministic mode, not global strict deterministic algorithms, and retained the PyTorch cuDNN TF32 default; this is explicitly preserved. Numerical equivalence across GPU architectures/framework versions is not guaranteed. A fresh full run is the canonical training reproduction. `--resume` restores a trusted locally produced optimizer checkpoint using MMEngine; data-order identity with older interrupted research jobs is not claimed.

`weights/manifest.json` includes original and exported SHA-256 hashes; exported state tensors are exactly equal to the originals. `logs/reference/` contains sanitized historical training/scalar logs and evaluation summaries. `logs/release_checks/` contains this package's validation, separate from original experiments. Reference results are in `docs/expected_metrics.csv`. The seed-mean full-model Avg is about **72.54%**; individual seed values should not be confused with that mean or quantized device results.

## Directory guide and release

- `rolehos/`, `third_party/`: original model/ablation implementations and minimal dependencies.
- `configs/`: one portable resolved config per checkpoint.
- `weights/`: 36 human-readable model-only checkpoint files; see its manifest/README.
- `tools/`, `scripts/`, `app.py`: training, evaluation, inference, verification and Gradio.
- `examples/`: six curated local test samples with selection criteria and provenance; full benchmark metrics are unchanged.
- `logs/`, `docs/`: archived evidence and reproducibility details.

`python -m tools.package_release --output /path/to/bundles` builds a code archive and per-configuration weight archives for later manual upload. It does not publish anything. Original release tooling is MIT; vendored code, DINOv3 weights and dataset samples retain their own terms, detailed in [NOTICE.md](NOTICE.md). Download links and extraction instructions are above.
