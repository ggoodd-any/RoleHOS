# Pretrained weight bundle

Folder: `<configuration>/rolehos_<configuration>_seed{0,1,2}_iter20000.pth`.

`full/` contains the three published RoleHOS seeds. Other folders contain the final manuscript ablation checkpoints. Each file contains only the model state dictionary, safely readable with `torch.load(..., weights_only=True)`. Optimizer state, training paths and pickle metadata were removed. Every exported tensor was compared exactly with its original checkpoint; `manifest.json` records original/export SHA-256, count, config, original evaluation and log identity.

These files support inference/evaluation and initialization for further work, not an exact optimizer resume. For a new 20,000-iteration reproduction use the official DINOv3 ConvNeXt-Tiny LVD-1689M initialization, not a final RoleHOS checkpoint. New training writes full MMEngine checkpoints to `outputs/`.

For Google Drive, keep this directory layout. `python -m tools.package_release --output <directory>` produces one archive per configuration with its manifest and required licenses; no upload is performed. [Download checkpoint archives from Google Drive](https://drive.google.com/drive/folders/1qylFVDv40EpEyo6JSRhAhpcej5M-VG0h). Extract at the repository root. Archive checksums are in `downloads.json`. See `../NOTICE.md` and `../licenses/DINOv3.md` before distribution.
