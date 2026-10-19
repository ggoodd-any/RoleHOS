# Pretrained weight bundle

Folder: `<configuration>/rolehos_<configuration>_seed{0,1,2}_iter20000.pth`.

`full/` contains the three published RoleHOS seeds. Other folders contain the final manuscript ablation checkpoints. Each file contains only the model state dictionary, safely readable with `torch.load(..., weights_only=True)`. Optimizer state, training paths and pickle metadata were removed. Every exported tensor was compared exactly with its original checkpoint; `manifest.json` records original/export SHA-256, count, config, original evaluation and log identity.

These files support inference/evaluation and initialization for further work, not an exact optimizer resume. For a new 20,000-iteration reproduction use the official DINOv3 ConvNeXt-Tiny LVD-1689M initialization, not a final RoleHOS checkpoint. New training writes full MMEngine checkpoints to `outputs/`.

For Google Drive, keep this directory layout. `python -m tools.package_release --output <directory>` produces one archive per configuration with its manifest and required licenses; no upload is performed. [Download checkpoint archives from Google Drive](https://drive.google.com/drive/folders/1qylFVDv40EpEyo6JSRhAhpcej5M-VG0h). Extract at the repository root. Archive checksums are in `downloads.json`. See `../NOTICE.md` and `../licenses/DINOv3.md` before distribution.

## Download each configuration

| Configuration | Checkpoints | Download |
|---|---|---|
| `full` | seeds 0, 1, 2 | [ZIP](https://drive.google.com/file/d/1idTg-AdrRZ0OBhg0znCXJMIF6TI-zfQS/view) |
| `without_both_zero` | seeds 0, 1, 2 | [ZIP](https://drive.google.com/file/d/1qNyZi1lCB4BjknxtyElSJ52Ul1_3OH5S/view) |
| `without_ricd` | seeds 0, 1, 2 | [ZIP](https://drive.google.com/file/d/1XP4BXERL9IDz5xrfyzRfoAlz22axjYwl/view) |
| `without_sfsc_zero` | seeds 0, 1, 2 | [ZIP](https://drive.google.com/file/d/1CKMDF9QGR2TUQxlmxYgU0dQ93OPf8egO/view) |
| `without_foreground_conditions` | seeds 0, 1, 2 | [ZIP](https://drive.google.com/file/d/1GCRCeiDykQNj7VSHlt-IvLs1SL6flAjX/view) |
| `p_only_zero` | seeds 0, 1, 2 | [ZIP](https://drive.google.com/file/d/17khg5qIysg9mUVqd0Fnph7rl4AhuE_gj/view) |
| `dense_only` | seeds 0, 1, 2 | [ZIP](https://drive.google.com/file/d/1Raar_CS7s0ITK-LPOQ3d89DIlBW532HQ/view) |
| `global_only` | seeds 0, 1, 2 | [ZIP](https://drive.google.com/file/d/1fCtF1eQzhtaaH_QnaUawPJ-gp0hOBv2u/view) |
| `p_only_weighted` | seeds 0, 1, 2 | [ZIP](https://drive.google.com/file/d/1-tyu3Zrv2ySQsVb0MNfqqT57Css8MgHU/view) |
| `token_ffn` | seeds 0, 1, 2 | [ZIP](https://drive.google.com/file/d/1e2025WVuW_dvJCUbTjucVFeE__AtFWc2/view) |
| `Q1_fixed5_noagg` | seeds 0, 1, 2 | [ZIP](https://drive.google.com/file/d/13Fj4N9AJSI7g53BDSZqxa7UcEih-v3Fy/view) |
| `Q0_hungarian5_noagg` | seeds 0, 1, 2 | [ZIP](https://drive.google.com/file/d/1RuSe7kvPWXiON_gwH8IUewC4buXRxeUu/view) |
