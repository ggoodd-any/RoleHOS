# License and attribution scope

The root MIT license covers original RoleHOS release tooling and original project contributions, subject to retained notices. It does not relicense third-party code, pretrained material, or dataset images.

- `third_party/mmseg/`: reduced local MMSegmentation research source, OpenMMLab Apache-2.0 notices retained (`licenses/Apache-2.0.txt`). Original class names are retained for strict checkpoint compatibility. Unrelated registries and the unused Mask2Former head were removed; `point_sample` uses an equivalent PyTorch reference. This is not a full MMSegmentation distribution.
- `third_party/eupe/models/convnext.py`: byte-identical locally used Meta/EUPE ConvNeXt implementation, under the retained **FAIR Noncommercial Research License**, including its acceptable-use policy (`licenses/FAIR-EUPE.md`). Its terms apply to this component; the complete distribution is not uniformly MIT.
- `weights/`: includes fine-tuned DINOv3 backbone tensors. The **DINOv3 License** applies to DINO materials and their derivatives (`licenses/DINOv3.md`). Retain this license with each weight distribution. These are not MIT-only weights. Training initialization is obtained separately from Meta under its license.
- `examples/`: local preview copies of EgoHOS / mini-HOI4D test images and role masks. Their original dataset/source terms apply. EgoHOS's repository MIT license covers software and does not establish an unrestricted image sublicense. Do not automatically publish these local image copies; `tools/package_release.py` excludes them by default and retains the sample manifest/import utility.

Original projects: [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [DINOv3](https://github.com/facebookresearch/dinov3), [EgoHOS](https://github.com/owenzlz/EgoHOS), [HOI4D](https://hoi4d.github.io/).

Code and checkpoint archives are distributed separately with these notices. A fully MIT-only backbone/weight distribution would require a different compatible implementation and appropriate weight rights; this package does not silently change the published model to achieve that.
