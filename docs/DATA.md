# Data preparation

Obtain the benchmark data from its authors: [EgoHOS](https://github.com/owenzlz/EgoHOS), and the **InterFormer benchmark dataset archive** linked in its official README (the archive used here also contains mini-HOI4D). Do not substitute arbitrary HOI4D frames for the prescribed 1,095 images. The locally used InterFormer dataset link is https://drive.google.com/file/d/1Pc6Ofd3q-yEwByxIqgaJfEJv4QtWtKjP/view . Dataset terms and access permissions remain with the source authors.

Prepare the extracted InterFormer archive with the exact image/label-only conversion used for training (second-order/unknown IDs become 255 ignore):
```bash
python -m tools.prepare_data --source /path/to/extracted/datasets --output /path/to/egohos
python -m tools.check_data --data-root /path/to/egohos
```
The converter checks every source image and normalized label against the frozen manifest. It does not generate proxy masks or download data. Label pixels, rather than PNG compression bytes, determine label identity.

Expected structure:
```
egohos/
  train/{image/*.jpg,label/*.png}          # 8,993
  val/{image/*.jpg,label/*.png}            # 1,124
  test_indomain/{image/*.jpg,label/*.png}  # 1,126
  test_outdomain/{image/*.jpg,label/*.png} # 500
  minihoi4d/{image/*.jpg,label/*.png}      # 1,095
```

Images/masks pair by stem. Use the six-class labels from the benchmark: 0 background, 1 LH, 2 RH, 3 LO, 4 RO, 5 TO, 255 ignore. The unconverted original EgoHOS dataset may include second-order labels 6–8; do not silently treat these as first-order roles. `python -m tools.check_data --data-root ...` validates exact sample identities and file hashes against the release manifest, and fails if the data differ. Depth/contact maps are not needed. Do not place licensed data in the code Git repository.

Evaluation uses bilinear RGB resize and nearest-neighbor mask resize to 448 × 448. Foreground mIoU derives from the accumulated six-class confusion matrix, excluding BG from the mean. Mini averages LH, RH, RO and TO only, while any erroneous LO predictions still affect the confusion matrix. Avg is the unweighted arithmetic mean of ID/OOD/Mini; 1,124 validation images are not included. Full test evaluation has 2,721 images, not 3,845.
