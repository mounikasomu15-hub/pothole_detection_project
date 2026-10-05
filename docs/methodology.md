# Methodology
1. **Preprocessing:** resize to `IMAGE_SIZE` (area interpolation for images, nearest-neighbour for masks), scale to [0,1], normalise, convert to tensor.
2. **Segmentation:** U-Net (4 encoder levels, bottleneck, 4 decoder levels with skip connections, 1x1 output). Loss = BCE-with-logits + soft Dice. AdamW, ReduceLROnPlateau, early stopping on validation Dice.
3. **Detection:** pixels with probability > `SEGMENTATION_THRESHOLD`; tiny regions removed; a pothole is *detected* if mask area ratio >= `MIN_AREA_RATIO` and segmentation confidence >= the confidence threshold.
4. **Features:** area, area ratio, bounding-box width/height, aspect ratio, width/height ratio, perimeter, contour area, circularity, relative size (box/image), solidity, shape irregularity (1 - solidity), number of regions.
5. **Severity:** Random Forest on `FEATURE_NAMES` (see `inference/feature_extraction.py`). Output class + `predict_proba()` confidence.
6. **Speed:** lookup table in `config.py` (LOW/MEDIUM/HIGH_SPEED). **Voice:** offline pyttsx3 with cooldown.

**Depth:** not measured. "Severity" is an image-based estimate from 2-D shape and size.
**Severity labels:** rule-based (`area_ratio` thresholds or dataset terciles), or manual. Document which one you used.
