# Model details
**U-Net:** input (N,3,S,S) -> output (N,1,S,S) logits; DoubleConv = 2 x (Conv3x3-BN-ReLU); channels c, 2c, 4c, 8c, 16c (c = `UNET_BASE_CHANNELS`, default 32);
transposed-convolution upsampling with skip concatenation; run `python models/unet.py` to print the output shape and parameter count.
**Metrics** (`utils/metrics.py`, micro-averaged over all test pixels): IoU = TP/(TP+FP+FN); Dice = 2TP/(2TP+FP+FN); precision; recall; F1; pixel accuracy.
**Random Forest:** `RandomForestClassifier(n_estimators=200, class_weight="balanced")`; features: area_ratio, width, height, aspect_ratio, perimeter, circularity, relative_size, solidity, num_regions.
**Confidence:** U-Net = mean probability over predicted pothole pixels; Random Forest = `predict_proba` max. These are model scores, not calibrated guarantees.
Final numbers: *generated after training* in `outputs/reports/*.json`.
