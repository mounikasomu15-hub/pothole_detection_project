# Testing
1. `python scripts/smoke_test.py` - runs every script end-to-end on synthetic data in a temp folder (installation check only).
2. `python scripts/validate_dataset.py` - data integrity on your real dataset.
3. `python models/unet.py` - model output shape check.
4. `python training/evaluate_unet.py` and `python training/evaluate_random_forest.py` - held-out test metrics (real, computed).
5. `python scripts/run_pipeline.py --image <file>` - one-image command-line test.
6. Manual UI test: upload an image with and without a pothole; try with models missing (expect a clear message); toggle voice (expect one alert per cooldown).
Record your measured results in your report only after running these.
