# assets/

Representative **real** inputs for prep (`config.yaml: model.example_input.assets_dir`). `profile_model.py`/`freeze_contract.py` pick the first image (`*.jpg|*.jpeg|*.png|*.webp|*.bmp`, sorted). Replace for a different model. Never use random/synthetic data — profile and golden must reflect a real forward.

- `car.png` — canonical Segment-Anything sample (real photo), SAM shakedown.
