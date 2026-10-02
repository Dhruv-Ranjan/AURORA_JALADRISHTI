# Remote training

This package streams Sentinel-2 L2A metadata and image windows from the
Microsoft Planetary Computer. Discovery requests STAC metadata only.
`RemoteWindowReader` opens each signed Cloud-Optimized GeoTIFF and calls
`rasterio.read(window=...)`; it never falls back to downloading a complete
scene.

The iterator counts valid training samples after retrieval and validation.
Rejected candidates are classified, and iteration continues through the
candidate pool until `target_valid_samples` is reached or the pool is
exhausted. A bounded access-time cache is optional and has a hard byte limit.

## Metadata smoke test

```bash
python -m remote_training.smoke_remote \
  --regions-json remote_training/regions.example.json \
  --max-candidates 5
```

This command performs metadata discovery only. It does not read imagery.

## Training an existing model

The trainer accepts a factory for the existing spectral model rather than
silently substituting a toy model:

```bash
python -m remote_training.train_remote \
  --model-factory model_defs:SpectralNetV10 \
  --regions-json remote_training/regions.example.json \
  --target-valid-samples 100000 \
  --cache-size-gb 2
```

Run from the project environment with the directory containing
`model_defs.py` on `PYTHONPATH`. CUDA is selected automatically when
available, mixed precision is enabled on CUDA, and checkpoints contain model,
optimizer, epoch, configuration, and sample-accounting state.

The reported valid count is the number of samples that entered training, not
the number of STAC candidates. No claim should be made about a target count
until `training_report.json` confirms it.
