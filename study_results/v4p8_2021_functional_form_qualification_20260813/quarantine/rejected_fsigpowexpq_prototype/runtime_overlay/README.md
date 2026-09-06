# v4.6 deterministic-restart runtime overlay

The v4.6 fits used the declared v4.2 result/card state plus two narrowly scoped
runtime-instrumentation modules archived here.  These files do not change the
analysis card, GP kernel, likelihood, covariance construction, injection, or
extraction definitions.

- `hps_gpr/gpr.py` adds an explicit optimizer-restart random state and records
  optimizer warnings.  SHA-256:
  `1c83cae238e87a4e94928c97fb737947c22a3f88b16dfaf955d48ab6b4771dd5`.
- `hps_gpr/io.py` propagates the optimizer seed and warning diagnostics into
  `BlindPrediction`.  SHA-256:
  `b36f8da7671a0fc0958b663e11d83a1a4421e90d1aab9b10e40c31ce078035db`.

The archived copies are bitwise identical to the live modules used for
production.  Their uninstrumented v4.2 base blobs and the base `hps_gpr` tree
are identified separately in `study_spec.json`; the release validator checks
both layers.  To reconstruct the production runtime, check out the declared
integration commit and replace the two corresponding modules with these exact
files before running `run_study.py`.
