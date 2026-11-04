# Data Availability

The method-validation benchmark inputs, analysis implementations, saved predictions, and result tables are provided in the [VentureBridge repository](https://github.com/nicoletacristache/VentureBridge/tree/main/method_validation).

- [Benchmark inputs and protocol assets](method_validation/datasets/benchmark_data.zip)
- [Final prediction archive](method_validation/reference_outputs/frozen_predictions.zip)
- [Analysis code](method_validation/analysis/)
- [Summary tables and figures](method_validation/results/)

Use `method_validation/prepare_assets.py --with-predictions` to restore the archived files and verify their original hashes. Original-file identifiers and checksums are listed in `method_validation/source_manifest.json`.

Dataset origin, reference-label interpretation, and the scope of this release are documented centrally in [PROVENANCE.md](method_validation/PROVENANCE.md). This statement covers method-validation assets only; it does not describe availability of separate application-validation or classroom-study data.
