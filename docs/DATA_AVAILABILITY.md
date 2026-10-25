# Data and code availability

## Data availability

This study uses the CICEVSE2024 host-event dataset distributed by the Canadian
Institute for Cybersecurity (CIC) at the University of New Brunswick. The raw
dataset is publicly available from the CICEVSE2024 distribution. It is not
redistributed in this repository.

The repository includes the derived session-assignment table and the
label-repair evidence needed to audit the preparation. The preparation script
reproduces both files from the public raw export. The expected file name,
checksum, input shape, and preparation summary are documented in
[`data/README.md`](../data/README.md).

## Code availability

All experiment code, configuration, saved predictions, metrics, confusion
matrices, and figure-generation scripts are available in this repository. The
four experimental conditions, five random seeds, and both model families are
enumerated in `configs/run_matrix.csv`.

The XGBoost results are the primary analysis. LightGBM provides an independent
model-family check using the same splits and evaluation protocol. Saved
prediction files can be re-scored with:

```bash
python -m src.evaluation.recompute_metrics
```

## Licensing

The source code and documentation are released under the MIT License. The raw
CICEVSE2024 data remain subject to the terms of their original distribution.
