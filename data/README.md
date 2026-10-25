# Data

The raw CICEVSE2024 host-event table is not redistributed in this repository.
Download the dataset from the Canadian Institute for Cybersecurity (CIC)
CICEVSE2024 distribution and place the host-event file at:

```text
data/raw/EVSE-B-HPC-Kernel-Events-Combined.csv
```

The preparation script expects the file to be the combined host-event export
with 8,474 rows and 915 columns. Its SHA-256 checksum is:

```text
0025e946bb0c7959752df00df3b516012faabddbde0644577d5fb732963de553
```

The pipeline is deterministic. It removes rows with a blank `Attack` field,
removes the writeback artefact row, repairs the documented 2,302-row backdoor
positional shift, merges `serice-detection` into `service-detection`, assigns
sessions at timestamp resets, drops non-informative columns, and removes rows
with more than 50% missing features.

The prepared modelling table has:

- 8,468 rows
- 308 numeric host-event features
- 36 capture sessions
- 18 attack classes

## Committed derived files

| File | Description |
| --- | --- |
| `session_assignments.csv` | One row per capture session with row count, class composition, scenarios, states, time range, and source-row bounds. |
| `label_repair_evidence.csv` | Row ranges and class fields for the two backdoor sessions affected by the positional displacement. |

The 2,302 repaired rows are the two `backdoor` sessions: 1,151 `Charging`
rows from CSV source rows 6173-7323 and 1,151 `idle` rows from CSV source rows
7324-8475. Both groups map to `backdoor`, `host-attack`, `attack`, and
interface `any`.

## Regenerating derived files

```bash
python -m src.data_prep.data_prep
```

This command reads the raw file, validates its checksum and shape, and rewrites
the two derived files in this directory.
