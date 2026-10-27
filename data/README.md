# Data

Neither dataset is redistributed here; both are public.

## UCI Mushroom

Downloaded automatically by the Mushroom scripts via `ucimlrepo` (`fetch_ucirepo(id=73)`).

Mushroom [Dataset]. (1981). UCI Machine Learning Repository. https://doi.org/10.24432/C5959T (CC BY 4.0).

## Gene expression (PyAerial benchmark)

File: `cell_cancertype_NonSmallCellLungCarcinoma_lnic50.csv` (86 rows x 18,108 genes, values low/normal/high).

Download it from the PyAerial repository:
https://github.com/DiTEC-project/pyaerial/tree/main/illustrative_experiments/data

and place it in `scripts/` (or next to whichever script you run).

SHA-256 of the file used in the paper:

```
1a8ed010cba48820024e21c604ef5e05af65325fe292bb4f8ce081a820fb6525
```

Origin: Gao et al. (2015), Nature Medicine 21, 1318–1325, https://doi.org/10.1038/nm.3954, as pre-processed by
Ruiz et al. (2023, NeurIPS) and discretised by z-score binning in the PyAerial repository.
