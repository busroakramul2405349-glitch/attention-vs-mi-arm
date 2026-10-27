# Does Attention Help Neuro-Symbolic Association Rule Mining?

Code and per-run results for the manuscript *"Does Attention Help Neuro-Symbolic Association Rule
Mining? An Empirical Study and a Mutual-Information-Guided Baseline for High-Dimensional Tabular Data"*
(Umam, Prasetya, Gunawan).

Every number in the manuscript can be traced to a file in `results/`.

## Contents

| Folder / file | What it is | Manuscript |
| --- | --- | --- |
| `scripts/run_pyaerial_mushroom.py` | Replication of PyAerial on UCI Mushroom with the settings of the PyAerial paper | Sec. 5.1 (context) |
| `scripts/vatten_mushroom_experiment.py` | VATTEN-Rule vs FP-Growth, Apriori, PyAerial, random features on Mushroom | Table 1 |
| `scripts/mi_guided_mushroom_experiment.py` | Same as above plus MI-noRL and MI+RL (superset; used for Table 1) | Tables 1–2, Fig. 3 |
| `scripts/vatten_scalability_experiment.py` | Scalability on the PyAerial gene-expression benchmark, 50–1,000 genes | Table 3, Fig. 1 |
| `scripts/vatten_ablation_experiment.py` | Equal-budget ablation: attention, lowest attention, random, mutual information | Table 4, Fig. 2 |
| `scripts/mi_guided_gene_scalability.py` | MI-guided mining up to all 18,108 genes, plus random baseline | Table 3, Fig. 1 |
| `scripts/make_figures.py` | Rebuilds Figures 1–3 from `results/` | Figs. 1–3 |
| `results/` | Logs, per-run CSVs and summaries exactly as produced by the scripts | all tables |
| `figures/` | Figures 1–3 (300 dpi) | Figs. 1–3 |
| `data/README.md` | How to obtain the two datasets | Sec. 4.1 |

`results/mushroom/first_run_vatten_only/` holds the first Mushroom run, made before the MI guide was added.
Its VATTEN, FP-Growth, Apriori and random results are identical to those in `results/mushroom/`.

## Environment used for the reported results

HP EliteBook x360 1030 G3 laptop, Intel Core i5-8350U (4 cores / 8 threads, 1.70 GHz), 8 GB RAM,
Windows 11 Pro. Python 3.13.14, NumPy 2.4.6, pandas 3.0.3, PyTorch 2.13.0 (CPU), MLxtend 0.25.0,
PyAerial 1.1.0. Runs were made on 25–26 October 2026.

## Reproducing

```
pip install -r requirements.txt
python scripts/mi_guided_mushroom_experiment.py      # about 5 minutes
python scripts/vatten_scalability_experiment.py      # about 1.5 hours (includes 1,000 s timeouts)
python scripts/vatten_ablation_experiment.py         # about 10 minutes
python scripts/mi_guided_gene_scalability.py         # about 2 minutes
python scripts/make_figures.py
```

The Mushroom scripts download the data automatically with `ucimlrepo`. The gene-expression scripts look for
`cell_cancertype_NonSmallCellLungCarcinoma_lnic50.csv` next to the script; see `data/README.md`.
Each script writes its outputs next to itself and keeps the console window open at the end on Windows.

## Seeds and repetitions

Mushroom: seeds 0–4. Gene benchmark: PyAerial and VATTEN-Rule three runs per size (seeds 0–2); FP-Growth and
the MI guide are deterministic. Ablation: seeds 0–2 (random choice uses seed 1000 + s).

## Licence

Code: MIT (see `LICENSE`). The datasets keep their own licences (see `data/README.md`).
