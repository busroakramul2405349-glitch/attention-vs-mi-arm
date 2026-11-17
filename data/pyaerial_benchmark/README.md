# PyAerial rule-quality benchmark datasets

These nine CSV files are copied unchanged from the PyAerial repository
(https://github.com/DiTEC-project/pyaerial, folder `benchmark/datasets`, commit
9e78b20f03ee0c101e9708b1c70660df27acc927, release v1.1.0, MIT licence). They are the complete set of datasets used by
PyAerial's own rule-quality benchmark (`benchmark/rule_quality.py`). All originate from the UCI Machine Learning
Repository and are distributed there under CC BY 4.0.

`scripts/multi_dataset_benchmark.py` prepares them exactly as the PyAerial benchmark does: rows with missing values are
dropped and numerical columns with more than five distinct values are discretised into five equal-frequency bins; the
last column is the class.

| File | UCI id | Rows (raw) | Columns | SHA-256 |
| --- | --- | --- | --- | --- |
| acute_inflammations.csv | 184 | 120 | 7 | 7a992356eac452f8835866e3ffe65c9163cac14015f58fca3b5251d899e5af5d |
| autism_screening_adolescent.csv | 420 | 104 | 21 | 19dc0851e0b10063803458c4889bbd27f895fba3b295c73ee011136c90439d05 |
| breast_cancer.csv | 14 | 286 | 10 | 68fc2ca428d0a05449a4136883850748fc234aa8ad3f94e978a4dd6363ec1225 |
| cervical_cancer_behavior_risk.csv | 537 | 72 | 20 | d8b207ff86fe805796c286b95018cfbfaf776e7e7571a3b9e27a42a885acbe67 |
| chess_king_rook_vs_king_pawn.csv | 22 | 3196 | 36 | 4e771d74e1253c9352ce1002d245bb2bf5ee1b11a0e06b6bcf89d6c25c2dd522 |
| congressional_voting.csv | 105 | 435 | 17 | ff6def8ff72ba647dc7bc7499330bdeac426afb9c0c48fb48f1582c8886a4f54 |
| fertility.csv | 244 | 100 | 10 | 3e18d8d16b6381493aee95ee5c730e28366dfd732e38dd6a1feb47e8a05e1ea4 |
| hepatitis.csv | 46 | 155 | 20 | 8f1f69fa2c6a6528cce9a18a8d5f319a330dd899f2461fee500cf5ccc5ba0c7e |
| mushroom.csv | 73 | 8124 | 23 | 0479d653aaaed91545a90364ddd8604f3fb64bd5db3b9d648fe81ab62f9bf887 |
