"""
MI-guided ARM: skalabilitas pada dataset gene expression PyAerial, sampai 18.108 gen.

Metode (tanpa neural network):
  1. Untuk setiap gen konsekuen i, pilih 2 gen antecedent dengan mutual information
     tertinggi terhadap gen i. MI dihitung per blok sehingga matriks MI penuh N x N
     tidak pernah disimpan (memori O(blok x N)).
  2. Kandidat dan penyaringan IDENTIK dengan VATTEN-noRL dan ablasi:
     nilai tunggal & pasangan nilai dari 2 gen terpilih -> konsekuen = nilai gen i
     dengan confidence tertinggi; simpan bila support >= min_support dan
     confidence >= 0.8.
  3. Pembanding dengan anggaran sama: 2 gen antecedent acak (3 seed).

min_support: untuk N <= 1000 memakai support 90% rule PyAerial dari run skalabilitas
(protokol PyAerial). Untuk N > 1000 PyAerial tidak selesai, sehingga dipakai 0.50
(rentang nilai PyAerial pada 50-1000 kolom: 0.43-0.51).

Output: migene_results.csv, migene_summary.txt, migene_log.txt
"""

import os
import sys
import time
import platform
import traceback

COLS = [50, 100, 200, 300, 500, 1000, 2000, 5000, 10000, 18108]
MIN_SUPPORT = {50: 0.4267, 100: 0.4613, 200: 0.4650, 300: 0.4380, 500: 0.4650, 1000: 0.5120}
MIN_SUPPORT_DEFAULT = 0.50
MIN_CONF = 0.8
TOP_ANT = 2
BLOCK = 256
RANDOM_SEEDS = [0, 1, 2]

DATA_NAME = "cell_cancertype_NonSmallCellLungCarcinoma_lnic50.csv"
SEARCH_DIRS = [".", r"D:\NS ARM BARU VS\data", r"D:\NS ARM BARU VS"]


def encode(df):
    import numpy as np
    cols = list(df.columns)
    cats = [sorted(df[c].unique()) for c in cols]
    cards = [len(c) for c in cats]
    codes = np.stack([df[c].map({v: i for i, v in enumerate(cats[k])}).values
                      for k, c in enumerate(cols)], 1).astype(np.int64)
    return codes, cards


def onehot(codes, cmax):
    import numpy as np
    n, f = codes.shape
    X = np.zeros((n, f, cmax), dtype=bool)
    X[np.arange(n)[:, None], np.arange(f)[None, :], codes] = True
    return X


def top_mi_partners(X, k=TOP_ANT, block=BLOCK):
    """Untuk tiap gen, k gen lain dengan MI tertinggi (blok demi blok)."""
    import numpy as np
    n, f, c = X.shape
    Xf = X.reshape(n, f * c).astype(np.float32)
    p = Xf.mean(0).reshape(f, c)
    out = np.zeros((f, k), dtype=np.int64)
    for s in range(0, f, block):
        e = min(s + block, f)
        P = (Xf[:, s * c:e * c].T @ Xf / n).reshape(e - s, c, f, c)
        outer = p[s:e, :, None, None] * p[None, None, :, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            MI = np.where(P > 0, P * np.log(P / outer), 0.0).sum(axis=(1, 3))
        MI[np.arange(e - s), np.arange(s, e)] = -np.inf
        idx = np.argpartition(-MI, k, axis=1)[:, :k]
        out[s:e] = idx
    return out


def random_partners(f, seed, k=TOP_ANT):
    import numpy as np
    rng = np.random.default_rng(seed)
    out = np.zeros((f, k), dtype=np.int64)
    for i in range(f):
        cand = rng.choice(f - 1, k, replace=False)
        out[i] = np.where(cand >= i, cand + 1, cand)
    return out


def mine(X, cards, partners, min_support):
    import numpy as np
    n, f, _ = X.shape
    sup_l, conf_l, lift_l, zhang_l, cons_l = [], [], [], [], []
    cov = np.zeros(n, dtype=bool)
    for i in range(f):
        cb = X[:, i, :cards[i]]
        pc = cb.mean(0)
        j1, j2 = partners[i]
        masks = [X[:, j, v] for j in (j1, j2) for v in range(cards[j])]
        masks += [X[:, j1, v1] & X[:, j2, v2] for v1 in range(cards[j1]) for v2 in range(cards[j2])]
        for m in masks:
            na = m.sum()
            if na == 0:
                continue
            cnt = cb[m].sum(0)
            v = int(cnt.argmax())
            sup, conf = cnt[v] / n, cnt[v] / na
            if sup < min_support or conf < MIN_CONF:
                continue
            pa, pb = na / n, pc[v]
            denom = max(sup * (1 - pa), pa * (pb - sup))
            sup_l.append(sup)
            conf_l.append(conf)
            lift_l.append(conf / pb)
            zhang_l.append((sup - pa * pb) / denom if denom > 0 else 0.0)
            cons_l.append(i)
            cov |= m
    if not sup_l:
        return {"rule_count": 0}
    lift = np.array(lift_l)
    return {"rule_count": len(sup_l), "average_support": float(np.mean(sup_l)),
            "average_confidence": float(np.mean(conf_l)), "average_lift": float(lift.mean()),
            "average_zhang": float(np.mean(zhang_l)), "data_coverage": float(cov.mean()),
            "consequent_genes_explained": len(set(cons_l)),
            "share_lift_gt_1_2": float((lift > 1.2).mean())}


class _Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        for st in self.s:
            try:
                st.write(x)
                st.flush()
            except Exception:
                pass

    def flush(self):
        pass


def main():
    import numpy as np
    import pandas as pd
    print(f"Python {platform.python_version()} | {platform.platform()} | numpy {np.__version__}")
    path = next((os.path.join(d, DATA_NAME) for d in SEARCH_DIRS
                 if os.path.exists(os.path.join(d, DATA_NAME))), None)
    if path is None:
        raise FileNotFoundError(f"{DATA_NAME} tidak ditemukan; salin ke folder skrip ini.")
    data = pd.read_csv(path).astype(str)
    print(f"Dataset: {path} -> {data.shape}")

    rows = []
    for n_cols in COLS:
        n_cols = min(n_cols, data.shape[1])
        ms = MIN_SUPPORT.get(n_cols, MIN_SUPPORT_DEFAULT)
        codes, cards = encode(data.iloc[:, :n_cols])
        X = onehot(codes, max(cards))
        t0 = time.perf_counter()
        partners = top_mi_partners(X)
        t_sel = time.perf_counter() - t0
        t1 = time.perf_counter()
        st = mine(X, cards, partners, ms)
        t_mine = time.perf_counter() - t1
        st.update(n_columns=n_cols, method="MI-guided", seed=None, min_support=ms,
                  selection_time_s=round(t_sel, 3), mining_time_s=round(t_mine, 3),
                  exec_time=round(t_sel + t_mine, 3))
        rows.append(st)
        print(f"[{n_cols:5d} kolom] MI-guided rules={st['rule_count']:5d} "
              f"lift={st.get('average_lift', 0):.3f} zhang={st.get('average_zhang', 0):.3f} "
              f"genes={st.get('consequent_genes_explained', 0)} "
              f"cov={st.get('data_coverage', 0):.3f} time={st['exec_time']:.1f}s")
        for seed in RANDOM_SEEDS:
            t1 = time.perf_counter()
            st = mine(X, cards, random_partners(len(cards), seed), ms)
            st.update(n_columns=n_cols, method="Random", seed=seed, min_support=ms,
                      selection_time_s=0.0, mining_time_s=round(time.perf_counter() - t1, 3),
                      exec_time=round(time.perf_counter() - t1, 3))
            rows.append(st)
        r = pd.DataFrame([x for x in rows if x["n_columns"] == n_cols and x["method"] == "Random"])
        print(f"               Random    rules={r.rule_count.mean():7.1f} "
              f"lift={r.average_lift.mean():.3f} zhang={r.average_zhang.mean():.3f} "
              f"genes={r.consequent_genes_explained.mean():.1f}")
        pd.DataFrame(rows).to_csv("migene_results.csv", index=False)

    res = pd.DataFrame(rows)
    agg = res.groupby(["n_columns", "method"], sort=False).mean(numeric_only=True).round(4)
    agg = agg.drop(columns=["seed"], errors="ignore")
    text = ("MI-guided ARM vs Random (anggaran 2 gen antecedent per konsekuen), dataset gene expression PyAerial\n\n"
            + agg.to_string())
    open("migene_summary.txt", "w", encoding="utf8").write(text)
    print("\n" + text)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    _log = open("migene_log.txt", "w", encoding="utf8")
    sys.stdout = sys.stderr = _Tee(sys.__stdout__, _log)
    try:
        main()
        print("\nSELESAI. Kirimkan migene_summary.txt, migene_results.csv, migene_log.txt.")
    except Exception:
        print("\nTERJADI ERROR:\n" + traceback.format_exc())
    sys.stdout = sys.__stdout__
    try:
        input("\nTekan Enter untuk menutup jendela...")
    except Exception:
        pass
