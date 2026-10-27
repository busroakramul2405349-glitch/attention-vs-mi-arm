"""
VATTEN-Rule: ABLASI pemilihan antecedent pada dataset gene expression PyAerial.

Pertanyaan: apakah attention VATTEN memilih rule yang lebih BERMAKNA daripada
pemilihan lain dengan anggaran yang SAMA (2 gen antecedent per gen konsekuen)?

Strategi pemilihan 2 gen antecedent untuk setiap gen konsekuen i:
  - Attention   : 2 gen dengan bobot attention A[i, j] tertinggi (VATTEN, diagonal dimasking)
  - Random      : 2 gen acak
  - AttnBottom  : 2 gen dengan attention terendah (uji kewajaran: harus paling buruk)
  - MutualInfo  : 2 gen dengan mutual information tertinggi terhadap gen i
                  (baseline statistik sederhana tanpa neural network)

Untuk semua strategi, kandidat dan penyaringan IDENTIK: nilai tunggal & pasangan nilai
dari 2 gen terpilih -> konsekuen = nilai gen i dengan confidence tertinggi;
disimpan bila support >= min_support dan confidence >= 0.8.

min_support per jumlah kolom diambil dari run skalabilitas sebelumnya
(support yang mencakup 90% rule PyAerial): 100 -> 0.4613, 300 -> 0.4380, 1000 -> 0.5120.

Metrik per rule set: jumlah rule, rata-rata support, confidence, LIFT, ZHANG's metric,
data coverage, jumlah gen konsekuen yang terjelaskan, dan waktu pemilihan.

Output: abl_results_per_run.csv, abl_results_summary.txt, abl_experiment_log.txt
"""

import os
import sys
import time
import platform
import subprocess
import traceback

COLS = [100, 300, 1000]
MIN_SUPPORT = {100: 0.4613, 300: 0.4380, 1000: 0.5120}
SEEDS = [0, 1, 2]
MIN_CONF = 0.8
TOP_ANT = 2

D_MODEL = 64
N_HEADS = 4
V_EPOCHS = 50
LR = 3e-3
BETA_KL = 1e-3

DATA_NAME = "cell_cancertype_NonSmallCellLungCarcinoma_lnic50.csv"
SEARCH_DIRS = [".", r"D:\NS ARM BARU VS\data", r"D:\NS ARM BARU VS"]
STRATEGIES = ["Attention", "Random", "AttnBottom", "MutualInfo"]


def encode(df):
    import numpy as np
    cols = list(df.columns)
    cats = [sorted(df[c].unique()) for c in cols]
    cards = [len(c) for c in cats]
    codes = np.stack([df[c].map({v: i for i, v in enumerate(cats[k])}).values
                      for k, c in enumerate(cols)], 1).astype(np.int64)
    return codes, cards


def onehot_padded(codes, cmax):
    import numpy as np
    n, f = codes.shape
    X = np.zeros((n, f, cmax), dtype=bool)
    X[np.arange(n)[:, None], np.arange(f)[None, :], codes] = True
    return X


def train_attention(codes, cards, seed):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    n_feat = len(cards)
    cmax = max(cards)

    class VATTEN(nn.Module):
        def __init__(self):
            super().__init__()
            slots = np.array(cards) + 1
            self.register_buffer("offset", torch.tensor(np.concatenate([[0], np.cumsum(slots)[:-1]])))
            self.register_buffer("mask_id", torch.tensor(cards))
            self.val_emb = nn.Embedding(int(slots.sum()), D_MODEL)
            self.feat_emb = nn.Parameter(torch.randn(n_feat, D_MODEL) * 0.1)
            self.to_mu = nn.Linear(D_MODEL, D_MODEL)
            self.to_logvar = nn.Linear(D_MODEL, D_MODEL)
            self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
            self.norm = nn.LayerNorm(D_MODEL)
            self.W = nn.Parameter(torch.randn(n_feat, D_MODEL, cmax) * 0.02)
            self.b = nn.Parameter(torch.zeros(n_feat, cmax))
            valid = torch.zeros(n_feat, cmax, dtype=torch.bool)
            for k, c in enumerate(cards):
                valid[k, :c] = True
            self.register_buffer("valid", valid)
            self.register_buffer("diag", torch.eye(n_feat, dtype=torch.bool))

        def forward(self, x):
            tok = self.val_emb(x + self.offset) + self.feat_emb
            mu, logvar = self.to_mu(tok), self.to_logvar(tok)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
            a, w = self.attn(z, z, z, attn_mask=self.diag, need_weights=True,
                             average_attn_weights=True)
            h = self.norm(a + self.feat_emb)
            logits = torch.einsum("bnd,ndc->bnc", h, self.W) + self.b
            return logits.masked_fill(~self.valid, -1e9), mu, logvar, w

    batch = 16 if n_feat <= 500 else 8
    model = VATTEN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    X = torch.tensor(codes)
    g = torch.Generator().manual_seed(seed)
    for ep in range(V_EPOCHS):
        model.train()
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), batch):
            xb = X[perm[i:i + batch]]
            drop = torch.rand(xb.shape, generator=g) < 0.15
            xin = torch.where(drop, model.mask_id.expand_as(xb), xb)
            logits, mu, logvar, _ = model(xin)
            rec = F.cross_entropy(logits.reshape(-1, cmax), xb.reshape(-1))
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec + BETA_KL * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        A = sum(model(X[i:i + batch])[3].sum(0) for i in range(0, len(X), batch)) / len(X)
    A = A.numpy().astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def mutual_info_matrix(codes, cmax):
    """MI antar semua pasangan gen, divektorisasi lewat tabel kontingensi one-hot."""
    import numpy as np
    n, f = codes.shape
    X = onehot_padded(codes, cmax).reshape(n, f * cmax).astype(np.float32)
    P = (X.T @ X / n).reshape(f, cmax, f, cmax)                   # P(gen_i = a, gen_j = b)
    p = X.mean(0).reshape(f, cmax)                                 # P(gen_i = a)
    outer = p[:, :, None, None] * p[None, None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(P > 0, P * np.log(P / outer), 0.0)
    MI = term.sum(axis=(1, 3))
    np.fill_diagonal(MI, 0.0)
    return MI


def pick_antecedents(strategy, A, MI, i, rng):
    import numpy as np
    f = A.shape[0]
    others = np.delete(np.arange(f), i)
    if strategy == "Attention":
        return np.argsort(A[i])[::-1][:TOP_ANT]
    if strategy == "AttnBottom":
        return others[np.argsort(A[i, others])[:TOP_ANT]]
    if strategy == "MutualInfo":
        return np.argsort(MI[i])[::-1][:TOP_ANT]
    return rng.choice(others, TOP_ANT, replace=False)


def mine(codes, cards, picks, min_support):
    """Kandidat & penyaringan identik untuk semua strategi. picks[i] = gen antecedent untuk konsekuen i."""
    import numpy as np
    n, f = codes.shape
    cmax = max(cards)
    X = onehot_padded(codes, cmax)                    # (n, f, cmax)
    rules = []
    for i in range(f):
        cons_block = X[:, i, :cards[i]]
        p_cons = cons_block.mean(0)
        j1, j2 = picks[i]
        cands = [((j, v),) for j in (j1, j2) for v in range(cards[j])]
        cands += [((j1, v1), (j2, v2)) for v1 in range(cards[j1]) for v2 in range(cards[j2])]
        for a in cands:
            m = np.ones(n, dtype=bool)
            for j, v in a:
                m &= X[:, j, v]
            na = m.sum()
            if na == 0:
                continue
            cnt = cons_block[m].sum(0)
            v = int(cnt.argmax())
            sup = cnt[v] / n
            conf = cnt[v] / na
            if sup < min_support or conf < MIN_CONF:
                continue
            pa, pb = na / n, p_cons[v]
            lift = conf / pb if pb > 0 else np.nan
            denom = max(sup * (1 - pa), pa * (pb - sup))
            zhang = (sup - pa * pb) / denom if denom > 0 else 0.0
            rules.append({"ant": a, "cons": (i, v), "support": sup, "confidence": conf,
                          "lift": lift, "zhang": zhang, "mask": m})
    return rules


def summarize(rules, n):
    import numpy as np
    if not rules:
        return {"rule_count": 0, "average_support": None, "average_confidence": None,
                "average_lift": None, "average_zhang": None, "data_coverage": 0.0,
                "consequent_genes_explained": 0, "share_lift_gt_1_2": None}
    cov = np.zeros(n, dtype=bool)
    for r in rules:
        cov |= r["mask"]
    lift = np.array([r["lift"] for r in rules])
    return {"rule_count": len(rules),
            "average_support": float(np.mean([r["support"] for r in rules])),
            "average_confidence": float(np.mean([r["confidence"] for r in rules])),
            "average_lift": float(np.nanmean(lift)),
            "average_zhang": float(np.mean([r["zhang"] for r in rules])),
            "data_coverage": float(cov.mean()),
            "consequent_genes_explained": len({r["cons"][0] for r in rules}),
            "share_lift_gt_1_2": float(np.nanmean(lift > 1.2))}


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
    import importlib.metadata as md

    print(f"Python {platform.python_version()} | {platform.platform()}")
    for p in ["numpy", "pandas", "torch"]:
        try:
            print(f"  {p} {md.version(p)}")
        except Exception:
            pass
    path = next((os.path.join(d, DATA_NAME) for d in SEARCH_DIRS
                 if os.path.exists(os.path.join(d, DATA_NAME))), None)
    if path is None:
        raise FileNotFoundError(f"{DATA_NAME} tidak ditemukan; salin ke folder skrip ini.")
    data = pd.read_csv(path).astype(str)
    print(f"Dataset: {path} -> {data.shape}")

    rows = []
    for n_cols in COLS:
        codes, cards = encode(data.iloc[:, :n_cols])
        n = len(codes)
        t = time.perf_counter()
        MI = mutual_info_matrix(codes, max(cards))
        t_mi = time.perf_counter() - t
        print(f"\n=== {n_cols} kolom (min_support={MIN_SUPPORT[n_cols]}) | MI dihitung {t_mi:.2f}s ===")
        for seed in SEEDS:
            t = time.perf_counter()
            A = train_attention(codes, cards, seed)
            t_attn = time.perf_counter() - t
            rng = np.random.default_rng(1000 + seed)
            for strat in STRATEGIES:
                t = time.perf_counter()
                picks = [pick_antecedents(strat, A, MI, i, rng) for i in range(len(cards))]
                rules = mine(codes, cards, picks, MIN_SUPPORT[n_cols])
                t_mine = time.perf_counter() - t
                s = summarize(rules, n)
                sel_time = {"Attention": t_attn, "AttnBottom": t_attn, "MutualInfo": t_mi,
                            "Random": 0.0}[strat]
                s.update(n_columns=n_cols, seed=seed, strategy=strat,
                         selection_time_s=round(sel_time, 2), mining_time_s=round(t_mine, 2))
                rows.append(s)
                pd.DataFrame(rows).to_csv("abl_results_per_run.csv", index=False)
                print(f"  seed {seed} {strat:11s} rules={s['rule_count']:4d} "
                      f"lift={s['average_lift'] or 0:.3f} zhang={s['average_zhang'] or 0:.3f} "
                      f"conf={s['average_confidence'] or 0:.3f} cov={s['data_coverage']:.3f} "
                      f"genes={s['consequent_genes_explained']} lift>1.2={s['share_lift_gt_1_2'] or 0:.2f}")
            # korelasi pilihan attention dan MI: seberapa sering gen pilihan sama
            top_a = [set(np.argsort(A[i])[::-1][:TOP_ANT]) for i in range(len(cards))]
            top_m = [set(np.argsort(MI[i])[::-1][:TOP_ANT]) for i in range(len(cards))]
            overlap = np.mean([len(a & b) / TOP_ANT for a, b in zip(top_a, top_m)])
            print(f"  seed {seed}: overlap gen pilihan Attention vs MutualInfo = {overlap:.3f}")
            rows[-1]["attention_mi_overlap"] = overlap

    res = pd.DataFrame(rows)
    res.to_csv("abl_results_per_run.csv", index=False)
    metrics = ["rule_count", "average_support", "average_confidence", "average_lift",
               "average_zhang", "data_coverage", "consequent_genes_explained",
               "share_lift_gt_1_2", "selection_time_s"]
    lines = [f"Ablasi pemilihan antecedent (anggaran {TOP_ANT} gen/konsekuen), seeds={SEEDS}", ""]
    for n_cols in COLS:
        lines.append(f"[{n_cols} kolom]")
        for strat in STRATEGIES:
            g = res[(res.n_columns == n_cols) & (res.strategy == strat)]
            parts = []
            for c in metrics:
                v = pd.to_numeric(g[c], errors="coerce").dropna()
                if len(v):
                    parts.append(f"{c}={v.mean():.4f}±{v.std(ddof=1) if len(v) > 1 else 0:.4f}")
            lines.append(f"  {strat:11s} " + "  ".join(parts))
    lines.append("")
    lines.append("Perbandingan berpasangan (per kolom x seed), Attention dikurangi strategi lain:")
    for other in ["Random", "AttnBottom", "MutualInfo"]:
        for c in ["average_lift", "average_zhang", "rule_count", "consequent_genes_explained"]:
            a = res[res.strategy == "Attention"].set_index(["n_columns", "seed"])[c]
            b = res[res.strategy == other].set_index(["n_columns", "seed"])[c]
            d = (pd.to_numeric(a) - pd.to_numeric(b)).dropna()
            lines.append(f"  vs {other:10s} {c:27s} rata-rata selisih={d.mean():+.4f}  "
                         f"Attention lebih tinggi {int((d > 0).sum())}/{len(d)}")
    text = "\n".join(lines)
    open("abl_results_summary.txt", "w", encoding="utf8").write(text)
    print("\n" + text)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    _log = open("abl_experiment_log.txt", "w", encoding="utf8")
    sys.stdout = sys.stderr = _Tee(sys.__stdout__, _log)
    try:
        missing = []
        for mod in ["numpy", "pandas", "torch"]:
            try:
                __import__(mod)
            except Exception:
                missing.append(mod)
        if missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        main()
        print("\nSELESAI. Kirimkan abl_results_summary.txt, abl_results_per_run.csv, abl_experiment_log.txt.")
    except Exception:
        print("\nTERJADI ERROR:\n" + traceback.format_exc())
    sys.stdout = sys.__stdout__
    try:
        input("\nTekan Enter untuk menutup jendela...")
    except Exception:
        pass
