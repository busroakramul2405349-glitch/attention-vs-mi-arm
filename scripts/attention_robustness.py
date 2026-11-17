"""
Attention robustness: is the weak attention guide an artefact of one configuration?

The equal-budget ablation (k antecedent features per consequent, identical candidates and filters) is repeated with
VATTEN variants, and with budgets k = 1, 2, 3 for the base model:
    base            d=64, 4 heads, 50 epochs, beta=1e-3 (the configuration of the manuscript)
    epochs150       three times longer training
    heads1 / heads8 single-head and eight-head attention
    d128            wider model
    beta0 / beta1e-2  no KL term / ten times stronger KL term
    noVAE           deterministic tokens (no variational embedding)
    sink            an extra learnable "sink" key/value that every feature may attend to; attention paid to the
                    sink is excluded, so features are ranked only by the attention they receive from each other
                    (tests whether attention-sink behaviour explains the weak guide)
References at the same k: Random features and mutual information (MI).

Datasets: gene expression (first GENE_COLS genes), chess, congressional voting and mushroom (PyAerial benchmark files).
Outputs: rob_per_run.csv, rob_summary.txt, rob_experiment_log.txt
"""
import os
import sys
import time
import platform
import subprocess
import traceback
import importlib.metadata as md
from itertools import combinations

DATASETS = ["gene", "chess_king_rook_vs_king_pawn", "congressional_voting", "mushroom"]
GENE_COLS = 300
GENE_MIN_SUPPORT = 0.4380        # PyAerial benchmark protocol for 300 genes
MIN_SUPPORT = 0.10
MIN_CONF = 0.80
SEEDS = [0, 1, 2]
K_VALUES = [1, 2, 3]             # budgets evaluated for the base model
VARIANTS = ["base", "epochs150", "heads1", "heads8", "d128", "beta0", "beta1e-2", "noVAE", "sink"]

BASE = {"epochs": 50, "d_model": 64, "n_heads": 4, "beta": 1e-3, "vae": True, "sink": False}
VARIANT_CFG = {
    "base": {},
    "epochs150": {"epochs": 150},
    "heads1": {"n_heads": 1},
    "heads8": {"n_heads": 8},
    "d128": {"d_model": 128},
    "beta0": {"beta": 0.0},
    "beta1e-2": {"beta": 1e-2},
    "noVAE": {"vae": False, "beta": 0.0},
    "sink": {"sink": True},
}
LR = 3e-3

GENE_FILE = "cell_cancertype_NonSmallCellLungCarcinoma_lnic50.csv"
DATA_DIRS = [".", "pyaerial_benchmark", os.path.join("..", "data", "pyaerial_benchmark")]
GENE_DIRS = [".", os.path.join("..", "data"), r"D:\NS ARM BARU VS\data", r"D:\NS ARM BARU VS"]
LOG_FILE = "rob_experiment_log.txt"

np = pd = None


def find(name, dirs):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def prepare(path):
    """PyAerial benchmark preparation: dropna, five equal-frequency bins for numerical columns."""
    df = pd.read_csv(path).dropna()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() > 5:
            try:
                df[col] = pd.qcut(df[col], q=5, duplicates="drop").astype(str)
            except ValueError:
                pass
    return df.astype(str).reset_index(drop=True)


def encode(df):
    cats = [sorted(df[c].unique()) for c in df.columns]
    cards = [len(c) for c in cats]
    codes = np.stack([df[c].map({v: i for i, v in enumerate(cats[k])}).values
                      for k, c in enumerate(df.columns)], 1).astype(np.int64)
    return codes, cards


def onehot_padded(codes, cmax):
    n, f = codes.shape
    X = np.zeros((n, f, cmax), dtype=bool)
    X[np.arange(n)[:, None], np.arange(f)[None, :], codes] = True
    return X


def train_attention(codes, cards, seed, epochs, d_model, n_heads, beta, vae, sink):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    n_feat, cmax = len(cards), max(cards)

    class VATTEN(nn.Module):
        def __init__(self):
            super().__init__()
            slots = np.array(cards) + 1
            self.register_buffer("offset", torch.tensor(np.concatenate([[0], np.cumsum(slots)[:-1]])))
            self.register_buffer("mask_id", torch.tensor(cards))
            self.val_emb = nn.Embedding(int(slots.sum()), d_model)
            self.feat_emb = nn.Parameter(torch.randn(n_feat, d_model) * 0.1)
            self.to_mu = nn.Linear(d_model, d_model)
            self.to_logvar = nn.Linear(d_model, d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            self.W = nn.Parameter(torch.randn(n_feat, d_model, cmax) * 0.02)
            self.b = nn.Parameter(torch.zeros(n_feat, cmax))
            valid = torch.zeros(n_feat, cmax, dtype=torch.bool)
            for k, c in enumerate(cards):
                valid[k, :c] = True
            self.register_buffer("valid", valid)
            mask = torch.eye(n_feat, dtype=torch.bool)
            if sink:
                self.sink = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)
                mask = torch.cat([mask, torch.zeros(n_feat, 1, dtype=torch.bool)], 1)  # sink always visible
            self.register_buffer("mask", mask)

        def forward(self, x):
            tok = self.val_emb(x + self.offset) + self.feat_emb
            if vae:
                mu, logvar = self.to_mu(tok), self.to_logvar(tok)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
            else:
                mu = logvar = None
                z = tok
            kv = torch.cat([z, self.sink.expand(z.shape[0], 1, -1)], 1) if sink else z
            a, w = self.attn(z, kv, kv, attn_mask=self.mask, need_weights=True, average_attn_weights=True)
            h = self.norm(a + self.feat_emb)
            logits = torch.einsum("bnd,ndc->bnc", h, self.W) + self.b
            return logits.masked_fill(~self.valid, -1e9), mu, logvar, w

    batch = (16 if n_feat <= 500 else 8) if len(codes) <= 1000 else 128
    model = VATTEN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    X = torch.tensor(codes)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), batch):
            xb = X[perm[i:i + batch]]
            drop = torch.rand(xb.shape, generator=g) < 0.15
            xin = torch.where(drop, model.mask_id.expand_as(xb), xb)
            logits, mu, logvar, _ = model(xin)
            loss = F.cross_entropy(logits.reshape(-1, cmax), xb.reshape(-1))
            if vae and beta > 0:
                loss = loss + beta * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        W = sum(model(X[i:i + batch])[3].sum(0) for i in range(0, len(X), batch)) / len(X)
    W = W.numpy().astype(float)
    sink_mass = float(W[:, n_feat:].sum(1).mean()) if sink else 0.0
    A = W[:, :n_feat].copy()
    np.fill_diagonal(A, 0.0)
    return A, sink_mass


def mutual_info_matrix(codes, cmax):
    n, f = codes.shape
    X = onehot_padded(codes, cmax).reshape(n, f * cmax).astype(np.float32)
    P = (X.T @ X / n).reshape(f, cmax, f, cmax)
    p = X.mean(0).reshape(f, cmax)
    outer = p[:, :, None, None] * p[None, None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(P > 0, P * np.log(P / outer), 0.0)
    MI = term.sum(axis=(1, 3))
    np.fill_diagonal(MI, 0.0)
    return MI


def top(row, i, k):
    order = np.argsort(row)[::-1]
    return [int(j) for j in order[order != i][:k]]


def mine(X, cards, picks, min_support):
    n = X.shape[0]
    rules = []
    for i, feats in enumerate(picks):
        cons_block = X[:, i, :cards[i]]
        p_cons = cons_block.mean(0)
        cands = [((j, v),) for j in feats for v in range(cards[j])]
        cands += [((j1, v1), (j2, v2)) for j1, j2 in combinations(feats, 2)
                  for v1 in range(cards[j1]) for v2 in range(cards[j2])]
        for a in cands:
            m = np.ones(n, dtype=bool)
            for j, v in a:
                m &= X[:, j, v]
            na = int(m.sum())
            if na == 0:
                continue
            cnt = cons_block[m].sum(0)
            v = int(cnt.argmax())
            sup, conf = cnt[v] / n, cnt[v] / na
            if sup < min_support or conf < MIN_CONF:
                continue
            pa, pb = na / n, p_cons[v]
            denom = max(sup * (1 - pa), pa * (pb - sup))
            rules.append({"cons": i, "support": sup, "confidence": conf,
                          "lift": conf / pb if pb > 0 else float("nan"),
                          "zhang": (sup - pa * pb) / denom if denom > 0 else 0.0, "mask": m})
    return rules


def summarize(rules, n):
    if not rules:
        return {"rule_count": 0, "average_support": None, "average_confidence": None, "average_lift": None,
                "average_zhang": None, "data_coverage": 0.0, "consequents_explained": 0}
    cov = np.zeros(n, dtype=bool)
    for r in rules:
        cov |= r["mask"]
    return {"rule_count": len(rules),
            "average_support": float(np.mean([r["support"] for r in rules])),
            "average_confidence": float(np.mean([r["confidence"] for r in rules])),
            "average_lift": float(np.nanmean([r["lift"] for r in rules])),
            "average_zhang": float(np.mean([r["zhang"] for r in rules])),
            "data_coverage": float(cov.mean()),
            "consequents_explained": len({r["cons"] for r in rules})}


def load(name):
    if name == "gene":
        p = find(GENE_FILE, GENE_DIRS)
        if p is None:
            return None, None, None
        return f"gene_{GENE_COLS}", pd.read_csv(p).astype(str).iloc[:, :GENE_COLS], GENE_MIN_SUPPORT
    p = find(name + ".csv", DATA_DIRS)
    if p is None:
        raise FileNotFoundError(f"{name}.csv not found (expected in data/pyaerial_benchmark/)")
    return name, prepare(p), MIN_SUPPORT


def main():
    print(f"Python {platform.python_version()} | {platform.platform()}")
    for p in ["numpy", "pandas", "torch"]:
        try:
            print(f"  {p} {md.version(p)}")
        except Exception:
            print(f"  {p} not available")
    rows = []
    for name in DATASETS:
        label, df, min_sup = load(name)
        if df is None:
            print(f"[warning] {GENE_FILE} not found; gene dataset skipped.")
            continue
        codes, cards = encode(df)
        n, f = codes.shape
        X = onehot_padded(codes, max(cards))
        MI = mutual_info_matrix(codes, max(cards))
        print(f"\n=== {label}: {n} rows x {f} features, min_support={min_sup} ===")

        def add(guide, seed, k, picks, extra):
            s = summarize(mine(X, cards, picks, min_sup), n)
            s.update(dataset=label, seed=seed, k=k, guide=guide, **extra)
            rows.append(s)
            pd.DataFrame(rows).to_csv("rob_per_run.csv", index=False)
            print(f"  seed {seed} k={k} {guide:26s} rules={s['rule_count']:5d} "
                  f"zhang={s['average_zhang'] or 0:.3f} lift={s['average_lift'] or 0:.3f} "
                  f"cons={s['consequents_explained']}")

        for seed in SEEDS:
            for k in K_VALUES:
                rng = np.random.default_rng(1000 + seed)
                add("Random", seed, k, [[int(j) for j in rng.choice(np.delete(np.arange(f), i), k, replace=False)]
                                        for i in range(f)], {})
                add("MutualInfo", seed, k, [top(MI[i], i, k) for i in range(f)], {})
            for v in VARIANTS:
                cfg = dict(BASE, **VARIANT_CFG[v])
                t = time.perf_counter()
                A, sink_mass = train_attention(codes, cards, seed, **cfg)
                t_train = time.perf_counter() - t
                extra = {"train_time_s": round(t_train, 2), "sink_mass": round(sink_mass, 4),
                         "attention_mi_overlap": float(np.mean([len(set(top(A[i], i, 2)) & set(top(MI[i], i, 2))) / 2
                                                                for i in range(f)]))}
                for k in (K_VALUES if v == "base" else [2]):
                    add(f"Attention[{v}]", seed, k, [top(A[i], i, k) for i in range(f)], extra)

    R = pd.DataFrame(rows)
    R.to_csv("rob_per_run.csv", index=False)
    lines = [f"Attention robustness | seeds={SEEDS} | datasets={sorted(R.dataset.unique())}", ""]
    for k in K_VALUES:
        T = R[R.k == k].pivot_table(index="dataset", columns="guide", values="average_zhang", aggfunc="mean")
        lines.append(f"Mean Zhang's metric, k = {k}:")
        lines.append("  " + T.round(4).T.to_string().replace("\n", "\n  "))
        lines.append("")
    att = R[R.guide.str.startswith("Attention") & (R.k == 2)]
    for v in VARIANTS:
        a = att[att.guide == f"Attention[{v}]"].set_index(["dataset", "seed"])["average_zhang"]
        r = R[(R.guide == "Random") & (R.k == 2)].set_index(["dataset", "seed"])["average_zhang"]
        m = R[(R.guide == "MutualInfo") & (R.k == 2)].set_index(["dataset", "seed"])["average_zhang"]
        j = pd.concat([a.rename("a"), r.rename("r"), m.rename("m")], axis=1).dropna()
        sm = att[att.guide == f"Attention[{v}]"]["sink_mass"].mean()
        lines.append(f"k=2 {v:10s}: beats Random in {int((j.a > j.r).sum())}/{len(j)} runs, beats MI in "
                     f"{int((j.a > j.m).sum())}/{len(j)} runs; mean Zhang {j.a.mean():.4f} "
                     f"(Random {j.r.mean():.4f}, MI {j.m.mean():.4f})"
                     + (f"; attention to sink {sm:.3f}" if v == "sink" else ""))
    lines.append("\nStatistical tests: statistical_tests.py (section C).")
    text = "\n".join(lines)
    with open("rob_summary.txt", "w", encoding="utf8") as f:
        f.write(text)
    print("\n" + text)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

    _log = open(LOG_FILE, "w", encoding="utf8")
    sys.stdout = sys.stderr = _Tee(sys.__stdout__, _log)
    code = 0
    try:
        missing = []
        for mod in ["numpy", "pandas", "torch"]:
            try:
                __import__(mod)
            except Exception:
                missing.append(mod)
        if missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        import numpy as np
        import pandas as pd
        t0 = time.time()
        main()
        print(f"\nSELESAI dalam {(time.time() - t0) / 60:.1f} menit. Kirimkan rob_summary.txt, rob_per_run.csv "
              "dan rob_experiment_log.txt.")
    except Exception:
        print("\nTERJADI ERROR:\n" + traceback.format_exc())
        code = 1
    sys.stdout = sys.__stdout__
    try:
        input("\nTekan Enter untuk menutup jendela...")
    except Exception:
        pass
    sys.exit(code)
