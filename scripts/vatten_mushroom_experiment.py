"""
VATTEN-Rule: eksperimen NYATA pada UCI Mushroom, dibandingkan dengan
FP-Growth, Apriori (MLxtend) dan PyAerial, memakai pengklasifikasi rule yang SAMA.

Semua angka yang dilaporkan dihitung dari data; tidak ada nilai simulasi.

Cara pakai (Windows):
    klik dua kali file ini, ATAU di Command Prompt:
    python vatten_mushroom_experiment.py

Output (di folder yang sama dengan skrip):
    vatten_results_per_seed.csv   satu baris per metode per seed
    vatten_results_summary.csv    rata-rata dan simpangan baku per metode
    vatten_results_summary.txt    ringkasan siap kutip
    vatten_rules_seed0.txt        rule akhir VATTEN-Rule (seed 0)
    vatten_attention_seed0.csv    matriks attention antar fitur (seed 0), baris = konsekuen
    vatten_rl_curve_seed0.csv     kurva reward RL per episode (seed 0)
    vatten_experiment_log.txt     log lengkap

Desain eksperimen
-----------------
- Data: UCI Mushroom (8.124 baris, 22 fitur kategorikal + kelas 'poisonous').
- Per seed: split stratified train/val/test = 56/14/30 (%).
- Semua metode menghasilkan CLASS RULES (antecedent -> poisonous=e/p) dengan
  panjang antecedent <= 2, min_support = 0.05, min_confidence = 0.80 (dihitung di train).
- Pengklasifikasi yang sama untuk semua metode: rule diurutkan menurut
  confidence (lalu support); baris uji diprediksi oleh rule pertama yang cocok;
  jika tidak ada yang cocok -> kelas mayoritas train.
- Metrik di data uji: accuracy, rule coverage (proporsi baris yang cocok >= 1 rule),
  jumlah rule, rata-rata support dan confidence (train), runtime (detik).

VATTEN-Rule (versi yang diperbaiki)
-----------------------------------
1. Tokenisasi: setiap fitur (termasuk kelas) menjadi satu token =
   embedding nilai + embedding identitas fitur.
2. Variational Tabular Embedding: tiap token -> (mu, logvar) -> z (reparameterisasi).
3. Multi-Head Self-Attention dengan diagonal DIMASKING: tiap fitur harus
   direkonstruksi dari fitur LAIN, sehingga bobot attention A[i, j] = seberapa kuat
   fitur j dipakai untuk memprediksi fitur i (j = antecedent, i = konsekuen).
4. Loss = cross-entropy rekonstruksi + beta * KL.
5. Kandidat rule: K fitur dengan attention tertinggi ke token kelas; kandidat =
   semua nilai tunggal dan pasangan nilai dari fitur-fitur itu yang lolos
   min_support / min_confidence di train.
6. RL pruning (REINFORCE, kebijakan Bernoulli per kandidat): reward dihitung di
   data VALIDASI = accuracy + lambda1 * coverage - lambda2 * jumlah_rule.

Ablasi
------
- VATTEN-noRL      : semua kandidat attention tanpa pruning RL
- RandomFeat+RL    : K fitur dipilih acak (tanpa attention), lalu RL yang sama
"""

import os
import sys
import time
import platform
import subprocess
import traceback
import importlib.metadata as md
from itertools import combinations

os.chdir(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = "vatten_experiment_log.txt"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


_log = open(LOG_FILE, "w", encoding="utf8")
sys.stdout = sys.stderr = _Tee(sys.__stdout__, _log)


def _pause_and_exit(code=0):
    _log.flush()
    sys.stdout = sys.__stdout__
    try:
        input("\nTekan Enter untuk menutup jendela...")
    except Exception:
        pass
    sys.exit(code)


# ---------------------------------------------------------------------------
# Instal library yang belum ada
# ---------------------------------------------------------------------------
REQUIRED = {"numpy": "numpy", "pandas": "pandas", "ucimlrepo": "ucimlrepo",
            "mlxtend": "mlxtend", "torch": "torch"}
missing = []
for mod, pkg in REQUIRED.items():
    try:
        __import__(mod)
    except Exception:
        missing.append(pkg)
if missing:
    print(f"Menginstal library yang belum ada: {', '.join(missing)} (bisa beberapa menit)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    except Exception:
        print("GAGAL instal otomatis:\n" + traceback.format_exc())
        print(f'Coba manual: "{sys.executable}" -m pip install {" ".join(missing)}')
        _pause_and_exit(1)

# PyAerial opsional: jika gagal diinstal, eksperimen tetap berjalan tanpa baseline ini
try:
    __import__("aerial")
except Exception:
    print("Menginstal pyaerial (opsional)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyaerial"])
    except Exception:
        print("  pyaerial gagal diinstal; baseline PyAerial akan dilewati.")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from ucimlrepo import fetch_ucirepo
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules

try:
    from aerial import model as aerial_model, rule_extraction as aerial_rules
    HAS_PYAERIAL = True
except Exception:
    HAS_PYAERIAL = False

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
TARGET = "poisonous"
MIN_SUPPORT = 0.05
MIN_CONF = 0.80
MAX_ANT = 2

# VATTEN
D_MODEL = 64          # dimensi key/latent (sesuai naskah)
N_HEADS = 4
EPOCHS = 30
BATCH = 256
LR = 3e-3
BETA_KL = 1e-3
TOP_K_FEATURES = 6    # fitur antecedent dengan attention tertinggi ke kelas

# RL (REINFORCE)
RL_EPISODES = 1000
RL_SAMPLES = 16       # subset yang di-sample per episode
RL_LR = 0.1
LAMBDA1 = 1.0         # bobot coverage (sesuai naskah)
LAMBDA2 = 0.05        # penalti per rule (sesuai naskah)


# ---------------------------------------------------------------------------
# Data dan utilitas rule
# ---------------------------------------------------------------------------
def load_mushroom():
    uci = fetch_ucirepo(id=73)
    df = pd.concat([uci.data.features, uci.data.targets], axis=1)
    df = df.astype(object).where(df.notna(), "?").astype(str)
    return df


def stratified_split(y, seed, test=0.30, val=0.14):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_te = int(round(len(idx) * test))
        n_va = int(round(len(idx) * val))
        te += list(idx[:n_te])
        va += list(idx[n_te:n_te + n_va])
        tr += list(idx[n_te + n_va:])
    return np.array(tr), np.array(va), np.array(te)


class ItemSpace:
    """Item = 'fitur=nilai' untuk fitur non-kelas."""

    def __init__(self, df_train, feats):
        self.items = [f"{f}={v}" for f in feats for v in sorted(df_train[f].unique())]
        self.index = {it: i for i, it in enumerate(self.items)}
        self.feats = feats

    def matrix(self, df):
        M = np.zeros((len(df), len(self.items)), dtype=bool)
        for f in self.feats:
            vals = df[f].values
            for v in np.unique(vals):
                key = f"{f}={v}"
                if key in self.index:
                    M[vals == v, self.index[key]] = True
        return M


class RuleSet:
    """Kumpulan class rule: antecedent (tuple indeks item) -> label kelas."""

    def __init__(self, ants, labels, Xtr, ytr):
        self.ants = list(ants)
        self.labels = np.array(labels, dtype=object)
        self.support, self.conf = [], []
        for a, lab in zip(self.ants, self.labels):
            m = Xtr[:, list(a)].all(1)
            n_ant = m.sum()
            n_both = (m & (ytr == lab)).sum()
            self.support.append(n_both / len(ytr))
            self.conf.append(n_both / n_ant if n_ant else 0.0)
        self.support = np.array(self.support)
        self.conf = np.array(self.conf)
        # urutan pengklasifikasi: confidence lalu support (menurun)
        self.order = np.lexsort((-self.support, -self.conf)) if len(self.ants) else np.array([], int)

    def __len__(self):
        return len(self.ants)

    def match(self, X):
        if not self.ants:
            return np.zeros((len(X), 0), dtype=bool)
        return np.stack([X[:, list(a)].all(1) for a in self.ants], axis=1)


def evaluate(rs, X, y, default, select=None):
    """Akurasi dan coverage pengklasifikasi rule-list. select = mask rule terpilih."""
    if len(rs) == 0:
        return float(np.mean(y == default)), 0.0, 0
    sel = np.ones(len(rs), bool) if select is None else select
    order = [i for i in rs.order if sel[i]]
    if not order:
        return float(np.mean(y == default)), 0.0, 0
    Mo = rs.match(X)[:, order]
    hit = Mo.any(1)
    first = Mo.argmax(1)
    pred = np.where(hit, rs.labels[np.array(order)][first], default)
    return float(np.mean(pred == y)), float(hit.mean()), len(order)


def filter_rules(cands, Xtr, ytr):
    """Ambil kandidat (antecedent) -> kelas mayoritas, lolos min_support & min_conf."""
    ants, labs = [], []
    seen = set()
    for a in cands:
        a = tuple(sorted(a))
        if a in seen:
            continue
        seen.add(a)
        m = Xtr[:, list(a)].all(1)
        n = m.sum()
        if n == 0:
            continue
        vals, cnt = np.unique(ytr[m], return_counts=True)
        lab = vals[cnt.argmax()]
        sup = cnt.max() / len(ytr)
        conf = cnt.max() / n
        if sup >= MIN_SUPPORT and conf >= MIN_CONF:
            ants.append(a)
            labs.append(lab)
    return ants, labs


# ---------------------------------------------------------------------------
# Baseline: FP-Growth / Apriori (MLxtend)
# ---------------------------------------------------------------------------
def mlxtend_rules(df_tr, space, Xtr, ytr, algo):
    onehot = pd.DataFrame(Xtr, columns=space.items)
    for c in np.unique(ytr):
        onehot[f"{TARGET}={c}"] = ytr == c
    t0 = time.perf_counter()
    fi = algo(onehot, min_support=MIN_SUPPORT, use_colnames=True, max_len=MAX_ANT + 1)
    try:
        ar = association_rules(fi, metric="confidence", min_threshold=MIN_CONF)
    except TypeError:
        ar = association_rules(fi, num_itemsets=len(onehot), metric="confidence",
                               min_threshold=MIN_CONF)
    runtime = time.perf_counter() - t0
    n_all = len(ar)
    cls_items = {f"{TARGET}={c}" for c in np.unique(ytr)}
    ants, labs = [], []
    for a, c in zip(ar["antecedents"], ar["consequents"]):
        if len(c) == 1 and next(iter(c)) in cls_items and not (set(a) & cls_items):
            ants.append(tuple(sorted(space.index[i] for i in a)))
            labs.append(next(iter(c)).split("=", 1)[1])
    return RuleSet(ants, labs, Xtr, ytr), runtime, n_all


# ---------------------------------------------------------------------------
# Baseline: PyAerial
# ---------------------------------------------------------------------------
def _unpack(res):
    if isinstance(res, dict):
        return res.get("rules", []) or []
    return list(res or [])


def pyaerial_rules(df_tr, space, Xtr, ytr):
    t0 = time.perf_counter()
    ae = aerial_model.train(df_tr, epochs=2, layer_dims=[10], show_progress=False)
    all_rules = _unpack(aerial_rules.generate_rules(ae, max_antecedents=MAX_ANT))
    cls = _unpack(aerial_rules.generate_rules(ae, max_antecedents=MAX_ANT,
                                               target_classes=[TARGET]))
    runtime = time.perf_counter() - t0
    ants, labs = [], []
    for r in cls:
        con = r.get("consequent", {})
        if con.get("feature") != TARGET:
            continue
        try:
            a = tuple(sorted(space.index[f"{x['feature']}={x['value']}"]
                             for x in r.get("antecedents", [])))
        except KeyError:
            continue
        if a:
            ants.append(a)
            labs.append(str(con.get("value")))
    return RuleSet(ants, labs, Xtr, ytr), runtime, len(all_rules)


# ---------------------------------------------------------------------------
# VATTEN-Rule
# ---------------------------------------------------------------------------
class VATTEN(nn.Module):
    def __init__(self, cards, d=D_MODEL, heads=N_HEADS):
        super().__init__()
        self.n = len(cards)
        self.val_emb = nn.ModuleList([nn.Embedding(c + 1, d) for c in cards])  # +1 = token mask
        self.feat_emb = nn.Parameter(torch.randn(self.n, d) * 0.1)
        self.to_mu = nn.Linear(d, d)
        self.to_logvar = nn.Linear(d, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.out = nn.ModuleList([nn.Linear(d, c) for c in cards])
        self.register_buffer("diag", torch.eye(self.n, dtype=torch.bool))

    def forward(self, x):
        tok = torch.stack([self.val_emb[f](x[:, f]) for f in range(self.n)], 1) + self.feat_emb
        mu, logvar = self.to_mu(tok), self.to_logvar(tok)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        a, w = self.attn(z, z, z, attn_mask=self.diag, need_weights=True,
                         average_attn_weights=True)
        h = self.norm(a + self.feat_emb)  # tanpa residual nilai sendiri: prediksi dari fitur lain
        logits = [self.out[f](h[:, f]) for f in range(self.n)]
        return logits, mu, logvar, w


def train_vatten(codes, cards, seed):
    torch.manual_seed(seed)
    model = VATTEN(cards)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    X = torch.tensor(codes, dtype=torch.long)
    mask_ids = torch.tensor(cards, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(X), generator=g)
        tot = 0.0
        model.train()
        for i in range(0, len(X), BATCH):
            xb = X[perm[i:i + BATCH]]
            drop = torch.rand(xb.shape, generator=g) < 0.15  # input dropout (denoising)
            xin = torch.where(drop, mask_ids.expand_as(xb), xb)
            logits, mu, logvar, _ = model(xin)
            rec = sum(F.cross_entropy(logits[f], xb[:, f]) for f in range(len(cards))) / len(cards)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec + BETA_KL * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        if (ep + 1) % 10 == 0:
            print(f"    VATTEN epoch {ep + 1}/{EPOCHS} loss {tot / len(X):.4f}")
    model.eval()
    with torch.no_grad():
        ws = [model(X[i:i + 1024])[3] for i in range(0, len(X), 1024)]
        A = torch.cat(ws).mean(0).numpy()
    np.fill_diagonal(A, 0.0)
    return A


def candidates_from_features(feat_list, space, Xtr):
    cols = {f: [space.index[it] for it in space.items if it.startswith(f + "=")] for f in feat_list}
    cands = [(i,) for f in feat_list for i in cols[f]]
    for f1, f2 in combinations(feat_list, 2):
        for i in cols[f1]:
            for j in cols[f2]:
                cands.append((i, j))
    return cands


def rl_prune(rs, Xva, yva, default, seed, log_curve=None):
    """REINFORCE dengan kebijakan Bernoulli independen per rule (dihitung dengan numpy)."""
    rng = np.random.default_rng(seed)
    M = len(rs)
    if M == 0:
        return np.zeros(0, bool)
    theta = np.zeros(M)
    m_t, v_t = np.zeros(M), np.zeros(M)  # Adam
    match = rs.match(Xva)
    order = rs.order

    def reward(sel):
        o = [i for i in order if sel[i]]
        if not o:
            return float(np.mean(yva == default)) - 1.0  # penalti subset kosong
        Mo = match[:, o]
        hit = Mo.any(1)
        pred = np.where(hit, rs.labels[np.array(o)][Mo.argmax(1)], default)
        acc = float(np.mean(pred == yva))
        return acc + LAMBDA1 * float(hit.mean()) - LAMBDA2 * len(o)

    for ep in range(1, RL_EPISODES + 1):
        p = 1.0 / (1.0 + np.exp(-theta))
        acts = rng.random((RL_SAMPLES, M)) < p
        R = np.array([reward(a) for a in acts])
        adv = R - R.mean()
        grad = (adv[:, None] * (acts - p)).mean(0)  # gradien log-prob Bernoulli
        m_t = 0.9 * m_t + 0.1 * grad
        v_t = 0.999 * v_t + 0.001 * grad ** 2
        theta += RL_LR * (m_t / (1 - 0.9 ** ep)) / (np.sqrt(v_t / (1 - 0.999 ** ep)) + 1e-8)
        if log_curve is not None:
            log_curve.append({"episode": ep, "mean_reward": R.mean(), "max_reward": R.max(),
                              "mean_rules_sampled": acts.sum(1).mean(),
                              "expected_rules": p.sum()})
    sel = theta > 0
    if not sel.any():
        sel[np.argmax(theta)] = True
    return sel


# ---------------------------------------------------------------------------
# Eksperimen utama
# ---------------------------------------------------------------------------
def main():
    print(f"Python {platform.python_version()} | {platform.platform()}")
    for p in ["numpy", "pandas", "torch", "mlxtend", "pyaerial"]:
        try:
            print(f"  {p} {md.version(p)}")
        except Exception:
            print(f"  {p} tidak tersedia")
    if not HAS_PYAERIAL:
        print("  [peringatan] PyAerial gagal di-import; baseline PyAerial dilewati.")

    df = load_mushroom()
    feats = [c for c in df.columns if c != TARGET]
    print(f"Mushroom: {df.shape[0]} baris, {len(feats)} fitur + kelas")

    rows, rule_dump = [], []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        np.random.seed(seed)
        y = df[TARGET].values
        tr, va, te = stratified_split(y, seed)
        dtr, dva, dte = df.iloc[tr], df.iloc[va], df.iloc[te]
        space = ItemSpace(dtr, feats)
        Xtr, Xva, Xte = space.matrix(dtr), space.matrix(dva), space.matrix(dte)
        ytr, yva, yte = y[tr], y[va], y[te]
        default = pd.Series(ytr).mode()[0]

        def record(name, rs, runtime, sel=None, n_all=None):
            acc, cov, k = evaluate(rs, Xte, yte, default, sel)
            idx = np.arange(len(rs)) if sel is None else np.where(sel)[0]
            rows.append({"seed": seed, "method": name, "n_rules": k,
                         "n_all_association_rules": n_all,
                         "test_accuracy": round(acc, 4), "test_coverage": round(cov, 4),
                         "avg_support": round(float(rs.support[idx].mean()), 4) if k else None,
                         "avg_confidence": round(float(rs.conf[idx].mean()), 4) if k else None,
                         "runtime_s": round(runtime, 3)})
            print(f"  {name:16s} rules={k:5d} acc={acc:.4f} cov={cov:.4f} time={runtime:.2f}s")

        # Baselines
        rs, t, n_all = mlxtend_rules(dtr, space, Xtr, ytr, fpgrowth)
        record("FP-Growth", rs, t, n_all=n_all)
        rs, t, n_all = mlxtend_rules(dtr, space, Xtr, ytr, apriori)
        record("Apriori", rs, t, n_all=n_all)
        if HAS_PYAERIAL:
            try:
                rs, t, n_all = pyaerial_rules(dtr, space, Xtr, ytr)
                record("PyAerial", rs, t, n_all=n_all)
            except Exception:
                print("  PyAerial gagal:\n" + traceback.format_exc())

        # VATTEN-Rule
        t0 = time.perf_counter()
        all_cols = feats + [TARGET]
        cats = {c: sorted(dtr[c].unique()) for c in all_cols}
        def encode(d):
            return np.stack([d[c].map({v: i for i, v in enumerate(cats[c])})
                             .fillna(len(cats[c])).astype(int).values for c in all_cols], 1)
        A = train_vatten(encode(dtr), [len(cats[c]) for c in all_cols], seed)
        cls_row = A[len(feats), :len(feats)]  # attention kelas (konsekuen) -> fitur (antecedent)
        top_feats = [feats[i] for i in np.argsort(cls_row)[::-1][:TOP_K_FEATURES]]
        print(f"  Top fitur menurut attention ke kelas: {top_feats}")
        ants, labs = filter_rules(candidates_from_features(top_feats, space, Xtr), Xtr, ytr)
        cand = RuleSet(ants, labs, Xtr, ytr)
        t_cand = time.perf_counter() - t0
        record("VATTEN-noRL", cand, t_cand)

        curve = [] if seed == SEEDS[0] else None
        t1 = time.perf_counter()
        sel = rl_prune(cand, Xva, yva, default, seed, curve)
        record("VATTEN-Rule", cand, t_cand + time.perf_counter() - t1, sel=sel)

        # Ablasi: fitur acak + RL yang sama
        rng = np.random.default_rng(seed + 100)
        rand_feats = list(rng.choice(feats, TOP_K_FEATURES, replace=False))
        t2 = time.perf_counter()
        ra, rl = filter_rules(candidates_from_features(rand_feats, space, Xtr), Xtr, ytr)
        rcand = RuleSet(ra, rl, Xtr, ytr)
        rsel = rl_prune(rcand, Xva, yva, default, seed)
        record("RandomFeat+RL", rcand, time.perf_counter() - t2, sel=rsel)

        if seed == SEEDS[0]:
            pd.DataFrame(A, index=all_cols, columns=all_cols).round(5).to_csv(
                "vatten_attention_seed0.csv")
            pd.DataFrame(curve).to_csv("vatten_rl_curve_seed0.csv", index=False)
            for i in [i for i in cand.order if sel[i]]:
                ant = " AND ".join(space.items[j] for j in cand.ants[i])
                rule_dump.append(f"IF {ant} THEN {TARGET}={cand.labels[i]}  "
                                 f"(support={cand.support[i]:.3f}, confidence={cand.conf[i]:.3f})")

    res = pd.DataFrame(rows)
    res.to_csv("vatten_results_per_seed.csv", index=False)
    metrics = ["n_rules", "n_all_association_rules", "test_accuracy", "test_coverage",
               "avg_support", "avg_confidence", "runtime_s"]
    summ = res.groupby("method", sort=False)[metrics].agg(["mean", "std"]).round(4)
    summ.to_csv("vatten_results_summary.csv")

    lines = [f"UCI Mushroom, seeds={SEEDS}, split train/val/test=56/14/30, "
             f"min_support={MIN_SUPPORT}, min_conf={MIN_CONF}, max_antecedents={MAX_ANT}",
             f"VATTEN: d={D_MODEL}, heads={N_HEADS}, epochs={EPOCHS}, top_k={TOP_K_FEATURES}, "
             f"RL episodes={RL_EPISODES}, lambda1={LAMBDA1}, lambda2={LAMBDA2}", ""]
    for m in res["method"].unique():
        s = res[res["method"] == m]
        parts = []
        for c in metrics:
            v = pd.to_numeric(s[c], errors="coerce").dropna()
            if len(v):
                parts.append(f"{c}={v.mean():.4f}±{v.std(ddof=1) if len(v) > 1 else 0:.4f}")
        lines.append(f"{m:16s} " + "  ".join(parts))
    summary = "\n".join(lines)
    open("vatten_results_summary.txt", "w", encoding="utf8").write(summary)
    open("vatten_rules_seed0.txt", "w", encoding="utf8").write("\n".join(rule_dump))
    print("\n" + summary)
    print("\nRule akhir VATTEN-Rule (seed 0):")
    print("\n".join(rule_dump) if rule_dump else "  (tidak ada)")


if __name__ == "__main__":
    try:
        main()
        print("\nSELESAI. Kirimkan file vatten_results_summary.txt, vatten_results_per_seed.csv,"
              " vatten_rules_seed0.txt, dan vatten_experiment_log.txt.")
    except Exception:
        print("\nTERJADI ERROR:\n" + traceback.format_exc())
        print("Kirimkan file vatten_experiment_log.txt untuk diperbaiki.")
    _pause_and_exit(0)
