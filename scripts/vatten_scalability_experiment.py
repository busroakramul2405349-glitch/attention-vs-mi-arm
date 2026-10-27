"""
VATTEN-Rule: eksperimen SKALABILITAS NYATA pada dataset gene expression yang
dipakai paper PyAerial (Karabulut et al., 2025, SoftwareX).

Dataset : cell_cancertype_NonSmallCellLungCarcinoma_lnic50.csv
          86 baris x 18.108 gen, nilai low/normal/high (identik dengan file di
          github.com/DiTEC-project/pyaerial/illustrative_experiments/data)

Protokol (mengikuti illustrative_experiments/experiment_benchmarking.py PyAerial):
  - ambil N kolom gen pertama, N = 50, 100, 200, 300, 500, 1000
  - PyAerial: epochs=5, layer_dims=[5], max_antecedents=2,
              min_rule_frequency=0.5, min_rule_strength=0.8
  - min_support FP-Growth = support yang mencakup 90% rule PyAerial (rata-rata run),
    min_confidence = 0.8, panjang antecedent <= 2 (max_len = 3)
  - metrik: jumlah rule, rata-rata support, rata-rata confidence,
    data coverage (proporsi baris yang memenuhi antecedent >= 1 rule), waktu (detik)
  - batas waktu per metode per N = 1000 detik

VATTEN-Rule (tanpa label kelas):
  - token per gen = embedding nilai + embedding identitas gen
  - VAE per token -> Multi-Head Self-Attention dengan diagonal dimasking
    (setiap gen direkonstruksi dari gen lain)
  - kandidat rule: untuk setiap gen konsekuen i, ambil 2 gen dengan attention A[i, j]
    tertinggi sebagai calon antecedent; kandidat = nilai tunggal & pasangan nilai,
    konsekuen = nilai gen i dengan confidence tertinggi; disaring dengan
    min_support (sama dengan FP-Growth) dan min_confidence 0.8
  - RL (REINFORCE): reward = rata-rata confidence + lambda1*coverage - lambda2*jumlah_rule
  - dilaporkan dua varian: VATTEN-noRL (semua kandidat attention) dan VATTEN-Rule (setelah RL)

Cara pakai:
  Letakkan skrip di folder mana saja. Skrip mencari file dataset di folder skrip,
  lalu di D:\\NS ARM BARU VS\\data\\. Jalankan dengan klik dua kali atau:
      python vatten_scalability_experiment.py

Output:
  scal_results_per_run.csv, scal_results_summary.csv, scal_results_summary.txt,
  scal_experiment_log.txt
"""

import os
import sys
import time
import platform
import subprocess
import traceback
import multiprocessing as mp

COLS = [50, 100, 200, 300, 500, 1000]
RUNS = 3                     # PyAerial dan VATTEN tidak deterministik
TIMEOUT_S = 1000
ANTECEDENTS = 2
MIN_CONF = 0.8

# PyAerial (sesuai experiment_benchmarking.py)
AE_EPOCHS = 5
AE_LAYERS = [5]
ANT_SIM = 0.5
CONS_SIM = 0.8

# VATTEN
D_MODEL = 64
N_HEADS = 4
V_EPOCHS = 50
TOP_ANT = 2
LR = 3e-3
BETA_KL = 1e-3
RL_EPISODES = 1000
RL_SAMPLES = 16
RL_LR = 0.1
LAMBDA1 = 1.0
LAMBDA2 = 0.05

DATA_NAME = "cell_cancertype_NonSmallCellLungCarcinoma_lnic50.csv"
SEARCH_DIRS = [".", r"D:\NS ARM BARU VS\data", r"D:\NS ARM BARU VS"]


# ---------------------------------------------------------------------------
# Pekerja yang dijalankan di proses terpisah (agar bisa diberi batas waktu)
# ---------------------------------------------------------------------------
def _onehot(df):
    import pandas as pd
    return pd.get_dummies(df.astype(str), prefix_sep="__").astype(bool)


def _coverage(antecedent_index_lists, X):
    import numpy as np
    cov = np.zeros(X.shape[0], dtype=bool)
    for idx in antecedent_index_lists:
        cov |= X[:, list(idx)].all(axis=1)
    return float(cov.mean())


def worker_fpgrowth(df, min_support, q):
    try:
        import numpy as np
        from mlxtend.frequent_patterns import fpgrowth, association_rules
        oh = _onehot(df)
        t0 = time.perf_counter()
        fi = fpgrowth(oh, min_support=min_support, max_len=ANTECEDENTS + 1)
        try:
            ar = association_rules(fi, metric="confidence", min_threshold=MIN_CONF)
        except TypeError:
            ar = association_rules(fi, num_itemsets=len(oh), metric="confidence",
                                   min_threshold=MIN_CONF)
        t = time.perf_counter() - t0
        X = oh.values
        col = {c: i for i, c in enumerate(oh.columns)}
        ants = [[col[a] if isinstance(a, str) else a for a in s] for s in ar["antecedents"]]
        q.put({"rule_count": len(ar),
               "average_support": float(ar["support"].mean()) if len(ar) else None,
               "average_confidence": float(ar["confidence"].mean()) if len(ar) else None,
               "data_coverage": _coverage(ants, X) if len(ar) else 0.0,
               "exec_time": t})
    except Exception:
        q.put({"error": traceback.format_exc()})


def worker_pyaerial(df, seed, q):
    try:
        import numpy as np
        import torch
        torch.manual_seed(seed)
        np.random.seed(seed)
        from aerial import model, rule_extraction
        t0 = time.perf_counter()
        ae = model.train(df, epochs=AE_EPOCHS, layer_dims=AE_LAYERS, show_progress=False)
        res = rule_extraction.generate_rules(ae, max_antecedents=ANTECEDENTS,
                                             min_rule_frequency=ANT_SIM,
                                             min_rule_strength=CONS_SIM)
        t = time.perf_counter() - t0
        rules = res.get("rules", []) if isinstance(res, dict) else list(res or [])
        st = res.get("statistics", {}) if isinstance(res, dict) else {}
        sup = sorted((r["support"] for r in rules), reverse=True)
        s90 = sup[max(int(len(sup) * 0.9) - 1, 0)] if sup else None
        q.put({"rule_count": len(rules),
               "average_support": st.get("average_support"),
               "average_confidence": st.get("average_confidence"),
               "data_coverage": st.get("data_coverage"),
               "exec_time": t, "support_90": s90})
    except Exception:
        q.put({"error": traceback.format_exc()})


def run_with_timeout(target, args):
    q = mp.Queue()
    p = mp.Process(target=target, args=(*args, q))
    p.start()
    p.join(TIMEOUT_S)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"timeout": True}
    try:
        return q.get(timeout=30)
    except Exception:
        return {"error": f"proses berhenti tanpa hasil (exit code {p.exitcode}; kemungkinan kehabisan memori)"}


# ---------------------------------------------------------------------------
# VATTEN-Rule (vektorisasi agar sanggup ratusan-ribuan kolom)
# ---------------------------------------------------------------------------
def run_vatten(df, min_support, seed):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cols = list(df.columns)
    n_feat = len(cols)
    cats = [sorted(df[c].unique()) for c in cols]
    cards = [len(c) for c in cats]
    cmax = max(cards)
    codes = np.stack([df[c].map({v: i for i, v in enumerate(cats[k])}).values
                      for k, c in enumerate(cols)], 1).astype(np.int64)

    class VATTEN(nn.Module):
        def __init__(self):
            super().__init__()
            slots = np.array(cards) + 1  # +1 slot mask per gen
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
            logits = logits.masked_fill(~self.valid, -1e9)
            return logits, mu, logvar, w

    batch = 16 if n_feat <= 500 else 8
    t0 = time.perf_counter()
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
    A = A.numpy()
    np.fill_diagonal(A, 0.0)
    t_train = time.perf_counter() - t0
    return rules_from_attention(codes, cards, A, min_support, seed, t_train)


def rules_from_attention(codes, cards, A, min_support, seed, t_train):
    """Kandidat rule dari matriks attention + pruning RL (numpy murni)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n_feat = len(cards)

    # Ruang item one-hot
    t1 = time.perf_counter()
    item_cols = [np.arange(c) for c in cards]
    base = np.concatenate([[0], np.cumsum(cards)[:-1]])
    Xb = np.zeros((len(codes), int(sum(cards))), dtype=bool)
    for k in range(n_feat):
        Xb[np.arange(len(codes)), base[k] + codes[:, k]] = True
    n = len(codes)

    ants, cons, sups, confs = [], [], [], []
    seen = set()
    for i in range(n_feat):
        top = np.argsort(A[i])[::-1][:TOP_ANT]
        cand = [(base[j] + v,) for j in top for v in item_cols[j]]
        if len(top) == 2:
            j1, j2 = top
            cand += [(base[j1] + v1, base[j2] + v2) for v1 in item_cols[j1] for v2 in item_cols[j2]]
        cons_block = Xb[:, base[i]:base[i] + cards[i]]
        for a in cand:
            m = Xb[:, list(a)].all(1)
            na = m.sum()
            if na == 0:
                continue
            cnt = cons_block[m].sum(0)
            v = int(cnt.argmax())
            sup = cnt[v] / n
            conf = cnt[v] / na
            key = (tuple(sorted(a)), base[i] + v)
            if sup >= min_support and conf >= MIN_CONF and key not in seen:
                seen.add(key)
                ants.append(key[0])
                cons.append(key[1])
                sups.append(sup)
                confs.append(conf)
    sups, confs = np.array(sups), np.array(confs)
    t_cand = time.perf_counter() - t1

    def stats(sel):
        idx = np.where(sel)[0]
        if len(idx) == 0:
            return {"rule_count": 0, "average_support": None, "average_confidence": None,
                    "data_coverage": 0.0}
        return {"rule_count": int(len(idx)), "average_support": float(sups[idx].mean()),
                "average_confidence": float(confs[idx].mean()),
                "data_coverage": _coverage([ants[k] for k in idx], Xb)}

    M = len(ants)
    norl = stats(np.ones(M, bool))
    norl["exec_time"] = t_train + t_cand

    # RL pruning
    t2 = time.perf_counter()
    if M > 0:
        match = np.stack([Xb[:, list(a)].all(1) for a in ants], 1)
        theta, m_t, v_t = np.zeros(M), np.zeros(M), np.zeros(M)
        for ep in range(1, RL_EPISODES + 1):
            p = 1 / (1 + np.exp(-theta))
            acts = rng.random((RL_SAMPLES, M)) < p
            R = np.empty(RL_SAMPLES)
            for s in range(RL_SAMPLES):
                k = acts[s].sum()
                if k == 0:
                    R[s] = -1.0
                else:
                    R[s] = confs[acts[s]].mean() + LAMBDA1 * match[:, acts[s]].any(1).mean() - LAMBDA2 * k
            grad = ((R - R.mean())[:, None] * (acts - p)).mean(0)
            m_t = 0.9 * m_t + 0.1 * grad
            v_t = 0.999 * v_t + 0.001 * grad ** 2
            theta += RL_LR * (m_t / (1 - 0.9 ** ep)) / (np.sqrt(v_t / (1 - 0.999 ** ep)) + 1e-8)
        sel = theta > 0
        if not sel.any():
            sel[np.argmax(theta)] = True
    else:
        sel = np.zeros(0, bool)
    rl = stats(sel)
    rl["exec_time"] = t_train + t_cand + (time.perf_counter() - t2)
    return norl, rl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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


def setup():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    log = open("scal_experiment_log.txt", "w", encoding="utf8")
    sys.stdout = sys.stderr = _Tee(sys.__stdout__, log)
    need = {"numpy": "numpy", "pandas": "pandas", "torch": "torch", "mlxtend": "mlxtend",
            "aerial": "pyaerial"}
    missing = []
    for mod, pkg in need.items():
        try:
            __import__(mod)
        except Exception:
            missing.append(pkg)
    if missing:
        print(f"Menginstal: {', '.join(missing)} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    return log


def find_data():
    for d in SEARCH_DIRS:
        p = os.path.join(d, DATA_NAME)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"File {DATA_NAME} tidak ditemukan. Salin file itu ke folder yang sama dengan skrip ini.")


def main():
    import numpy as np
    import pandas as pd
    import importlib.metadata as md

    print(f"Python {platform.python_version()} | {platform.platform()}")
    for p in ["numpy", "pandas", "torch", "mlxtend", "pyaerial"]:
        try:
            print(f"  {p} {md.version(p)}")
        except Exception:
            pass
    path = find_data()
    data = pd.read_csv(path).astype(str)
    print(f"Dataset: {path} -> {data.shape[0]} baris x {data.shape[1]} gen")

    rows = []

    def add(n_cols, run, method, st):
        r = {"n_columns": n_cols, "run": run, "method": method}
        if st.get("timeout"):
            r.update(status=f"timeout >{TIMEOUT_S}s")
        elif st.get("error"):
            r.update(status="error")
            print(f"    {method} ERROR:\n{st['error']}")
        else:
            r.update(status="ok", **{k: st.get(k) for k in
                                     ["rule_count", "average_support", "average_confidence",
                                      "data_coverage", "exec_time"]})
        rows.append(r)
        msg = r["status"] if r["status"] != "ok" else (
            f"rules={r['rule_count']} sup={r['average_support'] or 0:.3f} "
            f"conf={r['average_confidence'] or 0:.3f} cov={r['data_coverage'] or 0:.3f} "
            f"time={r['exec_time']:.2f}s")
        print(f"    {method:12s} {msg}")
        pd.DataFrame(rows).to_csv("scal_results_per_run.csv", index=False)  # simpan bertahap

    for n_cols in COLS:
        sub = data.iloc[:, :n_cols]
        print(f"\n=== {n_cols} kolom ===")
        s90 = []
        for run in range(RUNS):
            st = run_with_timeout(worker_pyaerial, (sub, run))
            add(n_cols, run, "PyAerial", st)
            if st.get("support_90") is not None:
                s90.append(st["support_90"])
        min_sup = float(np.mean(s90)) if s90 else 0.5
        print(f"  min_support (90% rule PyAerial) = {min_sup:.4f}")

        add(n_cols, 0, "FP-Growth", run_with_timeout(worker_fpgrowth, (sub, min_sup)))

        for run in range(RUNS):
            try:
                norl, rl = run_vatten(sub, min_sup, run)
                add(n_cols, run, "VATTEN-noRL", norl)
                add(n_cols, run, "VATTEN-Rule", rl)
            except Exception:
                add(n_cols, run, "VATTEN-Rule", {"error": traceback.format_exc()})

    res = pd.DataFrame(rows)
    ok = res[res["status"] == "ok"].copy()
    num = ["rule_count", "average_support", "average_confidence", "data_coverage", "exec_time"]
    for c in num:
        ok[c] = pd.to_numeric(ok[c], errors="coerce")
    summ = ok.groupby(["n_columns", "method"], sort=False)[num].agg(["mean", "std"]).round(4)
    summ.to_csv("scal_results_summary.csv")

    lines = [f"Gene expression (86 baris), kolom={COLS}, runs={RUNS}, timeout={TIMEOUT_S}s, "
             f"antecedent<={ANTECEDENTS}, min_conf={MIN_CONF}", ""]
    for n_cols in COLS:
        lines.append(f"[{n_cols} kolom]")
        for m in ["PyAerial", "FP-Growth", "VATTEN-noRL", "VATTEN-Rule"]:
            s = res[(res.n_columns == n_cols) & (res.method == m)]
            if s.empty:
                continue
            bad = s[s.status != "ok"]
            if len(bad) == len(s):
                lines.append(f"  {m:12s} {bad.status.iloc[0]}")
                continue
            g = ok[(ok.n_columns == n_cols) & (ok.method == m)]
            parts = [f"{c}={g[c].mean():.4f}" + (f"±{g[c].std():.4f}" if len(g) > 1 else "")
                     for c in num]
            lines.append(f"  {m:12s} " + "  ".join(parts))
    text = "\n".join(lines)
    open("scal_results_summary.txt", "w", encoding="utf8").write(text)
    print("\n" + text)


if __name__ == "__main__":
    mp.freeze_support()
    log = None
    try:
        log = setup()
        main()
        print("\nSELESAI. Kirimkan scal_results_summary.txt, scal_results_per_run.csv, "
              "dan scal_experiment_log.txt.")
    except Exception:
        print("\nTERJADI ERROR:\n" + traceback.format_exc())
        print("Kirimkan file scal_experiment_log.txt untuk diperbaiki.")
    sys.stdout = sys.__stdout__
    try:
        input("\nTekan Enter untuk menutup jendela...")
    except Exception:
        pass
