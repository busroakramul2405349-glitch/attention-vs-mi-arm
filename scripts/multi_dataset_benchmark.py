"""
Multi-dataset benchmark: does attention select better antecedents than simple dependence measures?

Datasets: the nine datasets of PyAerial's own rule-quality benchmark (data/pyaerial_benchmark/, prepared exactly as
PyAerial does: drop rows with missing values, five equal-frequency bins for numerical columns, last column = class)
plus, optionally, the PyAerial gene-expression benchmark (first GENE_COLS genes).

Part A - unsupervised, equal-budget guide ablation (every dataset):
    For every consequent feature i a guide chooses TOP_ANT antecedent features; candidates (single items and item
    pairs from the chosen features) and filters are identical for all guides.
    Guides: Attention (VATTEN), AttnBottom (lowest attention), Random, MutualInfo, CramerV.
    Reference (no budget): FP-Growth (MLxtend) and PyAerial with the settings of PyAerial's rule-quality benchmark.
Part B - supervised, class rules (the nine benchmark datasets):
    Stratified 56/14/30 split per seed; each guide picks TOP_K_FEATURES features for the class; candidates ->
    filter -> optional RL pruning -> the same CBA-style rule-list classifier for all methods.
    Baselines: FP-Growth and PyAerial class rules.

Outputs: md_unsup_per_run.csv, md_sup_per_run.csv, md_datasets.csv, md_summary.txt, md_experiment_log.txt
Statistical tests: run scripts/statistical_tests.py afterwards.
"""
import os
import sys
import time
import json
import random
import hashlib
import platform
import subprocess
import traceback
import importlib.metadata as md
from itertools import combinations

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASETS = ["acute_inflammations", "autism_screening_adolescent", "breast_cancer",
            "cervical_cancer_behavior_risk", "chess_king_rook_vs_king_pawn", "congressional_voting",
            "fertility", "hepatitis", "mushroom"]
INCLUDE_GENE = True          # gene-expression benchmark in Part A
GENE_COLS = 1000
GENE_MIN_SUPPORT = 0.5120    # PyAerial benchmark protocol for 1,000 genes (see Table 3 of the manuscript)
SEEDS = [0, 1, 2, 3, 4]
RUN_SUPERVISED = True
RUN_BASELINES = True

# Part A
MIN_SUPPORT = 0.10           # = min_rule_frequency of PyAerial's rule-quality benchmark
MIN_CONF = 0.80              # = min_rule_strength of PyAerial's rule-quality benchmark
TOP_ANT = 2
STRATEGIES = ["Attention", "AttnBottom", "Random", "MutualInfo", "CramerV"]

# Part B
SUP_MIN_SUPPORT = 0.05
TOP_K_FEATURES = 6
SUP_GUIDES = ["Attention", "MutualInfo", "CramerV", "Random"]

# Baselines (settings of benchmark/rule_quality.py in PyAerial v1.1.0)
PYAERIAL_EPOCHS = 10
PYAERIAL_MIN_RULE_FREQUENCY = 0.1
PYAERIAL_MIN_RULE_STRENGTH = 0.8
MAX_ANT = 2

# VATTEN (same model as the gene ablation)
D_MODEL = 64
N_HEADS = 4
V_EPOCHS = 50
LR = 3e-3
BETA_KL = 1e-3

# RL (same as the Mushroom experiment)
RL_EPISODES = 1000
RL_SAMPLES = 16
RL_LR = 0.1
LAMBDA1 = 1.0
LAMBDA2 = 0.05

GENE_FILE = "cell_cancertype_NonSmallCellLungCarcinoma_lnic50.csv"
DATA_DIRS = [".", "pyaerial_benchmark", os.path.join("..", "data", "pyaerial_benchmark"),
             os.path.join("data", "pyaerial_benchmark")]
GENE_DIRS = [".", os.path.join("..", "data"), r"D:\NS ARM BARU VS\data", r"D:\NS ARM BARU VS"]
SHA256 = {
    "acute_inflammations": "7a992356eac452f8835866e3ffe65c9163cac14015f58fca3b5251d899e5af5d",
    "autism_screening_adolescent": "19dc0851e0b10063803458c4889bbd27f895fba3b295c73ee011136c90439d05",
    "breast_cancer": "68fc2ca428d0a05449a4136883850748fc234aa8ad3f94e978a4dd6363ec1225",
    "cervical_cancer_behavior_risk": "d8b207ff86fe805796c286b95018cfbfaf776e7e7571a3b9e27a42a885acbe67",
    "chess_king_rook_vs_king_pawn": "4e771d74e1253c9352ce1002d245bb2bf5ee1b11a0e06b6bcf89d6c25c2dd522",
    "congressional_voting": "ff6def8ff72ba647dc7bc7499330bdeac426afb9c0c48fb48f1582c8886a4f54",
    "fertility": "3e18d8d16b6381493aee95ee5c730e28366dfd732e38dd6a1feb47e8a05e1ea4",
    "hepatitis": "8f1f69fa2c6a6528cce9a18a8d5f319a330dd899f2461fee500cf5ccc5ba0c7e",
    "mushroom": "0479d653aaaed91545a90364ddd8604f3fb64bd5db3b9d648fe81ab62f9bf887",
    "gene": "1a8ed010cba48820024e21c604ef5e05af65325fe292bb4f8ce081a820fb6525",
}
LOG_FILE = "md_experiment_log.txt"

np = pd = None  # imported in main()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find(name, dirs):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def equal_frequency_discretization(df, n_bins=5):
    """Copy of aerial.discretization.equal_frequency_discretization (PyAerial v1.1.0)."""
    out = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        n_unique, n_total = df[col].nunique(), len(df[col].dropna())
        if n_total > 0 and n_unique > n_bins:
            try:
                out[col] = pd.qcut(df[col], q=n_bins, duplicates="drop").astype(str)
            except ValueError:
                pass
        else:
            out[col] = out[col].astype(str)
    return out


def prepare(path):
    """Same preparation as PyAerial benchmark/rule_quality.py: dropna, then discretise numerical columns."""
    df = pd.read_csv(path).dropna()
    if df.select_dtypes(include="number").shape[1] > 0:
        df = equal_frequency_discretization(df)
    df = df.astype(str).reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def encode(df):
    cols = list(df.columns)
    cats = [sorted(df[c].unique()) for c in cols]
    cards = [len(c) for c in cats]
    codes = np.stack([df[c].map({v: i for i, v in enumerate(cats[k])}).values
                      for k, c in enumerate(cols)], 1).astype(np.int64)
    return codes, cards, cats


def onehot_padded(codes, cmax):
    n, f = codes.shape
    X = np.zeros((n, f, cmax), dtype=bool)
    X[np.arange(n)[:, None], np.arange(f)[None, :], codes] = True
    return X


# ---------------------------------------------------------------------------
# Guides
# ---------------------------------------------------------------------------
def train_attention(codes, cards, seed, epochs=V_EPOCHS, d_model=D_MODEL, n_heads=N_HEADS, beta=BETA_KL,
                    batch=None):
    """VATTEN with diagonal-masked self-attention; returns the attention matrix A[consequent, antecedent]."""
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
            self.register_buffer("diag", torch.eye(n_feat, dtype=torch.bool))

        def forward(self, x):
            tok = self.val_emb(x + self.offset) + self.feat_emb
            mu, logvar = self.to_mu(tok), self.to_logvar(tok)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
            a, w = self.attn(z, z, z, attn_mask=self.diag, need_weights=True, average_attn_weights=True)
            h = self.norm(a + self.feat_emb)
            logits = torch.einsum("bnd,ndc->bnc", h, self.W) + self.b
            return logits.masked_fill(~self.valid, -1e9), mu, logvar, w

    if batch is None:
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
            rec = F.cross_entropy(logits.reshape(-1, cmax), xb.reshape(-1))
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        A = sum(model(X[i:i + batch])[3].sum(0) for i in range(0, len(X), batch)) / len(X)
    A = A.numpy().astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def _joint(codes, cmax):
    n, f = codes.shape
    X = onehot_padded(codes, cmax).reshape(n, f * cmax).astype(np.float32)
    P = (X.T @ X / n).reshape(f, cmax, f, cmax)
    p = X.mean(0).reshape(f, cmax)
    return P, p


def mutual_info_matrix(codes, cmax):
    P, p = _joint(codes, cmax)
    outer = p[:, :, None, None] * p[None, None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(P > 0, P * np.log(P / outer), 0.0)
    MI = term.sum(axis=(1, 3))  # float32 as in the gene ablation script (same tie order)
    np.fill_diagonal(MI, 0.0)
    return MI


def cramers_v_matrix(codes, cmax):
    """Cramer's V between all feature pairs: sqrt(phi^2 / (min(r, c) - 1)), phi^2 = chi^2 / n."""
    P, p = _joint(codes, cmax)
    outer = p[:, :, None, None] * p[None, None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(outer > 0, (P - outer) ** 2 / outer, 0.0)
    phi2 = term.sum(axis=(1, 3)).astype(float)
    r = (p > 0).sum(1)
    k = np.minimum(r[:, None], r[None, :]) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        V = np.where(k > 0, np.sqrt(np.clip(phi2, 0, None) / np.maximum(k, 1)), 0.0)
    np.fill_diagonal(V, 0.0)
    return V


def _top(row, i, k):
    """k largest entries of row, excluding i (same ordering as the gene ablation script)."""
    order = np.argsort(row)[::-1]
    return [int(j) for j in order[order != i][:k]]


def pick(strategy, i, f, k, A, MI, CV, rng):
    others = np.delete(np.arange(f), i)
    if strategy == "Attention":
        return _top(A[i], i, k)
    if strategy == "AttnBottom":
        return [int(j) for j in others[np.argsort(A[i, others])[:k]]]
    if strategy == "MutualInfo":
        return _top(MI[i], i, k)
    if strategy == "CramerV":
        return _top(CV[i], i, k)
    return [int(j) for j in rng.choice(others, k, replace=False)]


# ---------------------------------------------------------------------------
# Budgeted mining and rule metrics
# ---------------------------------------------------------------------------
def mine(X, cards, picks, min_support):
    """Identical candidates and filter for all guides. picks[i] = antecedent features for consequent i."""
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
            sup = cnt[v] / n
            conf = cnt[v] / na
            if sup < min_support or conf < MIN_CONF:
                continue
            rules.append(rule_record(a, (i, v), sup, na / n, p_cons[v], m))
    return rules


def rule_record(ant, cons, sup, pa, pb, mask):
    conf = sup / pa
    lift = conf / pb if pb > 0 else float("nan")
    denom = max(sup * (1 - pa), pa * (pb - sup))
    zhang = (sup - pa * pb) / denom if denom > 0 else 0.0
    return {"ant": ant, "cons": cons, "support": sup, "confidence": conf, "lift": lift, "zhang": zhang,
            "mask": mask}


def measure(X, ant, cons):
    """Metrics of an arbitrary rule (used for FP-Growth and PyAerial rules)."""
    n = X.shape[0]
    m = np.ones(n, dtype=bool)
    for j, v in ant:
        m &= X[:, j, v]
    c = X[:, cons[0], cons[1]]
    pa, pb, sup = m.mean(), c.mean(), (m & c).mean()
    if pa == 0:
        return None
    return rule_record(ant, cons, sup, pa, pb, m)


def summarize(rules, n):
    if not rules:
        return {"rule_count": 0, "average_support": None, "average_confidence": None, "average_lift": None,
                "average_zhang": None, "data_coverage": 0.0, "consequents_explained": 0,
                "share_lift_gt_1_2": None}
    cov = np.zeros(n, dtype=bool)
    seen = set()
    for r in rules:
        key = r["ant"]
        if key not in seen:
            seen.add(key)
            cov |= r["mask"]
    lift = np.array([r["lift"] for r in rules], dtype=float)
    return {"rule_count": len(rules),
            "average_support": float(np.mean([r["support"] for r in rules])),
            "average_confidence": float(np.mean([r["confidence"] for r in rules])),
            "average_lift": float(np.nanmean(lift)),
            "average_zhang": float(np.mean([r["zhang"] for r in rules])),
            "data_coverage": float(cov.mean()),
            "consequents_explained": len({r["cons"][0] for r in rules}),
            "share_lift_gt_1_2": float(np.nanmean(lift > 1.2))}


def fpgrowth_rules(X, cols, cats, min_support):
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    n, f, cmax = X.shape
    names, idx = [], []
    for j in range(f):
        for v in range(len(cats[j])):
            names.append(f"{j}|{v}")
            idx.append((j, v))
    onehot = pd.DataFrame(X.reshape(n, f * cmax)[:, [j * cmax + v for j, v in idx]], columns=names)
    fi = fpgrowth(onehot, min_support=min_support, use_colnames=True, max_len=MAX_ANT + 1)
    if len(fi) == 0:
        return []
    try:
        ar = association_rules(fi, metric="confidence", min_threshold=MIN_CONF)
    except TypeError:
        ar = association_rules(fi, num_itemsets=len(onehot), metric="confidence", min_threshold=MIN_CONF)
    rules = []
    mask_cache = {}
    for a, c, s, sa, sc in zip(ar["antecedents"], ar["consequents"], ar["support"],
                               ar["antecedent support"], ar["consequent support"]):
        if len(c) != 1 or len(a) > MAX_ANT:
            continue
        ant = tuple(sorted(tuple(map(int, x.split("|"))) for x in a))
        cons = tuple(map(int, next(iter(c)).split("|")))
        if ant not in mask_cache:
            m = np.ones(n, dtype=bool)
            for j, v in ant:
                m &= X[:, j, v]
            mask_cache[ant] = m
        rules.append(rule_record(ant, cons, float(s), float(sa), float(sc), mask_cache[ant]))
    return rules


def _aerial_generate(ae, **kw):
    from aerial import rule_extraction
    try:
        res = rule_extraction.generate_rules(ae, min_rule_frequency=PYAERIAL_MIN_RULE_FREQUENCY,
                                             min_rule_strength=PYAERIAL_MIN_RULE_STRENGTH,
                                             max_antecedents=MAX_ANT, **kw)
    except TypeError:  # older PyAerial versions
        res = rule_extraction.generate_rules(ae, ant_similarity=PYAERIAL_MIN_RULE_FREQUENCY,
                                             cons_similarity=PYAERIAL_MIN_RULE_STRENGTH,
                                             max_antecedents=MAX_ANT, **kw)
    if isinstance(res, dict):
        return res.get("rules", []) or [], res.get("statistics", {}) or {}
    return list(res or []), {}


def _seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pyaerial_rules(df, X, cols, cats, seed):
    from aerial import model as aerial_model
    _seed_all(seed)
    ae = aerial_model.train(df, epochs=PYAERIAL_EPOCHS, show_progress=False)
    raw, stats = _aerial_generate(ae)
    col_index = {c: j for j, c in enumerate(cols)}
    val_index = [{v: i for i, v in enumerate(c)} for c in cats]
    rules, skipped = [], 0
    for r in raw:
        try:
            ant = tuple(sorted((col_index[a["feature"]], val_index[col_index[a["feature"]]][str(a["value"])])
                               for a in r["antecedents"]))
            c = r["consequent"]
            cons = (col_index[c["feature"]], val_index[col_index[c["feature"]]][str(c["value"])])
        except (KeyError, TypeError):
            skipped += 1
            continue
        rr = measure(X, ant, cons)
        if rr is not None:
            rules.append(rr)
    return rules, stats, skipped


# ---------------------------------------------------------------------------
# Part B: class rules, RL pruning, rule-list classifier (as in the Mushroom experiment)
# ---------------------------------------------------------------------------
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
    def __init__(self, ants, labels, Xtr, ytr):
        self.ants = list(ants)
        self.labels = np.array(labels, dtype=object)
        sup, conf = [], []
        for a, lab in zip(self.ants, self.labels):
            m = Xtr[:, list(a)].all(1)
            n_ant = m.sum()
            n_both = (m & (ytr == lab)).sum()
            sup.append(n_both / len(ytr))
            conf.append(n_both / n_ant if n_ant else 0.0)
        self.support, self.conf = np.array(sup), np.array(conf)
        self.order = np.lexsort((-self.support, -self.conf)) if self.ants else np.array([], int)

    def __len__(self):
        return len(self.ants)

    def match(self, X):
        if not self.ants:
            return np.zeros((len(X), 0), dtype=bool)
        return np.stack([X[:, list(a)].all(1) for a in self.ants], axis=1)


def evaluate(rs, X, y, default, select=None):
    if len(rs) == 0:
        return float(np.mean(y == default)), 0.0, 0
    sel = np.ones(len(rs), bool) if select is None else select
    order = [i for i in rs.order if sel[i]]
    if not order:
        return float(np.mean(y == default)), 0.0, 0
    Mo = rs.match(X)[:, order]
    hit = Mo.any(1)
    pred = np.where(hit, rs.labels[np.array(order)][Mo.argmax(1)], default)
    return float(np.mean(pred == y)), float(hit.mean()), len(order)


def filter_rules(cands, Xtr, ytr):
    ants, labs, seen = [], [], set()
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
        if cnt.max() / len(ytr) >= SUP_MIN_SUPPORT and cnt.max() / n >= MIN_CONF:
            ants.append(a)
            labs.append(vals[cnt.argmax()])
    return ants, labs


def candidates_from_features(feat_list, space):
    cols = {f: [space.index[it] for it in space.items if it.startswith(f + "=")] for f in feat_list}
    cands = [(i,) for f in feat_list for i in cols[f]]
    for f1, f2 in combinations(feat_list, 2):
        cands += [(i, j) for i in cols[f1] for j in cols[f2]]
    return cands


def rl_prune(rs, Xva, yva, default, seed):
    rng = np.random.default_rng(seed)
    M = len(rs)
    if M == 0:
        return np.zeros(0, bool)
    theta, m_t, v_t = np.zeros(M), np.zeros(M), np.zeros(M)
    match, order = rs.match(Xva), rs.order

    def reward(sel):
        o = [i for i in order if sel[i]]
        if not o:
            return float(np.mean(yva == default)) - 1.0
        Mo = match[:, o]
        hit = Mo.any(1)
        pred = np.where(hit, rs.labels[np.array(o)][Mo.argmax(1)], default)
        return float(np.mean(pred == yva)) + LAMBDA1 * float(hit.mean()) - LAMBDA2 * len(o)

    for ep in range(1, RL_EPISODES + 1):
        p = 1.0 / (1.0 + np.exp(-theta))
        acts = rng.random((RL_SAMPLES, M)) < p
        R = np.array([reward(a) for a in acts])
        grad = ((R - R.mean())[:, None] * (acts - p)).mean(0)
        m_t = 0.9 * m_t + 0.1 * grad
        v_t = 0.999 * v_t + 0.001 * grad ** 2
        theta += RL_LR * (m_t / (1 - 0.9 ** ep)) / (np.sqrt(v_t / (1 - 0.999 ** ep)) + 1e-8)
    sel = theta > 0
    if not sel.any():
        sel[np.argmax(theta)] = True
    return sel


def dependence_with_class(dtr, feats, target):
    """MI (nats) and Cramer's V of every feature with the class on the training split."""
    y = dtr[target].values
    mi, cv = {}, {}
    for f in feats:
        ct = pd.crosstab(dtr[f].values, y).values.astype(float)
        P = ct / ct.sum()
        px, py = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
        E = px * py
        with np.errstate(divide="ignore", invalid="ignore"):
            mi[f] = float(np.nansum(np.where(P > 0, P * np.log(P / E), 0.0)))
            phi2 = float(np.nansum(np.where(E > 0, (P - E) ** 2 / E, 0.0)))
        k = min(ct.shape) - 1
        cv[f] = float(np.sqrt(phi2 / k)) if k > 0 else 0.0
    return mi, cv


def class_rules_fpgrowth(space, Xtr, ytr, target):
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    onehot = pd.DataFrame(Xtr, columns=space.items)
    classes = list(np.unique(ytr))
    for c in classes:
        onehot[f"{target}={c}"] = ytr == c
    fi = fpgrowth(onehot, min_support=SUP_MIN_SUPPORT, use_colnames=True, max_len=MAX_ANT + 1)
    if len(fi) == 0:
        return RuleSet([], [], Xtr, ytr), 0
    try:
        ar = association_rules(fi, metric="confidence", min_threshold=MIN_CONF)
    except TypeError:
        ar = association_rules(fi, num_itemsets=len(onehot), metric="confidence", min_threshold=MIN_CONF)
    cls_items = {f"{target}={c}": c for c in classes}
    ants, labs = [], []
    for a, c in zip(ar["antecedents"], ar["consequents"]):
        if len(c) == 1 and next(iter(c)) in cls_items and not (set(a) & set(cls_items)):
            ants.append(tuple(sorted(space.index[i] for i in a)))
            labs.append(cls_items[next(iter(c))])
    return RuleSet(ants, labs, Xtr, ytr), len(ar)


def class_rules_pyaerial(dtr, space, Xtr, ytr, target, seed):
    from aerial import model as aerial_model
    _seed_all(seed)
    ae = aerial_model.train(dtr, epochs=PYAERIAL_EPOCHS, show_progress=False)
    raw, _ = _aerial_generate(ae, target_classes=[target])
    ants, labs = [], []
    for r in raw:
        con = r.get("consequent", {})
        if con.get("feature") != target:
            continue
        try:
            a = tuple(sorted(space.index[f"{x['feature']}={x['value']}"] for x in r.get("antecedents", [])))
        except KeyError:
            continue
        if a:
            ants.append(a)
            labs.append(str(con.get("value")))
    return RuleSet(ants, labs, Xtr, ytr)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def load_datasets():
    out = []
    for name in DATASETS:
        p = find(name + ".csv", DATA_DIRS)
        if p is None:
            raise FileNotFoundError(f"{name}.csv not found (expected in data/pyaerial_benchmark/)")
        h = sha256(p)
        df = prepare(p)
        out.append({"name": name, "path": p, "sha256": h, "sha_ok": h == SHA256[name], "df": df,
                    "min_support": MIN_SUPPORT, "target": df.columns[-1], "supervised": True})
    if INCLUDE_GENE:
        p = find(GENE_FILE, GENE_DIRS)
        if p is None:
            print(f"[warning] {GENE_FILE} not found; the gene benchmark is skipped.")
        else:
            h = sha256(p)
            df = pd.read_csv(p).astype(str).iloc[:, :GENE_COLS]
            out.append({"name": f"gene_{GENE_COLS}", "path": p, "sha256": h, "sha_ok": h == SHA256["gene"],
                        "df": df, "min_support": GENE_MIN_SUPPORT, "target": None, "supervised": False})
    return out


def part_a(ds, rows):
    df = ds["df"]
    codes, cards, cats = encode(df)
    n, f = codes.shape
    X = onehot_padded(codes, max(cards))
    t = time.perf_counter()
    MI = mutual_info_matrix(codes, max(cards))
    t_mi = time.perf_counter() - t
    t = time.perf_counter()
    CV = cramers_v_matrix(codes, max(cards))
    t_cv = time.perf_counter() - t
    print(f"\n=== [A] {ds['name']}: {n} rows x {f} features, min_support={ds['min_support']} "
          f"(MI {t_mi:.2f}s, Cramer's V {t_cv:.2f}s) ===")

    def add(method, seed, rules, sel_time, mine_time, extra=None):
        s = summarize(rules, n)
        s.update(dataset=ds["name"], method=method, seed=seed, selection_time_s=round(sel_time, 3),
                 mining_time_s=round(mine_time, 3), budgeted=method in STRATEGIES)
        if extra:
            s.update(extra)
        rows.append(s)
        pd.DataFrame(rows).to_csv("md_unsup_per_run.csv", index=False)
        print(f"  seed {seed} {method:11s} rules={s['rule_count']:7d} lift={s['average_lift'] or 0:.3f} "
              f"zhang={s['average_zhang'] or 0:.3f} conf={s['average_confidence'] or 0:.3f} "
              f"cov={s['data_coverage']:.3f} cons={s['consequents_explained']}")

    if RUN_BASELINES and ds["supervised"]:
        try:
            t = time.perf_counter()
            rules = fpgrowth_rules(X, list(df.columns), cats, ds["min_support"])
            add("FP-Growth", -1, rules, 0.0, time.perf_counter() - t)
        except Exception:
            print("  FP-Growth failed:\n" + traceback.format_exc())

    for seed in SEEDS:
        t = time.perf_counter()
        A = train_attention(codes, cards, seed)
        t_attn = time.perf_counter() - t
        rng = np.random.default_rng(1000 + seed)
        sel_time = {"Attention": t_attn, "AttnBottom": t_attn, "Random": 0.0, "MutualInfo": t_mi,
                    "CramerV": t_cv}
        top = {}
        for strat in STRATEGIES:
            t = time.perf_counter()
            picks = [pick(strat, i, f, TOP_ANT, A, MI, CV, rng) for i in range(f)]
            rules = mine(X, cards, picks, ds["min_support"])
            top[strat] = [set(p) for p in picks]
            add(strat, seed, rules, sel_time[strat], time.perf_counter() - t)
        ov = {o: float(np.mean([len(a & b) / TOP_ANT for a, b in zip(top["Attention"], top[o])]))
              for o in ["MutualInfo", "CramerV", "Random"]}
        rows[-1]["overlap_attention_mi"] = ov["MutualInfo"]
        print(f"  seed {seed}: overlap of chosen features, Attention vs MI {ov['MutualInfo']:.3f}, "
              f"vs Cramer's V {ov['CramerV']:.3f}, vs Random {ov['Random']:.3f}")
        if RUN_BASELINES and ds["supervised"] and HAS_PYAERIAL:
            try:
                t = time.perf_counter()
                rules, stats, skipped = pyaerial_rules(df, X, list(df.columns), cats, seed)
                add("PyAerial", seed, rules, 0.0, time.perf_counter() - t,
                    {"pyaerial_reported_zhang": stats.get("average_zhangs_metric"),
                     "pyaerial_reported_rules": stats.get("rule_count"), "pyaerial_unmapped_rules": skipped})
            except Exception:
                print("  PyAerial failed:\n" + traceback.format_exc())


def part_b(ds, rows):
    df, target = ds["df"], ds["target"]
    feats = [c for c in df.columns if c != target]
    k = min(TOP_K_FEATURES, len(feats))
    y = df[target].values
    print(f"\n=== [B] {ds['name']}: class '{target}' {dict(zip(*np.unique(y, return_counts=True)))}, "
          f"K={k} features ===")
    for seed in SEEDS:
        tr, va, te = stratified_split(y, seed)
        dtr, dva, dte = df.iloc[tr], df.iloc[va], df.iloc[te]
        space = ItemSpace(dtr, feats)
        Xtr, Xva, Xte = space.matrix(dtr), space.matrix(dva), space.matrix(dte)
        ytr, yva, yte = y[tr], y[va], y[te]
        default = pd.Series(ytr).mode()[0]

        def record(name, rs, runtime, sel=None, n_all=None, chosen=None):
            acc, cov, kk = evaluate(rs, Xte, yte, default, sel)
            idx = np.arange(len(rs)) if sel is None else np.where(sel)[0]
            rows.append({"dataset": ds["name"], "seed": seed, "method": name, "n_rules": kk,
                         "n_all_association_rules": n_all, "test_accuracy": round(acc, 4),
                         "test_coverage": round(cov, 4),
                         "avg_support": round(float(rs.support[idx].mean()), 4) if kk else None,
                         "avg_confidence": round(float(rs.conf[idx].mean()), 4) if kk else None,
                         "runtime_s": round(runtime, 3), "n_test": len(yte),
                         "chosen_features": "|".join(chosen) if chosen else None})
            pd.DataFrame(rows).to_csv("md_sup_per_run.csv", index=False)
            print(f"  seed {seed} {name:16s} rules={kk:5d} acc={acc:.4f} cov={cov:.4f} time={runtime:.2f}s")

        if RUN_BASELINES:
            try:
                t = time.perf_counter()
                rs, n_all = class_rules_fpgrowth(space, Xtr, ytr, target)
                record("FP-Growth", rs, time.perf_counter() - t, n_all=n_all)
            except Exception:
                print("  FP-Growth failed:\n" + traceback.format_exc())
            if HAS_PYAERIAL:
                try:
                    t = time.perf_counter()
                    rs = class_rules_pyaerial(dtr, space, Xtr, ytr, target, seed)
                    record("PyAerial", rs, time.perf_counter() - t)
                except Exception:
                    print("  PyAerial failed:\n" + traceback.format_exc())

        t0 = time.perf_counter()
        all_cols = feats + [target]
        cats = {c: sorted(dtr[c].unique()) for c in all_cols}
        codes = np.stack([dtr[c].map({v: i for i, v in enumerate(cats[c])}).astype(int).values
                          for c in all_cols], 1)
        A = train_attention(codes, [len(cats[c]) for c in all_cols], seed)
        t_attn = time.perf_counter() - t0
        cls_row = A[len(feats), :len(feats)]
        t0 = time.perf_counter()
        mi, cv = dependence_with_class(dtr, feats, target)
        t_dep = time.perf_counter() - t0
        rng = np.random.default_rng(seed + 100)
        chosen = {"Attention": [feats[i] for i in np.argsort(-cls_row, kind="stable")[:k]],
                  "MutualInfo": sorted(feats, key=lambda f: -mi[f])[:k],
                  "CramerV": sorted(feats, key=lambda f: -cv[f])[:k],
                  "Random": list(rng.choice(feats, k, replace=False))}
        guide_time = {"Attention": t_attn, "MutualInfo": t_dep, "CramerV": t_dep, "Random": 0.0}
        for g in SUP_GUIDES:
            t = time.perf_counter()
            ants, labs = filter_rules(candidates_from_features(chosen[g], space), Xtr, ytr)
            cand = RuleSet(ants, labs, Xtr, ytr)
            t_c = guide_time[g] + time.perf_counter() - t
            record(f"{g}-noRL", cand, t_c, chosen=chosen[g])
            t = time.perf_counter()
            sel = rl_prune(cand, Xva, yva, default, seed)
            record(f"{g}+RL", cand, t_c + time.perf_counter() - t, sel=sel, chosen=chosen[g])


def summary_text(unsup, sup, dsinfo):
    lines = [f"Multi-dataset benchmark | seeds={SEEDS} | Part A: k={TOP_ANT} antecedent features per consequent, "
             f"min_support={MIN_SUPPORT} (gene: {GENE_MIN_SUPPORT}), min_conf={MIN_CONF} | Part B: K={TOP_K_FEATURES} "
             f"features for the class, min_support={SUP_MIN_SUPPORT}", ""]
    lines.append("Datasets (after PyAerial preparation):")
    for d in dsinfo:
        lines.append(f"  {d['dataset']:32s} rows={d['rows']:5d} features={d['features']:4d} items={d['items']:5d} "
                     f"sha256_ok={d['sha256_ok']}")
    if len(unsup):
        u = unsup.copy()
        lines += ["", "[A] Mean over seeds (equal budget for the five guides; FP-Growth/PyAerial unbudgeted):"]
        piv = {}
        for c in ["average_zhang", "average_lift", "consequents_explained", "rule_count", "data_coverage"]:
            piv[c] = u.pivot_table(index="dataset", columns="method", values=c, aggfunc="mean")
        for c, t in piv.items():
            lines.append(f"\n  {c}")
            lines.append("  " + t.round(4).to_string().replace("\n", "\n  "))
        z = piv["average_zhang"]
        lines.append("")
        for a, b in [("MutualInfo", "Attention"), ("CramerV", "Attention"), ("Attention", "Random"),
                     ("Attention", "AttnBottom"), ("MutualInfo", "CramerV")]:
            if a in z and b in z:
                d = (z[a] - z[b]).dropna()
                lines.append(f"  Zhang: {a} > {b} on {int((d > 0).sum())}/{len(d)} datasets "
                             f"(mean difference {d.mean():+.4f})")
    if len(sup):
        s = sup.copy()
        lines += ["", "[B] Test accuracy, mean over seeds:"]
        t = s.pivot_table(index="dataset", columns="method", values="test_accuracy", aggfunc="mean")
        lines.append("  " + t.round(4).to_string().replace("\n", "\n  "))
        t = s.pivot_table(index="dataset", columns="method", values="n_rules", aggfunc="mean")
        lines += ["", "[B] Number of rules, mean over seeds:", "  " + t.round(1).to_string().replace("\n", "\n  ")]
    lines += ["", "Statistical tests: run statistical_tests.py on md_unsup_per_run.csv and md_sup_per_run.csv."]
    return "\n".join(lines)


def main():
    print(f"Python {platform.python_version()} | {platform.platform()}")
    for p in ["numpy", "pandas", "torch", "mlxtend", "pyaerial", "scipy"]:
        try:
            print(f"  {p} {md.version(p)}")
        except Exception:
            print(f"  {p} not available")
    datasets = load_datasets()
    dsinfo = []
    for d in datasets:
        codes, cards, _ = encode(d["df"])
        dsinfo.append({"dataset": d["name"], "rows": len(d["df"]), "features": len(cards), "items": int(sum(cards)),
                       "target": d["target"], "min_support_unsup": d["min_support"], "sha256": d["sha256"],
                       "sha256_ok": d["sha_ok"], "file": d["path"]})
        print(f"{d['name']:32s} {len(d['df']):5d} rows  {len(cards):4d} features  sha256 ok={d['sha_ok']}")
    pd.DataFrame(dsinfo).to_csv("md_datasets.csv", index=False)

    unsup, sup = [], []
    for d in datasets:
        part_a(d, unsup)
    if RUN_SUPERVISED:
        for d in datasets:
            if d["supervised"]:
                part_b(d, sup)
    U = pd.DataFrame(unsup)
    S = pd.DataFrame(sup)
    U.to_csv("md_unsup_per_run.csv", index=False)
    if len(S):
        S.to_csv("md_sup_per_run.csv", index=False)
    text = summary_text(U, S, dsinfo)
    with open("md_summary.txt", "w", encoding="utf8") as f:
        f.write(text)
    with open("md_config.json", "w", encoding="utf8") as f:
        json.dump({k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, list, bool))},
                  f, indent=2)
    print("\n" + text)


HAS_PYAERIAL = False

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
        req = {"numpy": "numpy", "pandas": "pandas", "torch": "torch", "mlxtend": "mlxtend", "scipy": "scipy"}
        missing = []
        for mod, pkg in req.items():
            try:
                __import__(mod)
            except Exception:
                missing.append(pkg)
        if missing:
            print(f"Installing missing libraries: {', '.join(missing)}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        try:
            __import__("aerial")
            HAS_PYAERIAL = True
        except Exception:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pyaerial"])
                __import__("aerial")
                HAS_PYAERIAL = True
            except Exception:
                print("[warning] PyAerial unavailable; the PyAerial baseline is skipped.")
        import numpy as np
        import pandas as pd
        import logging
        logging.getLogger("aerial").setLevel(logging.ERROR)
        t_start = time.time()
        main()
        print(f"\nSELESAI dalam {(time.time() - t_start) / 60:.1f} menit. Kirimkan md_summary.txt, "
              "md_unsup_per_run.csv, md_sup_per_run.csv, md_datasets.csv dan md_experiment_log.txt.")
    except Exception:
        print("\nTERJADI ERROR:\n" + traceback.format_exc())
        code = 1
    sys.stdout = sys.__stdout__
    try:
        input("\nTekan Enter untuk menutup jendela...")
    except Exception:
        pass
    sys.exit(code)
