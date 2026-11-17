"""
Statistical tests for the multi-dataset benchmark and the attention robustness study.

Follows Demsar (2006, JMLR 7:1-30) for comparing several methods over several datasets:
  - Friedman test on per-dataset means, average ranks (rank 1 = best),
    Nemenyi critical difference (alpha = 0.05) and a critical-difference diagram;
  - two-sided Wilcoxon signed-rank tests for planned pairs across datasets, Holm-corrected,
    with the matched-pairs rank-biserial correlation as effect size;
  - a secondary Wilcoxon test over all (dataset, seed) pairs, reported for transparency.

Inputs (searched in this folder, then in ../results/multi_dataset and ../results/robustness):
  md_unsup_per_run.csv, md_sup_per_run.csv  (multi_dataset_benchmark.py)
  rob_per_run.csv                           (attention_robustness.py, optional)
Outputs: stats_summary.txt, stats_friedman.csv, stats_pairwise.csv, stats_cd_*.png
"""
import os
import sys
import traceback

LOG_FILE = "stats_log.txt"
ALPHA = 0.05
SEARCH = [".", os.path.join("..", "results", "multi_dataset"), os.path.join("..", "results", "robustness")]

GUIDES = ["Attention", "AttnBottom", "Random", "MutualInfo", "CramerV"]
UNSUP_METRICS = ["average_zhang", "average_lift", "consequents_explained", "rule_count"]
UNSUP_PAIRS = [("MutualInfo", "Attention"), ("CramerV", "Attention"), ("Attention", "Random"),
               ("Attention", "AttnBottom"), ("MutualInfo", "Random"), ("MutualInfo", "CramerV")]
SUP_METHODS = ["FP-Growth", "PyAerial", "Attention+RL", "MutualInfo+RL", "CramerV+RL", "Random+RL"]
SUP_PAIRS = [("MutualInfo+RL", "Attention+RL"), ("CramerV+RL", "Attention+RL"), ("Attention+RL", "Random+RL"),
             ("MutualInfo-noRL", "Attention-noRL"), ("Attention-noRL", "Random-noRL"),
             ("FP-Growth", "MutualInfo+RL"), ("PyAerial", "MutualInfo+RL")]

# Nemenyi critical values q_0.05 (Demsar 2006, Table 5a), indexed by number of methods
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def find(name):
    for d in SEARCH:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def holm(pvals):
    import numpy as np
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def wilcoxon(a, b):
    """Two-sided Wilcoxon signed-rank test; returns (n_nonzero, W+, W-, p, rank-biserial r)."""
    import numpy as np
    from scipy import stats
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    nz = d[d != 0]
    if len(nz) == 0:
        return len(d), 0.0, 0.0, 1.0, 0.0
    r = stats.rankdata(np.abs(nz))
    wp, wm = float(r[nz > 0].sum()), float(r[nz < 0].sum())
    try:
        p = float(stats.wilcoxon(nz, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        p = 1.0
    return len(d), wp, wm, p, (wp - wm) / (wp + wm)


def friedman(table, higher_is_better=True):
    """table: DataFrame datasets x methods (complete rows only)."""
    import numpy as np
    from scipy import stats
    t = table.dropna()
    ranks = (-t if higher_is_better else t).rank(axis=1, method="average")
    avg = ranks.mean().sort_values()
    n, k = t.shape
    if n < 2 or k < 3:
        return t, avg, None, None, None
    stat, p = stats.friedmanchisquare(*[t[c].values for c in t.columns])
    cd = Q05.get(k, None)
    cd = cd * np.sqrt(k * (k + 1) / (6.0 * n)) if cd else None
    return t, avg, float(stat), float(p), cd


def cd_diagram(avg, cd, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    names, ranks = list(avg.index), list(avg.values)
    k = len(names)
    half = (k + 1) // 2
    rows_ = max(half, k - half)
    fig, ax = plt.subplots(figsize=(7.0, 1.3 + 0.32 * rows_))
    ax.set_xlim(-0.6, k + 1.6)
    ax.set_ylim(-0.45 - 0.3 * rows_, 0.75)
    ax.axis("off")
    ax.hlines(0, 1, k, color="black", lw=1)
    for x in range(1, k + 1):
        ax.vlines(x, -0.05, 0.05, color="black", lw=1)
        ax.text(x, 0.1, str(x), ha="center", va="bottom", fontsize=9)
    for i, (nm, r) in enumerate(zip(names, ranks)):
        left = i < half
        y = -0.4 - 0.3 * (i if left else (k - 1 - i))
        xe = 0.7 if left else k + 0.3
        ax.plot([r, r, xe], [0, y, y], color="black", lw=0.8)
        ax.text(xe + (-0.05 if left else 0.05), y, f"{nm} ({r:.2f})", fontsize=9,
                ha="right" if left else "left", va="center")
    if cd:
        ax.plot([1, 1 + cd], [0.5, 0.5], color="black", lw=2)
        ax.vlines([1, 1 + cd], 0.46, 0.54, color="black", lw=1)
        ax.text(1 + cd + 0.08, 0.5, f"CD = {cd:.2f}", ha="left", va="center", fontsize=8)
        groups, sorted_r = [], sorted(zip(ranks, names))
        for i in range(k):
            j = i
            while j + 1 < k and sorted_r[j + 1][0] - sorted_r[i][0] < cd:
                j += 1
            if j > i and not any(g[0] <= i and j <= g[1] for g in groups):
                groups.append((i, j))
        for gi, (i, j) in enumerate(groups):
            yb = -0.12 - 0.07 * gi
            ax.plot([sorted_r[i][0] - 0.03, sorted_r[j][0] + 0.03], [yb, yb], color="black", lw=3)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return True


def main():
    import numpy as np
    import pandas as pd
    lines, fried_rows, pair_rows = [], [], []

    def report(section, table, higher=True, pairs=(), per_run=None, metric=None, method_col="method"):
        t, avg, stat, p, cd = friedman(table, higher)
        lines.append(f"\n  {section}: {len(t)} datasets x {t.shape[1]} methods")
        lines.append("    average rank (1 = best): " + ", ".join(f"{m} {r:.2f}" for m, r in avg.items()))
        if stat is not None:
            lines.append(f"    Friedman chi2 = {stat:.3f}, p = {p:.3g}; Nemenyi CD (alpha 0.05) = {cd:.3f}"
                         if cd else f"    Friedman chi2 = {stat:.3f}, p = {p:.3g}")
            fried_rows.append({"section": section, "n_datasets": len(t), "n_methods": t.shape[1],
                               "friedman_chi2": stat, "p_value": p, "nemenyi_cd": cd,
                               **{f"rank_{m}": r for m, r in avg.items()}})
        res = []
        for a, b in pairs:
            if a in table and b in table:
                tt = table[[a, b]].dropna()
                n, wp, wm, pv, rb = wilcoxon(tt[a], tt[b])
                rb = rb if higher else -rb  # positive = first method better
                wins = int((tt[a] > tt[b]).sum()) if higher else int((tt[a] < tt[b]).sum())
                ties = int((tt[a] == tt[b]).sum())
                row = {"section": section, "A": a, "B": b, "level": "datasets", "n": n,
                       "mean_A": tt[a].mean(), "mean_B": tt[b].mean(), "A_wins": wins, "ties": ties,
                       "W_plus": wp, "W_minus": wm, "p": pv, "rank_biserial": rb}
                if per_run is not None:
                    pa = per_run[(per_run[method_col] == a) & (per_run["seed"] >= 0)]
                    pb = per_run[(per_run[method_col] == b) & (per_run["seed"] >= 0)]
                    j = pd.concat([pa.set_index(["dataset", "seed"])[metric].rename("a"),
                                   pb.set_index(["dataset", "seed"])[metric].rename("b")], axis=1).dropna()
                    if len(j) > len(tt):
                        n2, _, _, p2, rb2 = wilcoxon(j["a"], j["b"])
                        rb2 = rb2 if higher else -rb2
                        row.update(n_runs=n2, p_runs=p2, rank_biserial_runs=rb2)
                res.append(row)
        if res:
            adj = holm([r["p"] for r in res])
            for r, pa_ in zip(res, adj):
                r["p_holm"] = pa_
                extra = (f"; over runs n={r['n_runs']}, p={r['p_runs']:.3g}, r={r['rank_biserial_runs']:+.2f}"
                         if "p_runs" in r else "")
                lines.append(f"    {r['A']} vs {r['B']}: mean {r['mean_A']:.4f} vs {r['mean_B']:.4f}, "
                             f"{r['A']} better on {r['A_wins']}/{r['n']} (ties {r['ties']}), "
                             f"Wilcoxon p = {r['p']:.3g} (Holm {pa_:.3g}), rank-biserial r = "
                             f"{r['rank_biserial']:+.2f}{extra}")
            pair_rows.extend(res)
        return avg, cd

    figs = []
    p_u = find("md_unsup_per_run.csv")
    if p_u:
        U = pd.read_csv(p_u)
        lines.append(f"[A] Equal-budget guide comparison ({p_u})")
        for metric in UNSUP_METRICS:
            T = U[U.method.isin(GUIDES)].pivot_table(index="dataset", columns="method", values=metric,
                                                     aggfunc="mean")
            T = T[[g for g in GUIDES if g in T]]
            # within the same candidate budget, more rules = more candidates that pass the filters
            avg, cd = report(f"A/{metric}", T, True, UNSUP_PAIRS, U[U.method.isin(GUIDES)], metric)
            if metric == "average_zhang" and cd_diagram(avg, cd, "Equal-budget guides: mean Zhang's metric",
                                                        "stats_cd_zhang.png"):
                figs.append("stats_cd_zhang.png")
        # unbudgeted reference
        if "FP-Growth" in set(U.method) or "PyAerial" in set(U.method):
            T = U.pivot_table(index="dataset", columns="method", values="average_zhang", aggfunc="mean")
            keep = [m for m in ["FP-Growth", "PyAerial", "MutualInfo", "CramerV", "Attention"] if m in T]
            report("A/average_zhang incl. FP-Growth and PyAerial (unbudgeted)", T[keep], True,
                   [("MutualInfo", "PyAerial"), ("MutualInfo", "FP-Growth"), ("Attention", "PyAerial")])
    else:
        lines.append("[A] md_unsup_per_run.csv not found")

    p_s = find("md_sup_per_run.csv")
    if p_s:
        S = pd.read_csv(p_s)
        lines.append(f"\n[B] Supervised class rules ({p_s})")
        T = S.pivot_table(index="dataset", columns="method", values="test_accuracy", aggfunc="mean")
        keep = [m for m in SUP_METHODS if m in T]
        avg, cd = report("B/test_accuracy", T[keep], True, [p for p in SUP_PAIRS if p[0] in T and p[1] in T],
                         S, "test_accuracy")
        if cd_diagram(avg, cd, "Class rules: test accuracy", "stats_cd_accuracy.png"):
            figs.append("stats_cd_accuracy.png")
        T2 = S.pivot_table(index="dataset", columns="method", values="test_coverage", aggfunc="mean")
        report("B/test_coverage", T2[keep], True, [p for p in SUP_PAIRS if p[0] in T2 and p[1] in T2],
               S, "test_coverage")
        T3 = S.pivot_table(index="dataset", columns="method", values="n_rules", aggfunc="mean")
        report("B/n_rules (fewer = better)", T3[keep], False,
               [p for p in SUP_PAIRS if p[0] in T3 and p[1] in T3], S, "n_rules")
    else:
        lines.append("\n[B] md_sup_per_run.csv not found")

    p_r = find("rob_per_run.csv")
    if p_r:
        R = pd.read_csv(p_r)
        lines.append(f"\n[C] Attention robustness ({p_r})")
        R["block"] = R["dataset"] + "|" + R["seed"].astype(str)
        for k_val in sorted(R["k"].unique()):
            Rk = R[R["k"] == k_val]
            T = Rk.pivot_table(index="block", columns="guide", values="average_zhang", aggfunc="mean")
            variants = [c for c in T.columns if c.startswith("Attention")]
            lines.append(f"\n  k = {k_val}: mean Zhang over {T.shape[0]} (dataset, seed) blocks")
            for v in variants + [c for c in ["Random", "MutualInfo"] if c in T]:
                lines.append(f"    {v:28s} {T[v].mean():.4f}")
            pairs = [(v, "Random") for v in variants] + [("MutualInfo", v) for v in variants]
            res = []
            for a, b in pairs:
                if a in T and b in T:
                    tt = T[[a, b]].dropna()
                    n, wp, wm, pv, rb = wilcoxon(tt[a], tt[b])
                    res.append({"section": f"C/k={k_val}", "A": a, "B": b, "level": "dataset-seed blocks",
                                "n": n, "mean_A": tt[a].mean(), "mean_B": tt[b].mean(),
                                "A_wins": int((tt[a] > tt[b]).sum()), "ties": int((tt[a] == tt[b]).sum()),
                                "W_plus": wp, "W_minus": wm, "p": pv, "rank_biserial": rb})
            if res:
                for r, pa_ in zip(res, holm([r["p"] for r in res])):
                    r["p_holm"] = pa_
                    lines.append(f"    {r['A']} vs {r['B']}: {r['A_wins']}/{r['n']} better, p = {r['p']:.3g} "
                                 f"(Holm {pa_:.3g}), r = {r['rank_biserial']:+.2f}")
                pair_rows.extend(res)
    else:
        lines.append("\n[C] rob_per_run.csv not found (run attention_robustness.py)")

    lines.append("\nNotes: Friedman/Nemenyi and Wilcoxon over datasets use per-dataset means (Demsar 2006). "
                 "Deterministic guides (MutualInfo, CramerV, FP-Growth) have the same value in every seed. "
                 "Rank-biserial r ranges from -1 to 1; positive values mean the first method of the pair is better "
                 "(for rule counts in part B: has fewer rules).")
    if figs:
        lines.append("Figures: " + ", ".join(figs))
    text = "\n".join(lines)
    pd.DataFrame(fried_rows).to_csv("stats_friedman.csv", index=False)
    pd.DataFrame(pair_rows).to_csv("stats_pairwise.csv", index=False)
    with open("stats_summary.txt", "w", encoding="utf8") as f:
        f.write(text)
    print(text)


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
        for mod in ["numpy", "pandas", "scipy", "matplotlib"]:
            try:
                __import__(mod)
            except Exception:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", mod])
        main()
        print("\nSELESAI. Kirimkan stats_summary.txt, stats_pairwise.csv, stats_friedman.csv dan gambar stats_cd_*.png.")
    except Exception:
        print("\nTERJADI ERROR:\n" + traceback.format_exc())
        code = 1
    sys.stdout = sys.__stdout__
    try:
        input("\nTekan Enter untuk menutup jendela...")
    except Exception:
        pass
    sys.exit(code)
