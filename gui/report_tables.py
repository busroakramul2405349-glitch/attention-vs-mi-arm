"""Builds the manuscript tables (Tables 1-8) from result files for the GUI tab "7. Tabel".

Every table is computed from CSV/TXT files, never typed in: either from the archived results in results/
(the numbers of the manuscript) or from the latest successful run in runs/.
"""
import os
import re


def _fmt(x, d=3):
    try:
        if x != x:  # NaN
            return "–"
        return f"{x:,.{d}f}"
    except Exception:
        return str(x)


def _ms(s, d=3):
    """mean ± sd (sd omitted when zero or a single value)."""
    s = s.dropna()
    if len(s) == 0:
        return "–"
    m, sd = s.mean(), s.std()
    if len(s) == 1 or sd != sd or abs(sd) < 1e-12:
        return _fmt(m, d)
    return f"{_fmt(m, d)} ± {_fmt(sd, d)}"


def build_tables(find):
    """find(key, filename) -> (path or None, source label)."""
    import pandas as pd
    out = []

    def table(no, title, key, fname, fn, note=""):
        path, label = find(key, fname)
        t = {"no": no, "title": title, "source": label, "file": fname, "note": note, "header": [], "rows": []}
        if path is None:
            t["missing"] = f"Belum ada file {fname}. Jalankan eksperimen terkait terlebih dahulu."
        else:
            try:
                t["header"], t["rows"] = fn(path)
            except Exception as ex:
                t["missing"] = f"Gagal membaca {fname}: {ex}"
        out.append(t)

    # Table 1 --------------------------------------------------------------
    def t1(p):
        d = pd.read_csv(p)
        d["method"] = d["method"].replace({"RandomFeat+RL": "Random+RL"})
        order = ["FP-Growth", "Apriori", "PyAerial", "VATTEN-noRL", "VATTEN-Rule", "MI-noRL", "MI+RL", "Random+RL"]
        rows = []
        for m in [o for o in order if o in set(d.method)] + [m for m in d.method.unique() if m not in order]:
            g = d[d.method == m]
            rows.append([m, _ms(g.n_rules, 1), _ms(g.test_accuracy), _ms(g.test_coverage), _fmt(g.avg_confidence.mean()),
                         _fmt(g.runtime_s.mean(), 2)])
        seeds = sorted(d.seed.unique().tolist())
        return ["Metode", "Class rules", "Akurasi uji", "Coverage uji", "Conf. rata-rata", "Waktu (s)"], rows + \
            [[f"seed: {seeds}", "", "", "", "", ""]]
    table(1, "UCI Mushroom (rata-rata ± simpangan baku antar seed)", "mushroom", "mi_results_per_seed.csv", t1)

    # Table 2 --------------------------------------------------------------
    def t2_files(p_att):
        rows = []
        for guide, path in [("Attention", p_att), ("MI", os.path.join(os.path.dirname(p_att), "mi_rules_MI_seed0.txt"))]:
            if not os.path.isfile(path):
                continue
            txt = open(path, encoding="utf8").read()
            for m in re.finditer(r"IF (.+?) THEN (.+?)\s+\(support=([\d.]+), confidence=([\d.]+)\)", txt):
                ant = m.group(1).replace(" AND ", " ∧ ").replace("=", " = ")
                cons = m.group(2).split("=")[-1]
                cons = {"p": "poisonous", "e": "edible"}.get(cons, cons)
                rows.append([guide, f"{ant} → {cons}", m.group(3), m.group(4)])
        return ["Pemandu", "Rule", "Support", "Confidence"], rows
    table(2, "Rule akhir setelah pemangkasan RL, Mushroom seed 0", "mushroom", "mi_rules_seed0.txt", t2_files)

    # Table 3 --------------------------------------------------------------
    def t3(p):
        d = pd.read_csv(p)
        rows = []
        p_mi, _ = find("migene", "migene_results.csv")
        mi = pd.read_csv(p_mi) if p_mi else None
        sizes = sorted(set(d.n_columns) | (set(mi.n_columns) if mi is not None else set()))
        for n in sizes:
            for m in ["FP-Growth", "PyAerial", "VATTEN-noRL", "VATTEN-Rule"]:
                g = d[(d.n_columns == n) & (d.method == m)]
                if len(g) == 0:
                    continue
                ok = g[g.status == "ok"]
                if len(ok) == 0:
                    st = g.status.iloc[0]
                    rows.append([f"{n:,}", m, st, "–", "–", "–", "> 1,000" if "timeout" in st else "–"])
                    continue
                tag = "" if len(ok) == len(g) else f" ({len(ok)}/{len(g)} run)"
                rows.append([f"{n:,}", m, _fmt(ok.rule_count.mean(), 1 if ok.rule_count.mean() < 10 else 0) + tag,
                             _fmt(ok.average_support.mean()), _fmt(ok.average_confidence.mean()),
                             _fmt(ok.data_coverage.mean(), 2), _fmt(ok.exec_time.mean(), 2)])
            if mi is not None:
                g = mi[(mi.n_columns == n) & (mi.method == "MI-guided")]
                if len(g):
                    rows.append([f"{n:,}", "MI-guided", _fmt(g.rule_count.mean(), 0), _fmt(g.average_support.mean()),
                                 _fmt(g.average_confidence.mean()), _fmt(g.data_coverage.mean(), 2),
                                 _fmt(g.exec_time.mean(), 2)])
        return ["Gen", "Metode", "Rule", "Supp. rata-rata", "Conf. rata-rata", "Coverage", "Waktu (s)"], rows
    table(3, "Benchmark ekspresi gen, semua ukuran (rata-rata antar run)", "scal", "scal_results_per_run.csv", t3,
          "Baris MI-guided diambil dari migene_results.csv (eksperimen 5).")

    # Table 4 --------------------------------------------------------------
    def t4(p):
        d = pd.read_csv(p)
        names = {"MutualInfo": "Mutual information", "Attention": "Attention", "AttnBottom": "Attention terendah",
                 "Random": "Acak"}
        rows = []
        for n in sorted(d.n_columns.unique()):
            for s in ["MutualInfo", "Attention", "AttnBottom", "Random"]:
                g = d[(d.n_columns == n) & (d.strategy == s)]
                if len(g) == 0:
                    continue
                rows.append([f"{n:,}", names[s], _ms(g.rule_count, 1), _fmt(g.average_lift.mean()),
                             _fmt(g.average_zhang.mean()), _fmt(g.consequent_genes_explained.mean(), 1),
                             f"{100 * g.share_lift_gt_1_2.mean():.1f}%", _fmt(g.selection_time_s.mean(), 2)])
        return ["Gen", "Pemandu", "Rule", "Lift", "Zhang", "Gen terjelaskan", "Lift > 1,2", "Waktu pemandu (s)"], rows
    table(4, "Ablasi anggaran sama (2 gen antecedent per konsekuen)", "abl", "abl_results_per_run.csv", t4)

    # Table 5 --------------------------------------------------------------
    def pivot(p, values, methods, d_=3, method_col="method", filt=None):
        d = pd.read_csv(p)
        if filt is not None:
            d = filt(d)
        t = d.pivot_table(index="dataset", columns=method_col, values=values, aggfunc="mean")
        cols = [m for m in methods if m in t.columns]
        rows = []
        for ds, r in t[cols].iterrows():
            best = r.max()
            rows.append([ds] + [(f"**{_fmt(v, d_)}**" if v == best and v == v else _fmt(v, d_)) for v in r.values])
        if len(t) > 1:
            rows.append(["Rata-rata"] + [_fmt(t[c].mean(), d_) for c in cols])
        return ["Dataset"] + cols, rows

    guides = ["MutualInfo", "CramerV", "Attention", "AttnBottom", "Random", "FP-Growth", "PyAerial"]
    table(5, "Multi-dataset (A): Zhang's metric rata-rata pada 10 dataset, anggaran sama untuk 5 pemandu pertama", "multi",
          "md_unsup_per_run.csv", lambda p: pivot(p, "average_zhang", guides),
          "Tebal = terbaik pada baris itu. FP-Growth dan PyAerial tanpa anggaran (pembanding).")

    # Table 6 --------------------------------------------------------------
    sup = ["FP-Growth", "PyAerial", "Attention+RL", "MutualInfo+RL", "CramerV+RL", "Random+RL",
           "Attention-noRL", "MutualInfo-noRL", "CramerV-noRL", "Random-noRL"]
    table(7, "Multi-dataset (B): akurasi uji class rule pada 9 dataset (rata-rata antar seed)", "multi", "md_sup_per_run.csv",
          lambda p: pivot(p, "test_accuracy", sup), "Tebal = terbaik pada baris itu.")

    # Table 7 --------------------------------------------------------------
    def t7(p):
        d = pd.read_csv(p)
        rows = []
        dsets = sorted(d.dataset.unique())
        gl = ["MutualInfo", "Random"] + sorted(g for g in d.guide.unique() if g.startswith("Attention"))
        for k in sorted(d.k.unique()):
            for g in gl:
                s = d[(d.k == k) & (d.guide == g)]
                if len(s) == 0:
                    continue
                vals = [_fmt(s[s.dataset == ds].average_zhang.mean()) for ds in dsets]
                rows.append([f"k = {k}", g] + vals + [_fmt(s.average_zhang.mean())])
        return ["Anggaran", "Pemandu / varian"] + dsets + ["Rata-rata"], rows
    table(8, "Ketahanan attention: Zhang's metric rata-rata per varian", "robust", "rob_per_run.csv", t7)

    # Table 8 --------------------------------------------------------------
    def t8a(p):
        d = pd.read_csv(p)
        rows = []
        for _, r in d.iterrows():
            ranks = {c[5:]: r[c] for c in d.columns if c.startswith("rank_") and r[c] == r[c]}
            rk = ", ".join(f"{m} {v:.2f}" for m, v in sorted(ranks.items(), key=lambda x: x[1]))
            rows.append([r["section"], int(r["n_datasets"]), int(r["n_methods"]), _fmt(r["friedman_chi2"], 2),
                         f"{r['p_value']:.3g}", _fmt(r["nemenyi_cd"], 2), rk])
        return ["Uji", "Dataset", "Metode", "χ² Friedman", "p", "CD Nemenyi", "Peringkat rata-rata (1 = terbaik)"], rows
    table("6a", "Uji Friedman + Nemenyi", "stats", "stats_friedman.csv", t8a)

    def t8b(p):
        d = pd.read_csv(p)
        rows = []
        for _, r in d.iterrows():
            sig = "ya" if r.get("p_holm", 1) < 0.05 else "tidak"
            rows.append([r["section"], f"{r['A']} vs {r['B']}", _fmt(r["mean_A"]), _fmt(r["mean_B"]),
                         f"{int(r['A_wins'])}/{int(r['n'])}", f"{r['p']:.3g}", f"{r.get('p_holm', float('nan')):.3g}",
                         f"{r['rank_biserial']:+.2f}", sig])
        return ["Uji", "Pasangan (A vs B)", "Rata-rata A", "Rata-rata B", "A lebih baik", "p Wilcoxon", "p Holm",
                "r rank-biserial", "Signifikan (α = 0,05)"], rows
    table("6b", "Uji Wilcoxon berpasangan dengan koreksi Holm", "stats", "stats_pairwise.csv", t8b)
    order = ["1", "2", "3", "4", "5", "6a", "6b", "7", "8"]
    out.sort(key=lambda t: order.index(str(t["no"])) if str(t["no"]) in order else 99)
    return out
