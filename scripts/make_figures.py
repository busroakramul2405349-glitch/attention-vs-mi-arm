"""Rebuilds Figures 1-3 of the manuscript from the files in ../results/."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#888"})
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"

# Figure 1: execution time vs number of gene columns
scal = pd.read_csv(os.path.join(RES, "gene_scalability", "scal_results_per_run.csv"))
mig = pd.read_csv(os.path.join(RES, "mi_gene", "migene_results.csv"))
g = scal[scal.status == "ok"].groupby(["method", "n_columns"]).exec_time.mean()
mi = mig[mig.method == "MI-guided"].set_index("n_columns").exec_time
fig, ax = plt.subplots(figsize=(6.3, 3.6))


def line(s, color, marker, label):
    ax.plot(s.index, s.values, color=color, lw=2, marker=marker, ms=6, mec="white", mew=1)
    ax.annotate(label, (s.index[-1], s.values[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8.5)


line(g["FP-Growth"], ORANGE, "s", "FP-Growth")
timeouts = scal[(scal.method == "FP-Growth") & (scal.status != "ok")].n_columns
ax.scatter(timeouts, [1000] * len(timeouts), marker="x", color=ORANGE, s=40, zorder=3)
ax.annotate("timeout >1000 s", (timeouts.max(), 1000), xytext=(6, 8), textcoords="offset points",
            fontsize=7.5, color="#555")
line(g["PyAerial"], BLUE, "o", "PyAerial")
line(g["VATTEN-noRL"], AQUA, "^", "VATTEN-noRL")
line(mi, VIOLET, "D", "MI-guided")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Number of gene columns"); ax.set_ylabel("Execution time (s, log scale)")
ax.set_xticks([50, 100, 300, 1000, 3000, 18108]); ax.set_xticklabels(["50", "100", "300", "1,000", "3,000", "18,108"])
ax.grid(True, color="#e6e6e6", lw=0.6); ax.set_xlim(40, 60000)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_runtime.png"), dpi=300); plt.close()

# Figure 2: equal-budget ablation
abl = pd.read_csv(os.path.join(RES, "ablation", "abl_results_per_run.csv"))
order = ["MutualInfo", "Attention", "AttnBottom", "Random"]
names = {"MutualInfo": "Mutual information", "Attention": "Attention (VATTEN)",
         "AttnBottom": "Lowest attention", "Random": "Random"}
cols = {"MutualInfo": VIOLET, "Attention": AQUA, "AttnBottom": "#9a9a95", "Random": "#c9c8c2"}
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.3))
for ax, met, lab in zip(axs, ["average_zhang", "consequent_genes_explained"],
                        ["(a) Mean Zhang's metric", "(b) Consequent genes explained (log)"]):
    sizes = [100, 300, 1000]; w = 0.2; x = np.arange(3)
    for k, s in enumerate(order):
        d = abl[abl.strategy == s].groupby("n_columns")[met]
        ax.bar(x + (k - 1.5) * w, d.mean().loc[sizes], w * 0.9, color=cols[s],
               yerr=d.std().fillna(0).loc[sizes], capsize=2, error_kw={"lw": 0.8, "ecolor": "#555"},
               label=names[s])
    ax.set_xticks(x); ax.set_xticklabels([f"{n:,} genes" for n in sizes])
    ax.set_title(lab, fontsize=9, loc="left"); ax.grid(axis="y", color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
axs[1].set_yscale("log")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(OUT, "fig2_ablation.png"), dpi=300); plt.close()

# Figure 3: attention received per Mushroom feature (seed 0)
A = pd.read_csv(os.path.join(RES, "mushroom", "mi_attention_seed0.csv"), index_col=0)
n = len(A)
recv = (A.sum(axis=0) / (n - 1)).sort_values()
fig, ax = plt.subplots(figsize=(6.3, 4.2))
colors = [AQUA if f in ("ring-number", "veil-color") else (VIOLET if f == "odor" else "#c9c8c2")
          for f in recv.index]
ax.barh(recv.index, recv.values, color=colors, height=0.7)
ax.axvline(1 / (n - 1), color="#444", ls="--", lw=1)
ax.annotate(f"uniform = 1/{n - 1}", (1 / (n - 1), 0), xytext=(4, -2), textcoords="offset points",
            fontsize=7.5, color="#444")
for f in ["ring-number", "veil-color", "odor"]:
    i = list(recv.index).index(f)
    ax.annotate(f"{recv[f]:.3f}", (recv[f], i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7.5)
ax.set_xlabel("Mean attention received from other tokens (Mushroom, seed 0)")
ax.tick_params(axis="y", labelsize=7.5); ax.grid(axis="x", color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_attention.png"), dpi=300); plt.close()
print("Figures written to", os.path.abspath(OUT))
