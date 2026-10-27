"""
Replikasi baseline PyAerial pada UCI Mushroom untuk naskah VATTEN-Rule.

Tujuan: menghasilkan angka PyAerial yang dapat dipertanggungjawabkan
(jumlah rule, support, confidence, coverage, akurasi, runtime) untuk
menggantikan angka di Tabel 5 / Appendix A naskah.

Cara pakai:
    pip install pyaerial ucimlrepo pandas numpy
    python run_pyaerial_mushroom.py

Output:
    pyaerial_mushroom_results.csv   (satu baris per seed)
    pyaerial_mushroom_summary.txt   (rata-rata +/- simpangan baku, siap kutip)
"""

import inspect
import platform
import time
import importlib.metadata as md

import os
import subprocess
import sys
import traceback

# Folder kerja = folder skrip; log dibuat paling awal agar error import tercatat
os.chdir(os.path.dirname(os.path.abspath(__file__)))
_EARLY_LOG = "pyaerial_mushroom_log.txt"


def _early_fail(msg):
    with open(_EARLY_LOG, "w", encoding="utf8") as f:
        f.write(msg)
    print(msg)
    input("\nTekan Enter untuk menutup jendela...")
    sys.exit(1)


print(f"Python yang dipakai: {sys.executable}")
_REQUIRED = {"numpy": "numpy", "pandas": "pandas", "ucimlrepo": "ucimlrepo", "aerial": "pyaerial"}
_missing = []
for _mod, _pkg in _REQUIRED.items():
    try:
        __import__(_mod)
    except Exception:
        _missing.append(_pkg)

if _missing:
    print(f"Library belum terinstal di Python ini: {', '.join(_missing)}")
    print("Menginstal otomatis (perlu internet, bisa beberapa menit karena PyTorch ikut terpasang)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *_missing])
    except Exception:
        _early_fail("GAGAL menginstal library otomatis.\n\n" + traceback.format_exc()
                    + f"\nCoba jalankan manual di Command Prompt:\n"
                    f'  "{sys.executable}" -m pip install {" ".join(_missing)}\n')

try:
    import numpy as np
    import pandas as pd
    from ucimlrepo import fetch_ucirepo
    from aerial import model, rule_extraction
except Exception:
    _early_fail("GAGAL import library.\n\n" + traceback.format_exc())

# ---------------------------------------------------------------------------
# Pengaturan (mengikuti Table 4 paper PyAerial: 2 epoch, 1 hidden layer dim 10,
# antecedent length 2, ant_similarity 0.5, cons_similarity 0.8)
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 2
LAYER_DIMS = [10]
MAX_ANTECEDENTS = 2
ANT_SIM = 0.5   # = min_rule_frequency di versi pyaerial terbaru
CONS_SIM = 0.8  # = min_rule_strength di versi pyaerial terbaru
TEST_SIZE = 0.3
TARGET = "poisonous"


def call_flex(fn, *args, **kwargs):
    """Panggil fungsi PyAerial; otomatis menyesuaikan nama parameter lama/baru."""
    params = inspect.signature(fn).parameters
    aliases = {"ant_similarity": "min_rule_frequency",
               "cons_similarity": "min_rule_strength"}
    clean = {}
    for k, v in kwargs.items():
        if k in params:
            clean[k] = v
        elif aliases.get(k) in params:
            clean[aliases[k]] = v
        else:
            print(f"  [info] parameter '{k}' tidak dikenal oleh {fn.__name__}, dilewati")
    return fn(*args, **clean)


def parse_items(obj):
    """Ubah representasi item PyAerial menjadi daftar (fitur, nilai)."""
    out = []
    if obj is None:
        return out
    if isinstance(obj, dict) and "feature" in obj:
        out.append((str(obj["feature"]), str(obj.get("value"))))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (list, dict)):
                out += parse_items(v)
            else:
                out.append((str(k), str(v)))
    elif isinstance(obj, (list, tuple, set)):
        for o in obj:
            out += parse_items(o)
    elif isinstance(obj, str):
        sep = "__" if "__" in obj else ("=" if "=" in obj else None)
        if sep:
            f, v = obj.split(sep, 1)
            out.append((f.strip(), v.strip()))
    return out


def rule_parts(rule):
    ant = rule.get("antecedents", rule.get("antecedent"))
    con = rule.get("consequent", rule.get("consequents"))
    return parse_items(ant), parse_items(con)


def stratified_split(df, y, test_size, seed):
    rng = np.random.default_rng(seed)
    test_idx = []
    for cls in y.unique():
        idx = np.where(y.values == cls)[0]
        rng.shuffle(idx)
        test_idx += list(idx[: int(round(len(idx) * test_size))])
    mask = np.zeros(len(df), dtype=bool)
    mask[test_idx] = True
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def matches(row, items):
    return all(str(row.get(f)) == v for f, v in items)


def classify(test_df, class_rules, default_cls):
    """Klasifikasi berbasis rule: pilih rule cocok dengan confidence tertinggi."""
    ranked = sorted(class_rules, key=lambda r: r.get("confidence", 0), reverse=True)
    preds, covered = [], 0
    for _, row in test_df.iterrows():
        rdict = row.to_dict()
        pred = None
        for r in ranked:
            ant, con = rule_parts(r)
            if ant and matches(rdict, ant):
                pred = next((v for f, v in con if f == TARGET), None)
                if pred is not None:
                    break
        if pred is None:
            pred = default_cls
        else:
            covered += 1
        preds.append(pred)
    acc = float(np.mean(np.array(preds) == test_df[TARGET].astype(str).values))
    return acc, covered / len(test_df)


def unpack(result):
    """pyaerial >=1.1 mengembalikan {'rules': [...], 'statistics': {...}}; versi lama list."""
    if isinstance(result, dict):
        return result.get("rules", []) or [], result.get("statistics", {}) or {}
    return list(result or []), {}


def main():
    ver = md.version("pyaerial")
    print(f"pyaerial {ver} | Python {platform.python_version()} | {platform.platform()}")

    uci = fetch_ucirepo(id=73)
    data = pd.concat([uci.data.features, uci.data.targets], axis=1).astype(str)
    data = data.replace({"nan": "?"})
    print(f"Mushroom: {data.shape[0]} baris, {data.shape[1]} kolom")

    rows = []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass
        np.random.seed(seed)

        # (1) Association rules pada seluruh data (sebanding Table 4 PyAerial)
        t0 = time.perf_counter()
        ae_all = call_flex(model.train, data, epochs=EPOCHS, layer_dims=LAYER_DIMS, show_progress=False)
        rules = call_flex(rule_extraction.generate_rules, ae_all,
                          ant_similarity=ANT_SIM, cons_similarity=CONS_SIM,
                          max_antecedents=MAX_ANTECEDENTS)
        runtime = time.perf_counter() - t0
        rules, st = unpack(rules)
        print(f"  association rules: {len(rules)} | runtime {runtime:.2f}s | stats {st}")

        # (2) Class association rules -> akurasi pada data uji (70/30 stratified)
        train_df, test_df = stratified_split(data, data[TARGET], TEST_SIZE, seed)
        ae_tr = call_flex(model.train, train_df, epochs=EPOCHS, layer_dims=LAYER_DIMS, show_progress=False)
        crules = call_flex(rule_extraction.generate_rules, ae_tr,
                           ant_similarity=ANT_SIM, cons_similarity=CONS_SIM,
                           max_antecedents=MAX_ANTECEDENTS, target_classes=[TARGET])
        crules, _ = unpack(crules)
        crules = [r for r in crules
                  if any(f == TARGET for f, _ in rule_parts(r)[1])]
        default_cls = train_df[TARGET].mode()[0]
        acc, test_cov = classify(test_df, crules, default_cls)
        print(f"  class rules: {len(crules)} | akurasi uji {acc:.4f} | coverage uji {test_cov:.4f}")

        rows.append({
            "seed": seed,
            "n_rules": len(rules),
            "support": st.get("average_support"),
            "confidence": st.get("average_confidence"),
            "data_coverage": st.get("data_coverage"),
            "avg_rule_coverage": st.get("average_coverage"),
            "zhangs_metric": st.get("average_zhangs_metric"),
            "runtime_s": round(runtime, 3),
            "n_class_rules": len(crules),
            "test_accuracy": round(acc, 4),
            "test_rule_coverage": round(test_cov, 4),
        })
        if seed == SEEDS[0] and rules:
            print("  contoh rule pertama (cek format):", rules[0])

    res = pd.DataFrame(rows)
    res.to_csv("pyaerial_mushroom_results.csv", index=False)

    lines = [f"PyAerial {ver} on UCI Mushroom (n={len(data)}), seeds={SEEDS}",
             f"Setup: epochs={EPOCHS}, layer_dims={LAYER_DIMS}, max_antecedents={MAX_ANTECEDENTS}, "
             f"ant_similarity={ANT_SIM}, cons_similarity={CONS_SIM}, test_size={TEST_SIZE}",
             ""]
    for col in res.columns[1:]:
        vals = pd.to_numeric(res[col], errors="coerce").dropna()
        if len(vals):
            lines.append(f"{col:20s}: {vals.mean():.4f} +/- {vals.std(ddof=1) if len(vals) > 1 else 0:.4f}")
    summary = "\n".join(lines)
    open("pyaerial_mushroom_summary.txt", "w", encoding="utf8").write(summary)
    print("\n" + summary)


class _Tee:
    """Tulis output ke layar sekaligus ke file log."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


if __name__ == "__main__":
    import os
    import sys
    import traceback

    # Simpan hasil di folder yang sama dengan skrip ini
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    log = open("pyaerial_mushroom_log.txt", "w", encoding="utf8")
    sys.stdout = sys.stderr = _Tee(sys.__stdout__, log)
    try:
        main()
        print("\nSELESAI. Hasil tersimpan di pyaerial_mushroom_results.csv dan pyaerial_mushroom_summary.txt")
    except Exception:
        print("\nTERJADI ERROR:")
        traceback.print_exc()
        print("\nKirimkan isi file pyaerial_mushroom_log.txt untuk diperbaiki.")
    finally:
        log.close()
        sys.stdout = sys.__stdout__
        input("\nTekan Enter untuk menutup jendela...")
