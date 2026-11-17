"""
ARM Experiment Runner - GUI lokal untuk menjalankan seluruh eksperimen naskah
"Does Attention Help Neuro-Symbolic Association Rule Mining?" secara nyata.

- Hanya memakai pustaka standar Python (tidak perlu instal apa pun untuk GUI-nya).
- Server hanya mendengarkan di 127.0.0.1 (tidak bisa diakses dari komputer lain).
- Skrip eksperimen di folder scripts/ TIDAK diubah. Setiap run memakai SALINAN skrip di
  folder runs/<waktu>_<eksperimen>_<mode>/; pada mode cepat hanya konstanta jumlah seed /
  kolom yang diganti, dan perubahannya dicatat di run_info.json.

Jalankan:  python gui/app_server.py      (atau klik dua kali JALANKAN_GUI.bat)
"""
import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

GUI_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(GUI_DIR)
SCRIPTS = os.path.join(ROOT, "scripts")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
RUNS = os.path.join(ROOT, "runs")
DATA = os.path.join(ROOT, "data")
SETTINGS_FILE = os.path.join(GUI_DIR, "settings.json")
PORT = int(os.environ.get("ARM_GUI_PORT", "8765"))

GENE_FILE = "cell_cancertype_NonSmallCellLungCarcinoma_lnic50.csv"
GENE_SHA256 = "1a8ed010cba48820024e21c604ef5e05af65325fe292bb4f8ce081a820fb6525"
GENE_SEARCH = [DATA, SCRIPTS, ROOT, r"D:\NS ARM BARU VS\data", os.path.expanduser("~/Downloads")]
MUSHROOM_CACHE = os.path.join(DATA, "mushroom_uci73.csv")

EXPERIMENTS = [
    {
        "key": "pyaerial",
        "no": 1,
        "title": "Replikasi PyAerial pada Mushroom",
        "script": "run_pyaerial_mushroom.py",
        "purpose": "Memeriksa klaim baseline: PyAerial dengan setelan paper aslinya pada UCI Mushroom.",
        "manuscript": "Konteks Bagian 5.1 (dan audit naskah lama)",
        "needs": "mushroom",
        "full": {}, "cepat": {"SEEDS": "[0]"},
        "est_full": "± 1 menit", "est_cepat": "± 20 detik",
        "summary": "pyaerial_mushroom_summary.txt",
        "archive": "pyaerial_replication",
    },
    {
        "key": "mushroom",
        "no": 2,
        "title": "Mushroom: VATTEN-Rule vs MI-guided vs baseline",
        "script": "mi_guided_mushroom_experiment.py",
        "purpose": "FP-Growth, Apriori, PyAerial, VATTEN (noRL & RL), MI (noRL & RL), fitur acak + RL; "
                   "pengklasifikasi rule yang sama untuk semua metode.",
        "manuscript": "Tabel 1, Tabel 2, Gambar 4",
        "needs": "mushroom",
        "full": {}, "cepat": {"SEEDS": "[0]"},
        "est_full": "± 5 menit", "est_cepat": "± 1 menit",
        "summary": "mi_results_summary.txt",
        "archive": "mushroom",
    },
    {
        "key": "scal",
        "no": 3,
        "title": "Skalabilitas data gen: PyAerial vs FP-Growth vs VATTEN",
        "script": "vatten_scalability_experiment.py",
        "purpose": "Protokol benchmark resmi PyAerial; batas waktu 1.000 detik per metode.",
        "manuscript": "Tabel 3, Gambar 2",
        "needs": "gene",
        "full": {}, "cepat": {"COLS": "[50, 100, 200]", "RUNS": "1"},
        "est_full": "± 1,5 jam", "est_cepat": "± 3 menit",
        "summary": "scal_results_summary.txt",
        "archive": "gene_scalability",
    },
    {
        "key": "abl",
        "no": 4,
        "title": "Ablasi: Attention vs acak vs attention terendah vs MI",
        "script": "vatten_ablation_experiment.py",
        "purpose": "Anggaran sama (2 gen antecedent per konsekuen); menguji apakah attention bermakna.",
        "manuscript": "Tabel 4, Gambar 3",
        "needs": "gene",
        "full": {}, "cepat": {"COLS": "[100]", "SEEDS": "[0]"},
        "est_full": "± 10 menit", "est_cepat": "± 30 detik",
        "summary": "abl_results_summary.txt",
        "archive": "ablation",
    },
    {
        "key": "migene",
        "no": 5,
        "title": "MI-guided sampai seluruh 18.108 gen",
        "script": "mi_guided_gene_scalability.py",
        "purpose": "Baseline mutual information tanpa neural network, dibandingkan pemilihan acak.",
        "manuscript": "Tabel 3, Gambar 2",
        "needs": "gene",
        "full": {}, "cepat": {"COLS": "[100, 1000, 18108]", "RANDOM_SEEDS": "[0]"},
        "est_full": "± 2 menit", "est_cepat": "± 1 menit",
        "summary": "migene_summary.txt",
        "archive": "mi_gene",
    },
    {
        "key": "multi",
        "no": 6,
        "title": "Multi-dataset: 9 dataset benchmark PyAerial + data gen",
        "script": "multi_dataset_benchmark.py",
        "purpose": "Semua dataset benchmark PyAerial + data gen 1.000 kolom. (A) ablasi anggaran sama: attention, "
                   "attention terendah, acak, MI, Cramer's V; (B) class rule + RL + pengklasifikasi yang sama; "
                   "pembanding FP-Growth dan PyAerial.",
        "manuscript": "Bagian 5.5-5.6: Tabel 5, 7; Gambar 5",
        "needs": "benchmark",
        "full": {},
        "cepat": {"DATASETS": '["acute_inflammations", "breast_cancer", "congressional_voting", "fertility"]',
                 "SEEDS": "[0]", "INCLUDE_GENE": "False"},
        "est_full": "\u00b1 45 menit", "est_cepat": "\u00b1 2 menit",
        "summary": "md_summary.txt",
        "archive": "multi_dataset",
    },
    {
        "key": "robust",
        "no": 7,
        "title": "Ketahanan attention: variasi model, sink token, k = 1/2/3",
        "script": "attention_robustness.py",
        "purpose": "Menguji apakah attention lemah hanya karena setelan: epoch, head, dimensi, beta, tanpa VAE, "
                   "dengan sink token; dibandingkan acak dan MI pada anggaran sama.",
        "manuscript": "Bagian 5.7: Tabel 8",
        "needs": "benchmark",
        "full": {},
        "cepat": {"DATASETS": '["congressional_voting"]', "SEEDS": "[0]",
                 "VARIANTS": '["base", "noVAE", "sink"]'},
        "est_full": "\u00b1 45 menit", "est_cepat": "\u00b1 1 menit",
        "summary": "rob_summary.txt",
        "archive": "robustness",
    },
    {
        "key": "stats",
        "no": 8,
        "title": "Uji statistik: Friedman + Nemenyi, Wilcoxon (Holm)",
        "script": "statistical_tests.py",
        "purpose": "Memakai hasil run multi-dataset dan robustness terakhir (atau arsip): peringkat rata-rata, "
                   "uji Friedman, critical difference Nemenyi, Wilcoxon berpasangan dengan koreksi Holm, effect size.",
        "manuscript": "Bagian 5.5: Tabel 6, Gambar 5",
        "needs": "stats",
        "full": {}, "cepat": {},
        "est_full": "\u00b1 10 detik", "est_cepat": "\u00b1 10 detik",
        "summary": "stats_summary.txt",
        "archive": "statistics",
    },
]
EXP = {e["key"]: e for e in EXPERIMENTS}


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_settings():
    try:
        with open(SETTINGS_FILE, encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(s):
    with open(SETTINGS_FILE, "w", encoding="utf8") as f:
        json.dump(s, f, indent=2)


_sha_cache = {}


def gene_dataset():
    s = load_settings()
    cands = ([s["gene_path"]] if s.get("gene_path") else []) + [os.path.join(d, GENE_FILE) for d in GENE_SEARCH]
    for p in cands:
        if p and os.path.isfile(p):
            key = (p, os.path.getmtime(p))
            if key not in _sha_cache:
                _sha_cache[key] = sha256(p)
            h = _sha_cache[key]
            return {"found": True, "path": p, "sha256": h, "match": h == GENE_SHA256}
    return {"found": False, "path": None, "sha256": None, "match": False}


def mushroom_status():
    if os.path.isfile(MUSHROOM_CACHE):
        return {"cached": True, "path": MUSHROOM_CACHE, "sha256": sha256(MUSHROOM_CACHE)}
    return {"cached": False, "path": MUSHROOM_CACHE, "sha256": None}


BENCH_DIR = os.path.join(DATA, "pyaerial_benchmark")
BENCH_SHA256 = {
    "acute_inflammations": "7a992356eac452f8835866e3ffe65c9163cac14015f58fca3b5251d899e5af5d",
    "autism_screening_adolescent": "19dc0851e0b10063803458c4889bbd27f895fba3b295c73ee011136c90439d05",
    "breast_cancer": "68fc2ca428d0a05449a4136883850748fc234aa8ad3f94e978a4dd6363ec1225",
    "cervical_cancer_behavior_risk": "d8b207ff86fe805796c286b95018cfbfaf776e7e7571a3b9e27a42a885acbe67",
    "chess_king_rook_vs_king_pawn": "4e771d74e1253c9352ce1002d245bb2bf5ee1b11a0e06b6bcf89d6c25c2dd522",
    "congressional_voting": "ff6def8ff72ba647dc7bc7499330bdeac426afb9c0c48fb48f1582c8886a4f54",
    "fertility": "3e18d8d16b6381493aee95ee5c730e28366dfd732e38dd6a1feb47e8a05e1ea4",
    "hepatitis": "8f1f69fa2c6a6528cce9a18a8d5f319a330dd899f2461fee500cf5ccc5ba0c7e",
    "mushroom": "0479d653aaaed91545a90364ddd8604f3fb64bd5db3b9d648fe81ab62f9bf887",
}


def benchmark_status():
    found, match = 0, 0
    for name, h in BENCH_SHA256.items():
        p = os.path.join(BENCH_DIR, name + ".csv")
        if os.path.isfile(p):
            found += 1
            match += sha256(p) == h
    return {"dir": BENCH_DIR, "found": found, "total": len(BENCH_SHA256), "matched": match,
            "all_match": match == len(BENCH_SHA256)}


def latest_output(key, fname, modes=("full", "cepat", "demo"), archive=True):
    """File hasil dari run 'selesai' terbaru (urutan mode sesuai `modes`), atau dari arsip results/."""
    if os.path.isdir(RUNS) and modes:
        for want in modes:
            for d in sorted(os.listdir(RUNS), reverse=True):
                if not d.endswith(f"_{key}_{want}"):
                    continue
                info = json.loads(read_text(os.path.join(RUNS, d, "run_info.json")) or "{}")
                p = os.path.join(RUNS, d, fname)
                if info.get("status") == "selesai" and os.path.isfile(p):
                    return p
    if archive:
        p = os.path.join(RESULTS, EXP[key]["archive"], fname)
        return p if os.path.isfile(p) else None
    return None


SOURCE_MODES = {"auto": ("full",), "latest": ("full", "cepat", "demo"), "archive": ()}


def find_source(key, fname, pref="auto"):
    p = latest_output(key, fname, SOURCE_MODES.get(pref, ("full",)))
    if p is None:
        return None, "-"
    rel = os.path.relpath(p, ROOT).replace("\\", "/")
    if rel.startswith("results/"):
        return p, f"arsip naskah: {rel}"
    run = rel.split("/")[1] if rel.startswith("runs/") else rel
    return p, f"run: {run}"


FIG_LIST = [
    ("1", "Gambar 1. Kerangka evaluasi (diagram alur metode)", None, "fig0_framework.png"),
    ("2", "Gambar 2. Waktu eksekusi vs jumlah kolom gen", None, "fig1_runtime.png"),
    ("3", "Gambar 3. Ablasi anggaran sama", None, "fig2_ablation.png"),
    ("4", "Gambar 4. Attention yang diterima tiap fitur Mushroom", None, "fig3_attention.png"),
    ("5", "Gambar 5. Diagram critical difference: Zhang's metric, 5 pemandu (10 dataset)", "stats",
     "stats_cd_zhang.png"),
    ("5b", "Pelengkap (tidak di naskah): diagram critical difference akurasi class rule", "stats",
     "stats_cd_accuracy.png"),
]


def figure_list(pref="auto"):
    out = []
    for no, title, key, name in FIG_LIST:
        if key is None:
            p = os.path.join(FIGURES, name)
            ok = os.path.isfile(p)
            src = f"figures/{name}" + ("" if no == "1" else " (dibuat dari results/ oleh make_figures.py)")
            out.append({"no": no, "title": title, "url": f"/figures/{name}" if ok else None, "source": src})
        else:
            p, label = find_source(key, name, pref)
            out.append({"no": no, "title": title, "source": label,
                        "url": f"/runfig?key={key}&name={name}&source={pref}" if p else None,
                        "missing": None if p else "Belum ada. Jalankan eksperimen 6 dan 7, lalu eksperimen 8."})
    return out


def cpu_name():
    if sys.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except Exception:
            pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "tidak diketahui"


def ram_gb():
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.ullTotalPhys / 1024 ** 3, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)
    except Exception:
        return None


def env_info():
    libs = {}
    for p in ["numpy", "pandas", "torch", "mlxtend", "pyaerial", "ucimlrepo", "matplotlib", "scipy"]:
        try:
            libs[p] = md.version(p)
        except Exception:
            libs[p] = None
    return {"python": platform.python_version(), "python_exe": sys.executable,
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "machine": platform.node(), "cpu": cpu_name(), "cores": os.cpu_count(), "ram_gb": ram_gb(),
            "libs": libs, "gene": gene_dataset(), "mushroom": mushroom_status(),
            "benchmark": benchmark_status(), "root": ROOT}


def patch_constants(src, overrides):
    """Ganti konstanta NAMA = ... (juga daftar yang ditulis lebih dari satu baris)."""
    applied = {}
    for name, value in overrides.items():
        pat = re.compile(rf"^{name}\s*=\s*.*$", re.M)
        m = pat.search(src)
        if not m:
            raise ValueError(f"Konstanta {name} tidak ditemukan di skrip")
        start, end = m.start(), m.end()
        rhs = m.group(0).split("=", 1)[1]
        depth = rhs.split("#")[0].count("[") - rhs.split("#")[0].count("]")
        while depth > 0:
            nl = src.find("\n", end + 1)
            nl = len(src) if nl == -1 else nl
            line = src[end + 1:nl]
            depth += line.split("#")[0].count("[") - line.split("#")[0].count("]")
            end = nl
        original = " ".join(x.split("#")[0].strip() for x in src[start:end].split("=", 1)[1].splitlines())
        applied[name] = {"asli": original.strip(), "cepat": value}
        src = src[:start] + f"{name} = {value}  # [MODE CEPAT] diubah oleh GUI" + src[end:]
    return src, applied


UCI_SHIM = '''"""Shim lokal ucimlrepo (dibuat oleh GUI): membaca cache UCI Mushroom (id=73) agar
eksperimen bisa dijalankan tanpa internet. Isi cache identik dengan unduhan ucimlrepo."""
import types
import pandas as pd
_CACHE = {cache!r}


def fetch_ucirepo(id=None, name=None):
    if id not in (73, None):
        raise ValueError("Cache lokal hanya berisi dataset UCI id=73 (Mushroom)")
    df = pd.read_csv(_CACHE, dtype=str, keep_default_na=False, na_values=[""])
    o = types.SimpleNamespace()
    o.data = types.SimpleNamespace(features=df.drop(columns=["poisonous"]), targets=df[["poisonous"]])
    o.metadata = {{"name": "Mushroom", "uci_id": 73, "source": "cache lokal GUI"}}
    return o
'''


# ---------------------------------------------------------------------------
# Pengelola run
# ---------------------------------------------------------------------------
class Runner:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.current = None      # dict info run aktif
        self.log = []            # baris log run aktif / terakhir
        self.queue = []
        self.stop_requested = False

    def busy(self):
        return self.current is not None

    def enqueue(self, keys, mode):
        with self.lock:
            if self.busy():
                return False, "Masih ada eksperimen yang berjalan."
            for k in keys:
                if k not in EXP:
                    return False, f"Eksperimen tidak dikenal: {k}"
            self.queue = [(k, mode) for k in keys]
            self.stop_requested = False
        threading.Thread(target=self._worker, daemon=True).start()
        return True, "ok"

    def stop(self):
        with self.lock:
            self.stop_requested = True
            self.queue = []
            p = self.proc
        if p is None:
            with self.lock:
                self.current = None
        if p and p.poll() is None:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
            else:
                p.terminate()
        return True

    def _emit(self, line):
        self.log.append(line.rstrip("\n"))

    def _prepare(self, key, mode):
        e = EXP[key]
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(RUNS, f"{stamp}_{key}_{mode}")
        os.makedirs(run_dir, exist_ok=True)
        src_path = os.path.join(SCRIPTS, e["script"])
        with open(src_path, encoding="utf8") as f:
            src = f.read()
        overrides = e["cepat"] if mode == "cepat" else e["full"]
        src2, applied = patch_constants(src, overrides)
        with open(os.path.join(run_dir, e["script"]), "w", encoding="utf8") as f:
            f.write(src2)
        info = {"experiment": key, "title": e["title"], "mode": mode, "script": e["script"],
                "script_sha256_original": sha256(src_path), "overrides": applied,
                "run_dir": run_dir, "started": datetime.now().isoformat(timespec="seconds"),
                "python": platform.python_version(), "cpu": cpu_name(), "ram_gb": ram_gb()}
        if e["needs"] == "gene":
            g = gene_dataset()
            if not g["found"]:
                raise RuntimeError("File dataset gen tidak ditemukan. Atur lokasinya di tab 'Lingkungan & Data'.")
            shutil.copyfile(g["path"], os.path.join(run_dir, GENE_FILE))
            info["dataset"] = {"file": g["path"], "sha256": g["sha256"], "cocok_dengan_paper": g["match"]}
        if e["needs"] == "benchmark":
            bench = benchmark_status()
            if bench["found"] < len(BENCH_SHA256):
                raise RuntimeError(f"Dataset benchmark PyAerial tidak lengkap di {BENCH_DIR} "
                                   f"({bench['found']}/{len(BENCH_SHA256)} file).")
            for name in BENCH_SHA256:
                shutil.copyfile(os.path.join(BENCH_DIR, name + ".csv"), os.path.join(run_dir, name + ".csv"))
            info["dataset"] = {"benchmark_dir": BENCH_DIR, "sha256_semua_cocok": bench["all_match"]}
            g = gene_dataset()
            if g["found"]:
                shutil.copyfile(g["path"], os.path.join(run_dir, GENE_FILE))
                info["dataset"]["gene"] = {"file": g["path"], "sha256": g["sha256"], "cocok_dengan_paper": g["match"]}
        if e["needs"] == "stats":
            info["inputs"] = {}
            for fname, key2 in [("md_unsup_per_run.csv", "multi"), ("md_sup_per_run.csv", "multi"),
                                ("rob_per_run.csv", "robust")]:
                src = latest_output(key2, fname)
                if src:
                    shutil.copyfile(src, os.path.join(run_dir, fname))
                    info["inputs"][fname] = src
        if e["needs"] == "mushroom":
            ms = mushroom_status()
            if ms["cached"]:
                with open(os.path.join(run_dir, "ucimlrepo.py"), "w", encoding="utf8") as f:
                    f.write(UCI_SHIM.format(cache=MUSHROOM_CACHE))
                info["dataset"] = {"file": MUSHROOM_CACHE, "sha256": ms["sha256"], "sumber": "cache lokal UCI id=73"}
            else:
                info["dataset"] = {"sumber": "diunduh langsung dari UCI (perlu internet)"}
        return run_dir, info

    def _worker(self):
        """Menjalankan antrean. Status 'current' SELALU dibersihkan, apa pun yang terjadi."""
        try:
            while True:
                with self.lock:
                    if not self.queue or self.stop_requested:
                        return
                    key, mode = self.queue.pop(0)
                self._run_one(key, mode)
        except Exception as ex:
            self._emit(f"[GUI] Kesalahan tak terduga: {ex}")
        finally:
            with self.lock:
                self.current = None
                self.proc = None

    def _run_one(self, key, mode):
        self.log = []
        try:
            run_dir, info = self._prepare(key, mode)
        except Exception as ex:
            self._emit(f"[GUI] GAGAL menyiapkan run {EXP[key]['title']}: {ex}")
            return
        t0 = time.time()
        self.current = {"key": key, "mode": mode, "run_id": os.path.basename(run_dir), "t0": t0,
                        "title": EXP[key]["title"]}
        self._emit(f"[GUI] {EXP[key]['title']} | mode {mode.upper()} | folder: {run_dir}")
        for n, v in info["overrides"].items():
            self._emit(f"[GUI] mode cepat: {n} {v['asli']} -> {v['cepat']}")
        self._emit(f"[GUI] Python: {sys.executable}")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        kw = {"creationflags": 0x08000000} if sys.platform == "win32" else {}  # tanpa jendela konsol
        code = -1
        try:
            self.proc = subprocess.Popen([sys.executable, "-u", EXP[key]["script"]], cwd=run_dir,
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, env=env, **kw)
            # skrip menunggu "Tekan Enter" di akhir; kirim Enter agar selesai sendiri
            try:
                self.proc.stdin.write(b"\n" * 20)
                self.proc.stdin.close()
            except Exception:
                pass
            for raw in iter(self.proc.stdout.readline, b""):
                self._emit(raw.decode("utf-8", errors="replace"))
            code = self.proc.wait()
        except Exception as ex:
            self._emit(f"[GUI] Gagal menjalankan: {ex}")
        dur = time.time() - t0
        stopped = self.stop_requested
        ok = code == 0 and os.path.isfile(os.path.join(run_dir, EXP[key]["summary"])) and not stopped
        info.update(status="selesai" if ok else ("dihentikan" if stopped else "gagal"),
                    exit_code=code, duration_s=round(dur, 1),
                    finished=datetime.now().isoformat(timespec="seconds"))
        try:
            os.makedirs(run_dir, exist_ok=True)
            info["output_files"] = sorted(os.listdir(run_dir))
            with open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
            with open(os.path.join(run_dir, "gui_console_log.txt"), "w", encoding="utf8") as f:
                f.write("\n".join(self.log))
        except Exception as ex:
            self._emit(f"[GUI] Gagal menyimpan catatan run: {ex}")
        self._emit(f"[GUI] {info['status'].upper()} dalam {dur:.1f} detik (exit code {code}).")
        self.proc = None


RUNNER = Runner()
FIG_LOCK = threading.Lock()


def list_runs():
    out = []
    if not os.path.isdir(RUNS):
        return out
    for d in sorted(os.listdir(RUNS), reverse=True):
        p = os.path.join(RUNS, d, "run_info.json")
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf8") as f:
                    i = json.load(f)
                out.append({"run_id": d, "title": i.get("title"), "experiment": i.get("experiment"),
                            "mode": i.get("mode"), "status": i.get("status"), "started": i.get("started"),
                            "duration_s": i.get("duration_s")})
            except Exception:
                pass
    return out


def read_text(path):
    try:
        with open(path, encoding="utf8", errors="replace") as f:
            return f.read()
    except Exception:
        return None


def run_detail(run_id):
    d = os.path.join(RUNS, os.path.basename(run_id))
    info = json.loads(read_text(os.path.join(d, "run_info.json")) or "{}")
    e = EXP.get(info.get("experiment"), {})
    return {"info": info,
            "summary": read_text(os.path.join(d, e.get("summary", ""))) if e else None,
            "archive": read_text(os.path.join(RESULTS, e.get("archive", ""), e.get("summary", ""))) if e else None,
            "archive_path": f"results/{e.get('archive')}/{e.get('summary')}" if e else None}


COMPARE = {
    "pyaerial": {"file": "pyaerial_mushroom_results.csv", "seed": "seed", "group": [],
                 "metrics": ["n_rules", "data_coverage", "test_accuracy"]},
    "mushroom": {"file": "mi_results_per_seed.csv", "seed": "seed", "group": ["method"],
                 "metrics": ["n_rules", "test_accuracy", "test_coverage", "runtime_s"]},
    "scal": {"file": "scal_results_per_run.csv", "seed": "run", "group": ["n_columns", "method"],
             "metrics": ["rule_count", "data_coverage", "exec_time"], "ok_only": True},
    "abl": {"file": "abl_results_per_run.csv", "seed": "seed", "group": ["n_columns", "strategy"],
            "metrics": ["rule_count", "average_lift", "average_zhang"]},
    "migene": {"file": "migene_results.csv", "seed": "seed", "group": ["n_columns", "method"],
               "metrics": ["rule_count", "average_lift", "average_zhang", "exec_time"]},
    "multi": {"file": "md_unsup_per_run.csv", "seed": "seed", "group": ["dataset", "method"],
              "metrics": ["rule_count", "average_lift", "average_zhang"]},
    "robust": {"file": "rob_per_run.csv", "seed": "seed", "group": ["dataset", "guide", "k"],
               "metrics": ["rule_count", "average_zhang"]},
}
TIME_METRICS = {"exec_time", "runtime_s"}


def compare_run(run_id):
    """Bandingkan angka kunci run dengan arsip pada seed/ukuran yang sama."""
    d = os.path.join(RUNS, os.path.basename(run_id))
    info = json.loads(read_text(os.path.join(d, "run_info.json")) or "{}")
    key = info.get("experiment")
    spec = COMPARE.get(key)
    if not spec:
        return {"rows": [], "note": "Tidak ada spesifikasi perbandingan."}
    run_csv = os.path.join(d, spec["file"])
    arc_csv = os.path.join(RESULTS, EXP[key]["archive"], spec["file"])
    if not os.path.isfile(run_csv):
        return {"rows": [], "note": f"File {spec['file']} tidak dihasilkan oleh run ini."}
    if not os.path.isfile(arc_csv):
        return {"rows": [], "note": f"Arsip {spec['file']} tidak ditemukan."}
    try:
        import pandas as pd
    except Exception:
        return {"rows": [], "note": "pandas belum terpasang."}
    r = pd.read_csv(run_csv)
    a = pd.read_csv(arc_csv)
    if spec.get("ok_only"):
        r = r[r["status"] == "ok"]
        a = a[a["status"] == "ok"]
    seeds = set(r[spec["seed"]].dropna().tolist()) if spec["seed"] in r else set()
    if spec["seed"] in a:
        a = a[a[spec["seed"]].isna() | a[spec["seed"]].isin(seeds)]
    g = spec["group"]
    mets = [m for m in spec["metrics"] if m in r.columns and m in a.columns]
    for df in (r, a):
        for m in mets:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    if g:
        rm = r.groupby(g)[mets].mean()
        am = a.groupby(g)[mets].mean()
    else:
        rm = r[mets].mean().to_frame().T
        am = a[mets].mean().to_frame().T
    rows = []
    for idx in rm.index:
        if idx not in am.index:
            continue
        label = " | ".join(str(x) for x in (idx if isinstance(idx, tuple) else (idx,))) if g else "semua"
        for m in mets:
            rv, av = rm.loc[idx, m], am.loc[idx, m]
            if pd.isna(rv) or pd.isna(av):
                continue
            diff = rv - av
            rel = abs(diff) / abs(av) if av else (0.0 if diff == 0 else 1.0)
            if m in TIME_METRICS:
                verdict = "waktu"
            elif abs(diff) < 1e-9:
                verdict = "identik"
            elif rel <= 0.02:
                verdict = "selisih kecil"
            else:
                verdict = "berbeda"
            rows.append({"group": label, "metric": m, "run": float(rv), "archive": float(av),
                         "diff": float(diff), "verdict": verdict})
    return {"rows": rows, "seeds": sorted(seeds),
            "note": f"Arsip difilter ke seed/run yang sama dengan run ini: {sorted(seeds) or 'deterministik'}."}


def regenerate_figures():
    with FIG_LOCK:
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "make_figures.py")], capture_output=True,
                           text=True, cwd=SCRIPTS, **({"creationflags": 0x08000000} if sys.platform == "win32" else {}))
        return r.returncode == 0, (r.stdout + r.stderr)[-2000:]


def cache_mushroom():
    try:
        from ucimlrepo import fetch_ucirepo
        import pandas as pd
        uci = fetch_ucirepo(id=73)
        df = pd.concat([uci.data.features, uci.data.targets], axis=1)
        os.makedirs(DATA, exist_ok=True)
        df.to_csv(MUSHROOM_CACHE, index=False)
        return True, f"Tersimpan: {MUSHROOM_CACHE} ({df.shape[0]} baris, {df.shape[1]} kolom)"
    except Exception as ex:
        return False, f"Gagal mengunduh (perlu internet dan pustaka ucimlrepo): {ex}"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _file(self, path, ctype):
        with open(path, "rb") as f:
            b = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._file(os.path.join(GUI_DIR, "index.html"), "text/html; charset=utf-8")
        if u.path == "/api/experiments":
            out = []
            for e in EXPERIMENTS:
                out.append({**{k: e[k] for k in ("key", "no", "title", "purpose", "manuscript", "needs",
                                                  "est_full", "est_cepat", "script", "summary")},
                            "cepat": e["cepat"],
                            "archive_summary": read_text(os.path.join(RESULTS, e["archive"], e["summary"]))})
            return self._json(out)
        if u.path == "/api/env":
            return self._json(env_info())
        if u.path == "/api/status":
            off = int(q.get("offset", ["0"])[0])
            cur = RUNNER.current
            return self._json({"running": cur is not None,
                               "current": None if not cur else {**cur, "elapsed": round(time.time() - cur["t0"], 1)},
                               "queue": [k for k, _ in RUNNER.queue], "log": RUNNER.log[off:],
                               "offset": len(RUNNER.log)})
        if u.path == "/api/runs":
            return self._json(list_runs())
        if u.path == "/api/run":
            return self._json(run_detail(q.get("id", [""])[0]))
        if u.path == "/api/compare":
            try:
                return self._json(compare_run(q.get("id", [""])[0]))
            except Exception as ex:
                return self._json({"rows": [], "note": f"Perbandingan gagal: {ex}"})
        if u.path == "/api/tables":
            try:
                import report_tables
                pref = q.get("source", ["auto"])[0]
                return self._json(report_tables.build_tables(lambda k, f: find_source(k, f, pref)))
            except Exception as ex:
                return self._json([{"no": "-", "title": "Tabel gagal dibuat", "missing": str(ex), "header": [],
                                    "rows": [], "source": "-", "note": ""}])
        if u.path == "/api/figlist":
            return self._json(figure_list(q.get("source", ["auto"])[0]))
        if u.path == "/runfig":
            key, name = q.get("key", [""])[0], os.path.basename(q.get("name", [""])[0])
            allowed = {(k, n) for _, _, k, n in FIG_LIST if k}
            if (key, name) in allowed:
                p, _ = find_source(key, name, q.get("source", ["auto"])[0])
                if p:
                    return self._file(p, "image/png")
        if u.path.startswith("/figures/"):
            name = os.path.basename(unquote(u.path))
            p = os.path.join(FIGURES, name)
            if name.endswith(".png") and os.path.isfile(p):
                return self._file(p, "image/png")
        self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path)
        b = self._body()
        if u.path == "/api/start":
            keys = b.get("keys") or []
            mode = {"demo": "cepat"}.get(b.get("mode"), b.get("mode"))
            mode = mode if mode in ("cepat", "full") else "cepat"
            ok, msg = RUNNER.enqueue(keys, mode)
            return self._json({"ok": ok, "message": msg}, 200 if ok else 409)
        if u.path == "/api/stop":
            RUNNER.stop()
            return self._json({"ok": True})
        if u.path == "/api/figures":
            ok, out = regenerate_figures()
            return self._json({"ok": ok, "output": out})
        if u.path == "/api/cache_mushroom":
            ok, msg = cache_mushroom()
            return self._json({"ok": ok, "message": msg})
        if u.path == "/api/gene_path":
            p = (b.get("path") or "").strip().strip('"')
            if p and os.path.isdir(p):
                p = os.path.join(p, GENE_FILE)
            if not os.path.isfile(p):
                return self._json({"ok": False, "message": f"File tidak ditemukan: {p}"}, 400)
            s = load_settings(); s["gene_path"] = p; save_settings(s)
            return self._json({"ok": True, "gene": gene_dataset()})
        if u.path == "/api/open_runs":
            os.makedirs(RUNS, exist_ok=True)
            target = os.path.join(RUNS, os.path.basename(b.get("run_id", ""))) if b.get("run_id") else RUNS
            try:
                if sys.platform == "win32":
                    os.startfile(target)
                return self._json({"ok": True, "path": target})
            except Exception as ex:
                return self._json({"ok": False, "message": str(ex)})
        self.send_error(404)


def main():
    os.makedirs(RUNS, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print("=" * 64)
    print(" ARM Experiment Runner berjalan di", url)
    print(" Jangan tutup jendela ini selama aplikasi dipakai.")
    print(" Tekan Ctrl+C untuk berhenti.")
    print("=" * 64)
    if os.environ.get("ARM_GUI_NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        RUNNER.stop()


if __name__ == "__main__":
    main()
