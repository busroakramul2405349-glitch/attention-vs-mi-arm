# Panduan ARM Experiment Runner (GUI)

GUI ini menjalankan seluruh eksperimen naskah secara nyata dari satu tempat dan menyimpan bukti
setiap run. GUI hanya memakai pustaka standar Python dan berjalan di browser pada alamat
`http://127.0.0.1:8765` (hanya bisa diakses dari laptop ini).

## Persiapan (sekali, sebelum hari ujian, perlu internet)

1. Sembilan dataset benchmark PyAerial untuk eksperimen 6–7 sudah ada di `data\pyaerial_benchmark\`
   (tab 2 menampilkan status SHA-256-nya). Pastikan juga dataset gen ada. GUI mencari otomatis di `data\`, `scripts\`, folder repositori,
   `D:\NS ARM BARU VS\data\` dan `Downloads`. Paling aman: salin
   `cell_cancertype_NonSmallCellLungCarcinoma_lnic50.csv` ke folder `data\`.
2. Klik dua kali `JALANKAN_GUI.bat`. Browser terbuka otomatis.
3. Tab **2. Lingkungan & Data**:
   - pastikan label hijau **IDENTIK dengan dataset paper PyAerial** muncul;
   - klik **Unduh & simpan untuk mode offline** agar Mushroom bisa dipakai tanpa internet;
   - pastikan semua pustaka (numpy, pandas, torch, mlxtend, pyaerial, ucimlrepo, matplotlib) sudah terpasang.
4. Tab **3**: klik **Jalankan semua (cepat)** sekali sebagai gladi bersih (± 10 menit). Semua run harus berstatus *selesai*.
5. Matikan Wi-Fi, lalu ulangi langkah 4 untuk memastikan semuanya berjalan offline.

## Saat ujian (± 10 menit)

| Langkah | Tab | Yang ditunjukkan ke penguji |
| --- | --- | --- |
| 1 | 1. Alur Penelitian | Alur eksperimen dan pemetaan ke tabel/gambar naskah |
| 2 | 2. Lingkungan & Data | Spesifikasi laptop; dataset identik (SHA-256) dengan dataset paper PyAerial |
| 3 | 3. Jalankan Eksperimen | Klik **Jalankan semua (cepat)**, atau satu eksperimen pilihan penguji |
| 4 | 4. Log Langsung | Keluaran program muncul langsung saat berjalan |
| 5 | 5. Hasil vs Arsip | Tabel **Angka kunci**: angka run saat itu dibandingkan arsip Zenodo pada seed yang sama |
| 6 | 6. Gambar | **Buat ulang gambar sekarang**: Gambar 2–4 dibuat ulang dari file hasil (Gambar 1 adalah diagram alur, `figures/fig0_framework.png`) |

## Mode cepat vs mode penuh

Kedua mode menjalankan **kode yang sama** dengan skrip di `scripts\`. Mode cepat hanya mengurangi jumlah
seed atau ukuran data agar selesai dalam hitungan menit; perubahan itu tercatat di `run_info.json` dan
ditampilkan di tab 5.

| Eksperimen | Mode cepat | Mode penuh (angka naskah) |
| --- | --- | --- |
| 1. Replikasi PyAerial | seed 0 | seed 0–4 |
| 2. Mushroom (Tabel 1–2) | seed 0 | seed 0–4 |
| 3. Skalabilitas (Tabel 3) | 50, 100, 200 kolom; 1 run | 50–1.000 kolom; 3 run |
| 4. Ablasi (Tabel 4) | 100 gen; seed 0 | 100, 300, 1.000 gen; seed 0–2 |
| 5. MI-guided (Tabel 3) | 100, 1.000, 18.108 gen; seed acak 0 | 50–18.108 gen; seed acak 0–2 |
| 6. Multi-dataset (9 dataset benchmark PyAerial + gen) | 4 dataset kecil; seed 0; tanpa data gen | 9 dataset + gen 1.000 kolom; seed 0–4 (± 45 menit) |
| 7. Ketahanan attention | Congressional Voting; seed 0; 3 varian | 4 dataset; seed 0–2; 9 varian (± 45 menit) |
| 8. Uji statistik | sama dengan mode penuh | memakai hasil run 6 dan 7 terbaru (mode penuh didahulukan) |

## Membaca tabel "Angka kunci"

- **identik**: angka run sama persis dengan arsip.
- **selisih kecil**: berbeda kurang dari 2%; biasa terjadi pada metode berbasis neural network
  (VATTEN, PyAerial) atau versi NumPy yang berbeda, karena urutan operasi floating-point.
- **waktu**: waktu eksekusi, selalu bergantung pada perangkat keras dan beban laptop saat itu.
- **berbeda**: perlu dijelaskan; periksa versi pustaka di tab 2.

## Bukti yang tersimpan

Setiap run membuat folder `runs\<tanggal-jam>_<eksperimen>_<mode>\` berisi salinan skrip yang dijalankan,
semua file hasil, `gui_console_log.txt` (log lengkap) dan `run_info.json` (mode, perubahan konstanta,
SHA-256 skrip asli dan dataset, CPU, RAM, waktu mulai/selesai, status).

## Jika ada masalah

- **Browser tidak terbuka**: buka manual `http://127.0.0.1:8765`.
- **Port sudah dipakai**: tutup jendela GUI sebelumnya, lalu jalankan lagi.
- **Eksperimen 1–2 gagal tanpa internet**: lakukan langkah persiapan 3 (cache offline).
- **Run terlalu lama**: klik **Hentikan** di kanan atas; statusnya tercatat *dihentikan*.
