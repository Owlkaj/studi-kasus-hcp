#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulasi Komputasi Paralel untuk Prediksi Indeks Bahaya Banjir dan Polusi
Udara (PM2.5) Lokal Berbasis Multi-Sensor
==========================================================================

Studi kasus tugas kuliah High Performance Computing (HPC):
"Bridging Global Infrastructure to Address Local Challenge with HPC".

Program ini membandingkan dua metode eksekusi untuk pekerjaan yang sama:
    a) Serial   -> 1 core CPU
    b) Paralel  -> semua core CPU dengan multiprocessing.Pool
                   (pola data parallelism: domain data dipecah menjadi
                   chunk, tiap chunk dihitung oleh satu worker)

Seluruh data sensor bersifat SINTETIS (dummy) dan hanya untuk simulasi.

Contoh penggunaan:
    python3 simulasi_hpc_banjir.py
    python3 simulasi_hpc_banjir.py --rows 200000 --iterations 20
    python3 simulasi_hpc_banjir.py --workers 8 --scaling
"""

import argparse
import math
import multiprocessing as mp
import os
import time

import matplotlib

matplotlib.use("Agg")  # backend tanpa GUI: aman dijalankan di server/cluster
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Konfigurasi default
# ---------------------------------------------------------------------------
JUMLAH_BARIS_DEFAULT = 1_000_000
ITERASI_DEFAULT = 25          # kedalaman simulasi per titik (beban komputasi)
SEED = 42
OUTPUT_GAMBAR = "benchmark_result.png"

# Profil wilayah (fiktif/sintetis): faktor hujan, level dasar sungai (m),
# dan level dasar PM2.5 (ug/m3). Nama wilayah mengacu pada kecamatan lokal
# hanya sebagai ilustrasi; angkanya BUKAN data pengukuran asli.
WILAYAH = {
    "Selebar":          {"hujan": 1.00, "sungai": 2.2, "pm25": 28.0},
    "Gading Cempaka":   {"hujan": 1.10, "sungai": 2.6, "pm25": 32.0},
    "Teluk Segara":     {"hujan": 1.25, "sungai": 3.1, "pm25": 38.0},
    "Muara Bangkahulu": {"hujan": 1.35, "sungai": 3.6, "pm25": 30.0},
    "Ratu Agung":       {"hujan": 0.95, "sungai": 2.0, "pm25": 41.0},
    "Ratu Samban":      {"hujan": 0.90, "sungai": 1.8, "pm25": 44.0},
    "Singaran Pati":    {"hujan": 1.05, "sungai": 2.3, "pm25": 35.0},
    "Kampung Melayu":   {"hujan": 1.40, "sungai": 3.8, "pm25": 33.0},
    "Sungai Serut":     {"hujan": 1.30, "sungai": 3.4, "pm25": 29.0},
}
NAMA_WILAYAH = list(WILAYAH.keys())

# Ambang status berdasarkan Skor Risiko (0-100)
AMBANG = [(75.0, "Bahaya"), (60.0, "Siaga"), (45.0, "Waspada"), (0.0, "Aman")]
STATUS_LABEL = ["Aman", "Waspada", "Siaga", "Bahaya"]


# ---------------------------------------------------------------------------
# 1. Generator data sensor dummy
# ---------------------------------------------------------------------------
def buat_data_sensor(n_baris, seed=SEED):
    """
    Membangkitkan data multi-sensor sintetis.

    Kolom (semua array float64 kecuali id_wilayah):
        id_wilayah : indeks wilayah (0..8)
        hujan      : curah hujan (mm/jam)
        sungai     : ketinggian air sungai (meter)
        tanah      : kelembaban tanah (0.0 - 1.0)
        pm25       : kadar PM2.5 (ug/m3)
    """
    rng = np.random.default_rng(seed)

    id_wilayah = rng.integers(0, len(NAMA_WILAYAH), size=n_baris)
    f_hujan = np.array([WILAYAH[w]["hujan"] for w in NAMA_WILAYAH])[id_wilayah]
    b_sungai = np.array([WILAYAH[w]["sungai"] for w in NAMA_WILAYAH])[id_wilayah]
    b_pm25 = np.array([WILAYAH[w]["pm25"] for w in NAMA_WILAYAH])[id_wilayah]

    hujan = rng.gamma(shape=1.6, scale=11.0, size=n_baris) * f_hujan
    sungai = np.clip(rng.normal(b_sungai, 0.9, size=n_baris), 0.2, None)
    sungai = sungai + hujan * 0.012            # hujan deras menaikkan sungai
    tanah = rng.beta(4.0, 2.5, size=n_baris)
    pm25 = rng.lognormal(mean=np.log(b_pm25), sigma=0.45, size=n_baris)

    return {
        "id_wilayah": id_wilayah.astype(np.int16),
        "hujan": hujan,
        "sungai": sungai,
        "tanah": tanah,
        "pm25": pm25,
    }


# ---------------------------------------------------------------------------
# 2. Fungsi komputasi berat (floating-point intensive)
# ---------------------------------------------------------------------------
def skor_satu_titik(hujan, sungai, tanah, pm25, iterasi):
    """
    Menghitung Skor Risiko Bencana & Polusi (0-100) untuk satu titik sensor.

    Model sederhana (bukan model hidrologi/atmosfer sungguhan) yang
    sengaja dibuat padat operasi floating-point (exp, tanh, sqrt, log, sin,
    cos) agar beban CPU terasa saat dijalankan serial vs paralel.

    - Bagian banjir : simulasi akumulasi limpasan air tiap langkah waktu.
    - Bagian PM2.5  : simulasi dispersi/fluktuasi partikel tiap langkah.
    """
    air = sungai
    pm = pm25
    infiltrasi = 1.0 - tanah
    puncak_air = air

    for k in range(1, iterasi + 1):
        t = k / iterasi
        # limpasan: makin deras hujan & makin jenuh tanah -> makin besar
        aliran = hujan * (1.0 - math.exp(-3.0 * t)) * (
            0.5 + 0.5 * math.tanh(2.0 * tanah - 1.0 + t)
        )
        air = air + 0.01 * aliran * math.sqrt(1.0 + air) \
            - 0.02 * infiltrasi * math.log1p(air)
        if air < 0.0:
            air = 0.0
        if air > puncak_air:
            puncak_air = air

        # dispersi PM2.5: osilasi harian + penyebaran angin
        pm = pm * (1.0 + 0.002 * math.sin(6.2831853 * t + pm / 50.0)) \
            + 0.05 * math.cos(t * pm / 30.0)
        if pm < 0.0:
            pm = 0.0

    # Normalisasi ke skala 0-100
    indeks_banjir = 100.0 / (1.0 + math.exp(-(puncak_air - 5.0) * 0.9))
    indeks_banjir *= 0.6 + 0.4 * tanah          # tanah jenuh memperburuk
    indeks_pm = min(100.0, (pm / 150.0) * 100.0)  # 150 ug/m3 ~ batas tinggi

    return 0.6 * indeks_banjir + 0.4 * indeks_pm


def proses_chunk(args):
    """
    Worker: menghitung skor untuk satu potongan (chunk) data.

    Fungsi ini dipakai OLEH KEDUA metode (serial & paralel) sehingga
    perbandingan waktunya adil: kode hitung identik, hanya cara
    eksekusinya yang berbeda.
    """
    hujan, sungai, tanah, pm25, iterasi = args
    hasil = np.empty(len(hujan), dtype=np.float64)
    # .tolist() -> float Python murni, lebih cepat diiterasi daripada
    # indexing elemen numpy satu per satu.
    h, s, t, p = hujan.tolist(), sungai.tolist(), tanah.tolist(), pm25.tolist()
    for i in range(len(h)):
        hasil[i] = skor_satu_titik(h[i], s[i], t[i], p[i], iterasi)
    return hasil


# ---------------------------------------------------------------------------
# 3. Metode eksekusi
# ---------------------------------------------------------------------------
def jalankan_serial(data, iterasi):
    """Metode A: serial, 1 core CPU, satu chunk berisi seluruh data."""
    mulai = time.perf_counter()
    skor = proses_chunk(
        (data["hujan"], data["sungai"], data["tanah"], data["pm25"], iterasi)
    )
    return skor, time.perf_counter() - mulai


def jalankan_paralel(data, iterasi, n_workers, chunks_per_worker=4):
    """
    Metode B: paralel dengan multiprocessing.Pool.

    Data dipecah menjadi (n_workers x chunks_per_worker) chunk agar beban
    terdistribusi merata (load balancing) lalu dikumpulkan kembali sesuai
    urutan asli (pool.map menjaga urutan).
    """
    n = len(data["hujan"])
    n_chunk = max(1, n_workers * chunks_per_worker)
    batas = np.linspace(0, n, n_chunk + 1, dtype=np.int64)

    tugas = [
        (
            data["hujan"][batas[i]:batas[i + 1]],
            data["sungai"][batas[i]:batas[i + 1]],
            data["tanah"][batas[i]:batas[i + 1]],
            data["pm25"][batas[i]:batas[i + 1]],
            iterasi,
        )
        for i in range(n_chunk)
        if batas[i + 1] > batas[i]
    ]

    mulai = time.perf_counter()
    with mp.Pool(processes=n_workers) as pool:
        potongan = pool.map(proses_chunk, tugas)
    skor = np.concatenate(potongan)
    return skor, time.perf_counter() - mulai


# ---------------------------------------------------------------------------
# 4. Analisis hasil & pelaporan
# ---------------------------------------------------------------------------
def klasifikasi_status(skor):
    """Mengubah skor (0-100) menjadi kode status 0..3 (Aman..Bahaya)."""
    kode = np.zeros(len(skor), dtype=np.int8)
    kode[skor >= 45.0] = 1
    kode[skor >= 60.0] = 2
    kode[skor >= 75.0] = 3
    return kode


def ringkas_per_wilayah(data, skor):
    """Ringkasan per wilayah: rata-rata skor, jumlah titik Siaga & Bahaya."""
    kode = klasifikasi_status(skor)
    ringkasan = []
    for idx, nama in enumerate(NAMA_WILAYAH):
        m = data["id_wilayah"] == idx
        total = int(m.sum())
        siaga = int(((kode == 2) & m).sum())
        bahaya = int(((kode == 3) & m).sum())
        ringkasan.append({
            "wilayah": nama,
            "total": total,
            "rata2": float(skor[m].mean()) if total else 0.0,
            "siaga": siaga,
            "bahaya": bahaya,
            "persen": 100.0 * (siaga + bahaya) / total if total else 0.0,
        })
    ringkasan.sort(key=lambda r: r["persen"], reverse=True)
    return ringkasan


def cetak_ringkasan(ringkasan, skor):
    kode = klasifikasi_status(skor)
    print("\n" + "=" * 78)
    print(" RINGKASAN WILAYAH BERSTATUS SIAGA / BAHAYA")
    print("=" * 78)
    print(f"{'Wilayah':<18}{'Titik':>9}{'Rata2 Skor':>12}{'Siaga':>10}"
          f"{'Bahaya':>10}{'% Siaga+Bhy':>13}")
    print("-" * 78)
    for r in ringkasan:
        print(f"{r['wilayah']:<18}{r['total']:>9,}{r['rata2']:>12.2f}"
              f"{r['siaga']:>10,}{r['bahaya']:>10,}{r['persen']:>12.2f}%")
    print("-" * 78)
    print("Distribusi status seluruh titik:")
    for i, label in enumerate(STATUS_LABEL):
        jumlah = int((kode == i).sum())
        print(f"  {label:<8}: {jumlah:>10,} titik ({100.0 * jumlah / len(skor):5.2f}%)")
    if ringkasan:
        print(f"\nWilayah paling rawan: {ringkasan[0]['wilayah']} "
              f"({ringkasan[0]['persen']:.2f}% titik berstatus Siaga/Bahaya)")


def simpan_grafik(t_serial, t_paralel, speedup, n_workers, ringkasan,
                  n_baris, nama_file=OUTPUT_GAMBAR, hasil_scaling=None):
    """Menyimpan grafik perbandingan waktu eksekusi + ringkasan wilayah."""
    ada_scaling = bool(hasil_scaling)
    kolom = 3 if ada_scaling else 2
    fig, axes = plt.subplots(1, kolom, figsize=(6 * kolom, 5.2))

    # --- Panel 1: perbandingan waktu eksekusi
    ax = axes[0]
    bars = ax.bar(
        ["Serial\n(1 core)", f"Paralel\n({n_workers} core)"],
        [t_serial, t_paralel],
        color=["#d9534f", "#2e86c1"], width=0.55,
    )
    for b, v in zip(bars, [t_serial, t_paralel]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f} s",
                ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Waktu eksekusi (detik)")
    ax.set_title(f"Waktu Eksekusi ({n_baris:,} titik sensor)\n"
                 f"Speedup = {speedup:.2f}x")
    ax.set_ylim(0, max(t_serial, t_paralel) * 1.2)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 2: titik Siaga/Bahaya per wilayah
    ax = axes[1]
    nama = [r["wilayah"] for r in ringkasan][::-1]
    siaga = [r["siaga"] for r in ringkasan][::-1]
    bahaya = [r["bahaya"] for r in ringkasan][::-1]
    ax.barh(nama, siaga, color="#f0ad4e", label="Siaga")
    ax.barh(nama, bahaya, left=siaga, color="#c0392b", label="Bahaya")
    ax.set_xlabel("Jumlah titik sensor")
    ax.set_title("Titik Berstatus Siaga / Bahaya per Wilayah")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    # --- Panel 3 (opsional): scaling terhadap jumlah worker
    if ada_scaling:
        ax = axes[2]
        w = [x[0] for x in hasil_scaling]
        sp = [x[2] for x in hasil_scaling]
        ax.plot(w, sp, "o-", color="#2e86c1", label="Speedup terukur")
        ax.plot(w, w, "--", color="gray", label="Speedup ideal")
        ax.set_xlabel("Jumlah worker (core)")
        ax.set_ylabel("Speedup (x)")
        ax.set_title("Scaling terhadap Jumlah Core")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Benchmark HPC: Prediksi Indeks Bahaya Banjir & PM2.5 Lokal",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(nama_file, dpi=150)
    plt.close(fig)
    print(f"\nGrafik benchmark disimpan: {nama_file}")


# ---------------------------------------------------------------------------
# 5. Program utama
# ---------------------------------------------------------------------------
def deteksi_core():
    """Gunakan alokasi SLURM jika ada, jika tidak pakai semua core lokal."""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        return int(slurm)
    return os.cpu_count() or 1


def main():
    parser = argparse.ArgumentParser(
        description="Simulasi HPC: serial vs paralel (multiprocessing)")
    parser.add_argument("--rows", type=int, default=JUMLAH_BARIS_DEFAULT,
                        help="jumlah baris data sensor (default: 1.000.000)")
    parser.add_argument("--iterations", type=int, default=ITERASI_DEFAULT,
                        help="kedalaman simulasi per titik (default: 25)")
    parser.add_argument("--workers", type=int, default=None,
                        help="jumlah proses paralel (default: semua core)")
    parser.add_argument("--scaling", action="store_true",
                        help="uji tambahan: speedup untuk 1,2,4,... worker")
    parser.add_argument("--output", default=OUTPUT_GAMBAR,
                        help="nama file grafik hasil benchmark")
    args = parser.parse_args()

    n_workers = args.workers or deteksi_core()

    print("=" * 78)
    print(" SIMULASI HPC: PREDIKSI INDEKS BAHAYA BANJIR & POLUSI UDARA (PM2.5)")
    print("=" * 78)
    print(f" Jumlah titik data : {args.rows:,}")
    print(f" Iterasi per titik : {args.iterations}")
    print(f" Core terdeteksi   : {os.cpu_count()}  | worker dipakai: {n_workers}")

    # 1) Generate data
    print("\n[1/4] Membangkitkan data sensor dummy ...")
    t0 = time.perf_counter()
    data = buat_data_sensor(args.rows)
    print(f"      selesai dalam {time.perf_counter() - t0:.2f} detik")

    # 2) Serial
    print("[2/4] Menjalankan metode SERIAL (1 core) ...")
    skor_serial, t_serial = jalankan_serial(data, args.iterations)
    print(f"      waktu serial   : {t_serial:.3f} detik")

    # 3) Paralel
    print(f"[3/4] Menjalankan metode PARALEL ({n_workers} core) ...")
    skor_paralel, t_paralel = jalankan_paralel(data, args.iterations, n_workers)
    print(f"      waktu paralel  : {t_paralel:.3f} detik")

    # Verifikasi kebenaran: hasil paralel harus sama dengan serial
    sama = np.allclose(skor_serial, skor_paralel)
    speedup = t_serial / t_paralel if t_paralel > 0 else float("inf")
    efisiensi = speedup / n_workers * 100.0

    print("\n" + "=" * 78)
    print(" PERBANDINGAN KINERJA")
    print("=" * 78)
    print(f" Waktu Serial   : {t_serial:10.3f} detik")
    print(f" Waktu Paralel  : {t_paralel:10.3f} detik")
    print(f" Speedup        : {speedup:10.2f} x")
    print(f" Efisiensi      : {efisiensi:10.1f} %  (speedup / jumlah core)")
    print(f" Hasil identik  : {'YA' if sama else 'TIDAK (periksa kode!)'}")

    # Uji scaling opsional
    hasil_scaling = None
    if args.scaling:
        print("\n[Scaling] Menguji variasi jumlah worker ...")
        daftar = sorted({1, 2, 4, 8, 16, 32, n_workers} & set(range(1, n_workers + 1)))
        hasil_scaling = []
        for w in daftar:
            _, tw = jalankan_paralel(data, args.iterations, w)
            hasil_scaling.append((w, tw, t_serial / tw))
            print(f"      {w:>3} worker : {tw:8.3f} detik | speedup {t_serial / tw:5.2f}x")

    # 4) Ringkasan hasil & grafik
    print("\n[4/4] Menyusun ringkasan hasil prediksi ...")
    ringkasan = ringkas_per_wilayah(data, skor_paralel)
    cetak_ringkasan(ringkasan, skor_paralel)
    simpan_grafik(t_serial, t_paralel, speedup, n_workers, ringkasan,
                  args.rows, args.output, hasil_scaling)


if __name__ == "__main__":
    # Wajib untuk multiprocessing di Windows/macOS (metode start "spawn")
    mp.freeze_support()
    main()
