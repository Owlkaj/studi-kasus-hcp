#!/bin/bash
#=============================================================================
# job_submission.sh
# Contoh batch script ala SLURM untuk menjalankan simulasi HPC:
# "Prediksi Indeks Bahaya Banjir dan Polusi Udara (PM2.5) Berbasis Multi-Sensor"
#
# Cara pakai di cluster/supercomputer yang memakai SLURM:
#     sbatch job_submission.sh        # kirim job ke antrean
#     squeue -u $USER                 # pantau status job
#     scancel <JOBID>                 # batalkan job
#     cat logs/banjir_<JOBID>.out     # lihat keluaran program
#
# CATATAN: nilai partition, modul, dan batas sumber daya di bawah ini
# bersifat CONTOH. Sesuaikan dengan kebijakan cluster tempat Anda bekerja.
#=============================================================================

#SBATCH --job-name=hpc_banjir_pm25        # nama job di antrean
#SBATCH --output=logs/banjir_%j.out       # file stdout (%j = Job ID)
#SBATCH --error=logs/banjir_%j.err        # file stderr
#SBATCH --partition=compute               # partisi/antrean (sesuaikan)
#SBATCH --nodes=1                         # multiprocessing hanya 1 node
#SBATCH --ntasks=1                        # 1 proses utama Python
#SBATCH --cpus-per-task=16                # jumlah core untuk multiprocessing.Pool
#SBATCH --mem=16G                         # memori RAM yang diminta
#SBATCH --time=00:30:00                   # batas waktu (HH:MM:SS)
##SBATCH --mail-type=END,FAIL             # (opsional) notifikasi email
##SBATCH --mail-user=email@kampus.ac.id   # (opsional) alamat email

set -euo pipefail

echo "=============================================="
echo " Job ID        : ${SLURM_JOB_ID:-lokal}"
echo " Nama job      : ${SLURM_JOB_NAME:-lokal}"
echo " Node          : ${SLURM_NODELIST:-$(hostname)}"
echo " CPU per task  : ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo " Mulai         : $(date)"
echo "=============================================="

# Pindah ke direktori tempat job dikirim, lalu siapkan folder log
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# --- Memuat lingkungan perangkat lunak (module system ala cluster) ---------
# Hapus tanda # di bawah jika cluster Anda menyediakan 'module'.
# module purge
# module load python/3.11

# --- (Opsional) virtual environment ----------------------------------------
# python3 -m venv .venv
# source .venv/bin/activate
# pip install --quiet numpy matplotlib

# Cegah library numerik membuat thread tambahan di dalam tiap proses
# (agar tidak bentrok dengan multiprocessing: 1 proses = 1 core).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Menjalankan simulasi ---------------------------------------------------
# Program otomatis membaca SLURM_CPUS_PER_TASK sebagai jumlah worker.
python3 simulasi_hpc_banjir.py --rows 1000000 --scaling

echo "=============================================="
echo " Selesai       : $(date)"
echo " Hasil grafik  : benchmark_result.png"
echo "=============================================="
