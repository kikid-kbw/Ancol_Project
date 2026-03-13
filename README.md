# Ancol Project

Repository ini digunakan untuk menyimpan pekerjaan pemodelan Ancol agar bisa dikerjakan sinkron antara PC kantor dan PC rumah menggunakan GitHub.

## Struktur Folder

### 01_Hidrodinamika
Berisi file kerja pemodelan hidrodinamika, termasuk:
- grid
- batimetri
- boundary condition
- file setup model
- file pendukung analisis

### 02_Extreme_wave
Berisi file kerja pemodelan gelombang ekstrem, termasuk:
- skenario arah gelombang
- return period
- grid dan depth
- boundary gelombang
- file setup SWAN / D-Waves
- file hasil olahan pendukung

## Tujuan Repository
Repository ini dibuat agar:
- pekerjaan dapat ditrace melalui riwayat commit
- perubahan file dapat terdokumentasi
- pekerjaan dapat dilanjutkan di perangkat lain tanpa copy manual lewat harddisk eksternal

## Workflow Kerja
Urutan kerja yang digunakan:

1. Pull terlebih dahulu dari GitHub
2. Edit / tambah / hapus file yang diperlukan
3. Commit perubahan di lokal
4. Push ke GitHub

## Aturan Singkat
- File input, script, konfigurasi, grid, boundary, dan dokumentasi disimpan di repository
- File output besar, file sementara, dan file hasil run yang bisa dibuat ulang sebaiknya tidak dimasukkan ke repository
- Gunakan pesan commit yang jelas, misalnya:
  - update hydrodynamic grid
  - revise wave boundary setup
  - add bathymetry support files

## Catatan
Branch utama yang digunakan saat ini adalah `master`.
Folder utama repository:
- `01_Hidrodinamika`
- `02_Extreme_wave`
