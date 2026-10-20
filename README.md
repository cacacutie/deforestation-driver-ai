# Deforestation Driver Classification Using CNN

Prototipe pembelajaran untuk mengklasifikasikan kategori penyebab langsung kehilangan tutupan hutan dari citra satelit RGB menggunakan CNN (TensorFlow/Keras).

## Kaitan dengan studi kasus
Proyek ini dibuat sebagai implementasi awal dari materi *Attributing Direct Drivers to Deforestation Alerts with Satellite Imagery and AI* (20 Oktober 2026). Dataset referensi adalah ForestNet dari Stanford ML Group.

**Batasan penting:** baseline ini menggunakan citra RGB yang telah disiapkan dan label dataset ForestNet. Ini bukan implementasi penuh pemantauan perubahan bulanan Sentinel-1/Sentinel-2, bukan sistem deteksi deforestasi otomatis dari seluruh wilayah, dan bukan alat untuk membuktikan pelanggaran. Prediksi perlu divalidasi dengan sumber independen/lapangan.

## Dataset
- Sumber resmi: https://stanfordmlgroup.github.io/projects/forestnet/
- Paper: https://arxiv.org/abs/2011.05479
- Dataset ForestNet memiliki empat kelompok label utama: Plantation, Smallholder Agriculture, Grassland/shrubland, dan Other.
- Ikuti petunjuk lisensi dan atribusi dari sumber dataset.
- Jangan mengunggah ulang dataset berukuran besar ke repository GitHub.

## Struktur folder
Siapkan citra yang sudah diorganisasikan berdasarkan label asli ke struktur berikut:

```text
data/
  train/
    Plantation/
    Smallholder Agriculture/
    Grassland shrubland/
    Other/
  val/
    Plantation/
    Smallholder Agriculture/
    Grassland shrubland/
    Other/
  test/
    Plantation/
    Smallholder Agriculture/
    Grassland shrubland/
    Other/
```

Nama folder di setiap split harus identik. Struktur di atas adalah format masukan yang diharapkan kode, bukan struktur arsip asli dataset. Sesuaikan nama kelas dengan label yang benar-benar ada. Gunakan split resmi jika tersedia dan hindari citra/lokasi/kejadian yang sama tersebar antara train, validation, dan test.

## Menjalankan di Google Colab
1. Buka https://colab.research.google.com/ dan buat notebook baru.
2. Unggah `src/train.py` dan sesuaikan `DATA_DIR` bila dataset berada di lokasi lain, atau unggah proyek dan data yang telah disiapkan ke sesi Colab.
3. Instal dependensi:

   ```python
   !pip install -r requirements.txt
   ```

4. Jalankan pelatihan dari direktori proyek:

   ```python
   !python src/train.py
   ```

5. Setelah pelatihan, periksa `outputs/accuracy.png`, `outputs/loss.png`, `outputs/confusion_matrix.png`, dan `outputs/classification_report.json`.

Untuk menjalankan `predict.py`:

```bash
python src/predict.py path/to/image.jpg
```

## Hasil
Jangan mengisi angka akurasi di laporan sebelum model benar-benar dijalankan. Catat metrik yang dihasilkan dari test set. Model baseline ditujukan untuk pembelajaran, bukan untuk penggunaan operasional tanpa validasi.

## Teknologi
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

## Referensi
Irvin, J. et al. (2020). *ForestNet: Classifying Drivers of Deforestation in Indonesia Using Deep Learning on Satellite Imagery*. https://arxiv.org/abs/2011.05479

ForestNet dataset: https://stanfordmlgroup.github.io/projects/forestnet/
