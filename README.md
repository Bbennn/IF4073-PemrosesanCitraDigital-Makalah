# IF4073-PemrosesanCitraDigital-Makalah
Perhitungan Hampiran Luas Tanah atau Gedung dengan Pemanfaatan Image Segmentation

## Deskripsi
Proyek ini bertujuan untuk membangun sebuah aplikasi berbasis GUI yang mampu melakukan **segmentasi bangunan dari citra aerial** serta menghitung **hampiran luas bangunan** secara otomatis. Aplikasi memanfaatkan berbagai teknik **Pemrosesan Citra Digital (PCD)** untuk meningkatkan kualitas citra, melakukan segmentasi, serta mengekstraksi informasi geometris dari setiap bangunan yang terdeteksi.

Aplikasi ini dikembangkan sebagai bagian dari tugas mata kuliah **IF4073 Pemrosesan Citra Digital**.

---

## Fitur Utama
- Input citra aerial (JPEG, PNG, BMP, TIFF)
- Pre-processing citra:
  - Konversi ruang warna RGB ke CIELAB
  - Median blur untuk reduksi noise
  - Bilateral filter untuk mempertahankan tepi
  - CLAHE untuk peningkatan kontras
- Metode segmentasi:
  - Marker-Controlled Watershed
  - Adaptive Thresholding
  - Edge Detection + Morphological Operations
  - K-Means Clustering
- Instance segmentation dan pelabelan bangunan
- Perhitungan properti geometris bangunan:
  - Luas (pixel² dan satuan nyata)
  - Perimeter
  - Compactness
  - Aspect ratio
- Visualisasi hasil segmentasi
- Ekspor hasil:
  - Gambar segmentasi
  - Laporan CSV
  - Citra tahapan pemrosesan

---

## Alur Proses Kerja
1. Input citra aerial  
2. Pre-processing  
   - RGB → LAB  
   - Median blur  
   - Bilateral filter  
   - CLAHE  
3. Segmentasi (Watershed / Adaptive / Edge / K-Means)  
4. Operasi morfologi  
5. Instance labeling  
6. Filtering berdasarkan luas  
7. Visualisasi dan pengukuran  
8. Ekspor hasil  

---

### Pustaka yang Digunakan
- `tkinter`
- `opencv-python`
- `numpy`
- `scipy`
- `scikit-image`
- `Pillow`

---

