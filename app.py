import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, segmentation, feature, filters
from skimage.measure import label, regionprops

class AdvancedBuildingSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentasi dan Perhitungan Luas Bangunan - Advanced")
        self.root.geometry("1600x900")
        
        self.original_image = None
        self.segmented_image = None
        self.processed_steps = {}
        self.results = []
        
        self.setup_gui()
    
    def setup_gui(self):
        # Frame utama
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Konfigurasi grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Panel kontrol (kiri)
        control_frame = ttk.LabelFrame(main_frame, text="Kontrol", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Upload gambar
        ttk.Label(control_frame, text="1. Upload Gambar Aerial:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Button(control_frame, text="Pilih Gambar", command=self.load_image).grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Input skala
        ttk.Label(control_frame, text="2. Skala Peta:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=(15, 5))
        
        scale_frame = ttk.Frame(control_frame)
        scale_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(scale_frame, text="1 pixel =").grid(row=0, column=0)
        self.scale_entry = ttk.Entry(scale_frame, width=10)
        self.scale_entry.grid(row=0, column=1, padx=5)
        self.scale_entry.insert(0, "0.5")
        
        self.unit_combo = ttk.Combobox(scale_frame, values=["m", "cm", "km"], width=5, state="readonly")
        self.unit_combo.grid(row=0, column=2)
        self.unit_combo.set("m")
        
        # Metode segmentasi
        ttk.Label(control_frame, text="3. Metode Segmentasi:", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky=tk.W, pady=(15, 5))
        
        self.method_var = tk.StringVar(value="watershed")
        methods = [
            ("Watershed + Marker", "watershed"),
            ("Adaptive Threshold", "adaptive"),
            ("Edge Detection + Morphology", "edge"),
            ("K-Means Clustering", "kmeans")
        ]
        
        for i, (text, value) in enumerate(methods):
            ttk.Radiobutton(control_frame, text=text, variable=self.method_var, 
                           value=value).grid(row=5+i, column=0, sticky=tk.W, pady=2)
        
        # Parameter segmentasi
        ttk.Label(control_frame, text="4. Parameter:", font=("Arial", 10, "bold")).grid(row=9, column=0, sticky=tk.W, pady=(15, 5))
        
        ttk.Label(control_frame, text="Min. Luas (pixel²):").grid(row=10, column=0, sticky=tk.W)
        self.min_area_entry = ttk.Entry(control_frame, width=10)
        self.min_area_entry.grid(row=11, column=0, sticky=tk.W, pady=5)
        self.min_area_entry.insert(0, "300")
        
        ttk.Label(control_frame, text="Max. Luas (pixel²):").grid(row=12, column=0, sticky=tk.W)
        self.max_area_entry = ttk.Entry(control_frame, width=10)
        self.max_area_entry.grid(row=13, column=0, sticky=tk.W, pady=5)
        self.max_area_entry.insert(0, "50000")
        
        # Pre-processing
        self.use_bilateral = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Bilateral Filter (Noise Reduction)", 
                       variable=self.use_bilateral).grid(row=14, column=0, sticky=tk.W, pady=5)
        
        self.use_clahe = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="CLAHE (Contrast Enhancement)", 
                       variable=self.use_clahe).grid(row=15, column=0, sticky=tk.W, pady=2)
        
        # Tombol proses
        ttk.Button(control_frame, text="🔍 Proses Segmentasi", 
                  command=self.process_segmentation, 
                  style="Accent.TButton").grid(row=16, column=0, sticky=(tk.W, tk.E), pady=20)
        
        # Tombol lihat tahapan
        ttk.Button(control_frame, text="👁 Lihat Tahapan Proses", 
                  command=self.show_processing_steps).grid(row=17, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Tombol ekspor
        ttk.Button(control_frame, text="💾 Ekspor Hasil", 
                  command=self.export_results).grid(row=18, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Panel gambar (tengah & kanan)
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        
        # Gambar asli
        original_frame = ttk.LabelFrame(image_frame, text="Gambar Asli", padding="5")
        original_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        self.original_canvas = tk.Canvas(original_frame, width=600, height=450, bg="white")
        self.original_canvas.pack()
        
        # Gambar segmentasi
        segmented_frame = ttk.LabelFrame(image_frame, text="Hasil Instance Segmentation", padding="5")
        segmented_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        self.segmented_canvas = tk.Canvas(segmented_frame, width=600, height=450, bg="white")
        self.segmented_canvas.pack()
        
        # Panel hasil (bawah)
        results_frame = ttk.LabelFrame(main_frame, text="Hasil Deteksi & Analisis", padding="10")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Tabel hasil
        columns = ("No", "Luas (px²)", "Luas Real", "Perimeter", "Compactness", "Aspect Ratio")
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=8)
        
        col_widths = [50, 100, 120, 100, 110, 110]
        for col, width in zip(columns, col_widths):
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=width, anchor=tk.CENTER)
        
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscroll=scrollbar.set)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Label ringkasan
        self.summary_label = ttk.Label(results_frame, text="Total Bangunan: 0 | Total Luas: 0.00 m²", 
                                      font=("Arial", 11, "bold"))
        self.summary_label.grid(row=1, column=0, columnspan=2, pady=10)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar Aerial",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.display_image(self.original_image, self.original_canvas)
            messagebox.showinfo("Sukses", "Gambar berhasil dimuat!")
    
    def display_image(self, img, canvas):
        # Resize untuk display
        h, w = img.shape[:2]
        max_width, max_height = 600, 450
        
        scale = min(max_width/w, max_height/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        canvas.delete("all")
        canvas.create_image(max_width//2, max_height//2, image=img_tk, anchor=tk.CENTER)
        canvas.image = img_tk
    
    # def preprocess_image(self, img):
    #     """Pre-processing dengan bilateral filter dan CLAHE"""
    #     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     self.processed_steps['1_grayscale'] = gray
        
    #     # Bilateral filter untuk noise reduction dengan preservasi edge
    #     if self.use_bilateral.get():
    #         gray = cv2.bilateralFilter(gray, 9, 75, 75)
    #         self.processed_steps['2_bilateral'] = gray
        
    #     # CLAHE untuk contrast enhancement
    #     if self.use_clahe.get():
    #         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #         gray = clahe.apply(gray)
    #         self.processed_steps['3_clahe'] = gray
        
    #     return gray
    
    def preprocess_image(self, img):
        """Pre-processing yang lebih kuat untuk citra aerial"""
        # Konversi ke LAB untuk memisahkan pencahayaan (L) dari warna (A,B)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Gunakan kanal L untuk struktur, tapi kurangi noise dengan Median Blur
        denoised_l = cv2.medianBlur(l, 5)
        
        # Bilateral Filter tetap bagus untuk menjaga tepi (edge)
        if self.use_bilateral.get():
            denoised_l = cv2.bilateralFilter(denoised_l, 9, 75, 75)
            
        # CLAHE pada kanal L (Luminance) memberikan kontras tanpa merusak warna
        if self.use_clahe.get():
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            denoised_l = clahe.apply(denoised_l)
            
        self.processed_steps['1_enhanced_l_channel'] = denoised_l
        return denoised_l
    
    # def watershed_segmentation(self, gray):
    #     """Watershed segmentation dengan marker-based approach"""
    #     # Otsu threshold
    #     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #     self.processed_steps['4_otsu_threshold'] = binary
        
    #     # Morphological operations
    #     kernel = np.ones((3, 3), np.uint8)
    #     opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    #     self.processed_steps['5_morphology_opening'] = opening
        
    #     # Sure background area
    #     sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
    #     # Distance transform untuk menemukan sure foreground
    #     dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    #     self.processed_steps['6_distance_transform'] = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    #     _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    #     sure_fg = np.uint8(sure_fg)
        
    #     # Unknown region
    #     unknown = cv2.subtract(sure_bg, sure_fg)
        
    #     # Marker labelling
    #     _, markers = cv2.connectedComponents(sure_fg)
    #     markers = markers + 1
    #     markers[unknown == 255] = 0
        
    #     # Apply watershed
    #     markers = cv2.watershed(self.original_image, markers)
        
    #     # Create mask
    #     mask = np.zeros_like(gray)
    #     mask[markers > 1] = 255
        
    #     self.processed_steps['7_watershed_result'] = mask
    #     return mask
    
    def watershed_segmentation(self, gray):
        """Improved Watershed dengan Marker Control"""
        # 1. Thresholding menggunakan Multi-Otsu atau Li
        thresh = filters.threshold_otsu(gray)
        binary = gray < thresh # Biasanya bangunan lebih gelap/terang tergantung jenis atap
        
        # Pastikan background adalah 0
        if np.mean(binary) > 0.5:
            binary = np.logical_not(binary)
            
        # 2. Pembersihan Noise dengan Morphological Reconstruction
        # Menghapus objek kecil tapi menjaga bentuk objek besar
        seed = morphology.erosion(binary, morphology.square(3))
        clean_binary = morphology.reconstruction(seed, binary)
        clean_binary = (clean_binary * 255).astype(np.uint8)
        
        self.processed_steps['4_clean_binary'] = clean_binary

        # 3. Distance Transform & Peak Finding
        # Meningkatkan margin agar objek berdempetan terpisah
        dist_transform = ndimage.distance_transform_edt(clean_binary)
        # H-minima/maxima: hanya ambil puncak yang signifikan
        local_max = feature.peak_local_max(dist_transform, min_distance=15, 
                                          labels=clean_binary)
        
        # 4. Markers
        markers = np.zeros(dist_transform.shape, dtype=int)
        for i, pt in enumerate(local_max):
            markers[pt[0], pt[1]] = i + 1
            
        # 5. Watershed pada Gradient
        # Watershed bekerja paling baik pada image gradient (tepi)
        elevation_map = filters.sobel(gray)
        labels = segmentation.watershed(elevation_map, markers, mask=clean_binary)
        
        # Konversi labels ke binary mask untuk diproses fungsi utama
        mask = np.zeros_like(gray)
        mask[labels > 0] = 255
        
        self.processed_steps['5_elevation_map'] = (elevation_map * 255).astype(np.uint8)
        self.processed_steps['6_watershed_final'] = mask
        return mask
    
    def adaptive_segmentation(self, gray):
        """Adaptive threshold segmentation"""
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        self.processed_steps['4_adaptive_threshold'] = binary
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        
        self.processed_steps['5_morphology'] = morph
        return morph
    
    def edge_segmentation(self, gray):
        """Edge detection dengan Canny + morphology"""
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        self.processed_steps['4_canny_edges'] = edges
        
        # Dilate edges untuk menghubungkan garis
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Flood fill dari edges
        mask = dilated.copy()
        h, w = mask.shape
        flood_mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(mask, flood_mask, (0, 0), 255)
        
        # Invert
        mask = cv2.bitwise_not(mask)
        
        # Combine dengan edges
        result = cv2.bitwise_or(dilated, mask)
        
        # Morphological closing
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        self.processed_steps['5_edge_morphology'] = result
        return result
    
    def kmeans_segmentation(self, gray):
        """K-means clustering segmentation"""
        # Reshape untuk K-means
        pixel_values = gray.reshape((-1, 1))
        pixel_values = np.float32(pixel_values)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 3
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        segmented = segmented.reshape(gray.shape)
        
        self.processed_steps['4_kmeans_clusters'] = segmented
        
        # Threshold pada cluster dengan intensitas rendah (bangunan)
        _, binary = cv2.threshold(segmented, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphology
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        self.processed_steps['5_kmeans_threshold'] = binary
        return binary
    
    def process_segmentation(self):
        if self.original_image is None:
            messagebox.showwarning("Peringatan", "Silakan upload gambar terlebih dahulu!")
            return
        
        try:
            self.processed_steps = {}
            
            # Ambil parameter
            min_area = int(self.min_area_entry.get())
            max_area = int(self.max_area_entry.get())
            scale_value = float(self.scale_entry.get())
            unit = self.unit_combo.get()
            method = self.method_var.get()
            
            # Konversi ke meter
            if unit == "cm":
                scale_value = scale_value / 100
            elif unit == "km":
                scale_value = scale_value * 1000
            
            # Pre-processing
            gray = self.preprocess_image(self.original_image)
            
            # Segmentasi berdasarkan metode
            if method == "watershed":
                mask = self.watershed_segmentation(gray)
            elif method == "adaptive":
                mask = self.adaptive_segmentation(gray)
            elif method == "edge":
                mask = self.edge_segmentation(gray)
            elif method == "kmeans":
                mask = self.kmeans_segmentation(gray)
            
            # Instance segmentation - label setiap objek
            labeled_mask = label(mask)
            regions = regionprops(labeled_mask)
            
            # Filter regions
            valid_regions = [r for r in regions if min_area <= r.area <= max_area]
            
            # Buat gambar hasil dengan warna berbeda untuk setiap instance
            result_img = self.original_image.copy()
            overlay = self.original_image.copy()
            
            # Hapus hasil sebelumnya
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            self.results = []
            total_area = 0
            
            # Generate colormap
            np.random.seed(42)
            colors = np.random.randint(50, 255, (len(valid_regions), 3))
            
            for i, region in enumerate(valid_regions):
                area_pixel = region.area
                area_real = area_pixel * (scale_value ** 2)
                perimeter = region.perimeter
                
                # Kompactness (circularity)
                compactness = (4 * np.pi * area_pixel) / (perimeter ** 2) if perimeter > 0 else 0
                
                # Aspect ratio
                minr, minc, maxr, maxc = region.bbox
                aspect_ratio = (maxc - minc) / (maxr - minr) if (maxr - minr) > 0 else 0
                
                # Warna untuk instance ini
                color = tuple(int(c) for c in colors[i])
                
                # Gambar kontur dan fill dengan transparansi
                mask_region = (labeled_mask == region.label).astype(np.uint8)
                contours, _ = cv2.findContours(mask_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Fill dengan warna semi-transparan
                cv2.drawContours(overlay, contours, -1, color, -1)
                # Border
                cv2.drawContours(result_img, contours, -1, (255, 255, 255), 2)
                
                # Label nomor
                cy, cx = region.centroid
                cv2.putText(result_img, str(i+1), (int(cx)-15, int(cy)+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Bounding box
                cv2.rectangle(result_img, (minc, minr), (maxc, maxr), (0, 255, 0), 1)
                
                self.results.append({
                    'no': i+1,
                    'area_pixel': area_pixel,
                    'area_real': area_real,
                    'perimeter': perimeter,
                    'compactness': compactness,
                    'aspect_ratio': aspect_ratio
                })
                
                self.results_tree.insert("", tk.END, values=(
                    i+1,
                    f"{area_pixel:.0f}",
                    f"{area_real:.2f} m²",
                    f"{perimeter:.2f}",
                    f"{compactness:.3f}",
                    f"{aspect_ratio:.2f}"
                ))
                
                total_area += area_real
            
            # Blend overlay dengan gambar asli
            result_img = cv2.addWeighted(overlay, 0.3, result_img, 0.7, 0)
            
            self.segmented_image = result_img
            self.processed_steps['8_final_segmentation'] = result_img
            self.display_image(result_img, self.segmented_canvas)
            
            # Update ringkasan
            avg_area = total_area / len(valid_regions) if valid_regions else 0
            self.summary_label.config(
                text=f"Total Bangunan: {len(valid_regions)} | Total Luas: {total_area:.2f} m² | Rata-rata: {avg_area:.2f} m²"
            )
            
            messagebox.showinfo("Sukses", 
                              f"Instance segmentation selesai!\n\nMetode: {method.upper()}\nDitemukan: {len(valid_regions)} bangunan\nTotal Luas: {total_area:.2f} m²")
        
        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan:\n{str(e)}")
    
    def show_processing_steps(self):
        """Tampilkan window dengan semua tahapan processing"""
        if not self.processed_steps:
            messagebox.showwarning("Peringatan", "Belum ada proses yang dilakukan!")
            return
        
        steps_window = tk.Toplevel(self.root)
        steps_window.title("Tahapan Pemrosesan")
        steps_window.geometry("1200x800")
        
        # Frame dengan scrollbar
        canvas = tk.Canvas(steps_window)
        scrollbar = ttk.Scrollbar(steps_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Tampilkan setiap step
        for i, (step_name, step_img) in enumerate(self.processed_steps.items()):
            frame = ttk.LabelFrame(scrollable_frame, text=step_name.replace('_', ' ').title(), padding="5")
            frame.grid(row=i//3, column=i%3, padx=5, pady=5, sticky=(tk.W, tk.E))
            
            # Convert untuk display
            if len(step_img.shape) == 2:
                display_img = cv2.cvtColor(step_img, cv2.COLOR_GRAY2RGB)
            else:
                display_img = step_img
            
            h, w = display_img.shape[:2]
            scale = min(350/w, 250/h)
            new_w, new_h = int(w*scale), int(h*scale)
            
            img_resized = cv2.resize(display_img, (new_w, new_h))
            img_pil = Image.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            label = tk.Label(frame, image=img_tk)
            label.image = img_tk
            label.pack()
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def export_results(self):
        if not self.results:
            messagebox.showwarning("Peringatan", "Belum ada hasil untuk diekspor!")
            return
        
        try:
            # Simpan gambar segmentasi
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            
            if file_path:
                # Simpan gambar
                img_bgr = cv2.cvtColor(self.segmented_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, img_bgr)
                
                # Simpan laporan CSV
                csv_path = file_path.rsplit('.', 1)[0] + '_laporan.csv'
                with open(csv_path, 'w', encoding='utf-8') as f:
                    f.write("No,Luas (pixel²),Luas (m²),Perimeter,Compactness,Aspect Ratio\n")
                    for r in self.results:
                        f.write(f"{r['no']},{r['area_pixel']:.0f},{r['area_real']:.2f},"
                               f"{r['perimeter']:.2f},{r['compactness']:.3f},{r['aspect_ratio']:.2f}\n")
                    
                    total_area = sum(r['area_real'] for r in self.results)
                    avg_area = total_area / len(self.results)
                    f.write(f"\nTotal Bangunan,{len(self.results)}\n")
                    f.write(f"Total Luas (m²),{total_area:.2f}\n")
                    f.write(f"Rata-rata Luas (m²),{avg_area:.2f}\n")
                
                # Simpan processing steps
                steps_dir = file_path.rsplit('.', 1)[0] + '_steps'
                import os
                os.makedirs(steps_dir, exist_ok=True)
                
                for step_name, step_img in self.processed_steps.items():
                    step_path = os.path.join(steps_dir, f"{step_name}.png")
                    if len(step_img.shape) == 2:
                        cv2.imwrite(step_path, step_img)
                    else:
                        cv2.imwrite(step_path, cv2.cvtColor(step_img, cv2.COLOR_RGB2BGR))
                
                messagebox.showinfo("Sukses", 
                                  f"Hasil berhasil diekspor!\n\nGambar: {file_path}\nLaporan: {csv_path}\nTahapan: {steps_dir}/")
        
        except Exception as e:
            messagebox.showerror("Error", f"Gagal mengekspor:\n{str(e)}")

def main():
    root = tk.Tk()
    app = AdvancedBuildingSegmentationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()