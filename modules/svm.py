"""
=============================================================================
MODUL 3: CLASSIFICATION — SUPPORT VECTOR MACHINE (SVM)
=============================================================================
Judul TA : Analisis Pengaruh Teknik Normalisasi Kata Gaul (Slang) terhadap
           Akurasi Deteksi Ujaran Kebencian Berbahasa Indonesia
=============================================================================
Implementasi MURNI Python. Tidak menggunakan Scikit-Learn atau library
klasifikasi manapun. Setiap rumus dijelaskan secara matematis.
=============================================================================

LANDASAN MATEMATIS SVM
======================

Support Vector Machine (SVM) mencari hyperplane optimal yang memaksimalkan
margin antara dua kelas. Untuk data yang tidak linearly separable,
digunakan soft-margin SVM dengan penalty parameter C.

Formulasi Primal (Optimasi yang ingin diselesaikan):
  min_{w,b,ξ}  (1/2)||w||² + C · Σᵢ ξᵢ
  subject to:  yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ,  ξᵢ ≥ 0

Formulasi Dual (yang benar-benar dioptimasi oleh SMO):
  max_α  Σᵢ αᵢ - (1/2) Σᵢ Σⱼ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
  subject to:  0 ≤ αᵢ ≤ C,  Σᵢ αᵢyᵢ = 0

Fungsi Keputusan:
  f(x) = Σᵢ (αᵢ · yᵢ · K(xᵢ, x)) + b
  Kelas = sign(f(x))  → +1 atau -1

Metode Optimasi: Sequential Minimal Optimization (SMO)
  - Mengoptimasi DUA αᵢ sekaligus per iterasi (memenuhi constraint Σαᵢyᵢ=0)
  - Lebih efisien daripada QP solver umum untuk dataset besar

Kernel yang didukung:
  1. Linear  : K(x,z) = x·z  (dot product)
  2. RBF     : K(x,z) = exp(-||x-z||² / (2σ²))
=============================================================================
"""

import math
import random


class SVMScratch:
    """
    Implementasi SVM dari nol (scratch) menggunakan algoritma SMO.

    Atribut setelah training:
      - alpha  : List[float] — Lagrange multipliers, ukuran N
                 αᵢ > 0 menandakan xᵢ adalah Support Vector
      - b      : float — bias/threshold hyperplane
      - X      : List[List[float]] — data training (N × V)
      - y      : List[int] — label training {+1, -1}
    """

    def __init__(self, kernel='linear', C=1.0, tol=1e-3, max_passes=5, sigma=0.5):
        """
        Inisialisasi Parameter SVM.

        Parameter Matematis:
        ────────────────────
        C (float):
            Penalty parameter (regularisasi). Mengontrol trade-off:
            - C besar → margin sempit, error training kecil (bisa overfit)
            - C kecil → margin lebar, toleransi error lebih besar (bisa underfit)
            Rentang umum: 0.01 ≤ C ≤ 100

        tol (float):
            Toleransi numerik untuk pengecekan kondisi KKT (Karush-Kuhn-Tucker).
            Jika |yᵢ · Eᵢ| < tol → αᵢ dianggap sudah optimal.

        max_passes (int):
            Jumlah maksimum iterasi berturut-turut TANPA perubahan alpha.
            Jika selama max_passes iterasi tidak ada alpha yang berubah,
            algoritma dianggap konvergen dan berhenti.

        sigma (float):
            Parameter bandwidth untuk kernel RBF (Gaussian).
            σ kecil → decision boundary lebih kompleks (bisa overfit)
            σ besar → decision boundary lebih smooth (bisa underfit)
            Hanya digunakan jika kernel='rbf'.
        """
        self.kernel_type = kernel
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.sigma = sigma

        # Variabel model (diisi saat training)
        self.alpha = []       # Lagrange multipliers αᵢ untuk setiap data
        self.b = 0            # Bias (threshold) hyperplane
        self.X = []           # Data training (disimpan untuk prediksi)
        self.y = []           # Label training (disimpan untuk prediksi)
        self.n_samples = 0    # N = jumlah sampel
        self.n_features = 0   # V = jumlah fitur (dimensi vocabulary TF-IDF)

    # ─── FUNGSI KERNEL (MANUAL) ──────────────────────────────────────────

    def _kernel_function(self, x1, x2):
        """
        Kernel Trick — Transformasi ruang fitur secara implisit.

        Kernel menghitung dot product di ruang fitur berdimensi tinggi
        TANPA perlu melakukan transformasi eksplisit φ(x):
          K(x₁, x₂) = φ(x₁) · φ(x₂)

        1. Linear Kernel:
           ────────────────
           K(x₁, x₂) = x₁ · x₂ = Σⱼ x₁ⱼ · x₂ⱼ
           - Dot product biasa di ruang input asli
           - Cocok untuk data yang linearly separable
           - Kompleksitas: O(V) — V = jumlah fitur

        2. RBF (Radial Basis Function) / Gaussian Kernel:
           ────────────────────────────────────────────────
           K(x₁, x₂) = exp( -||x₁ - x₂||² / (2σ²) )
           dimana:
             ||x₁ - x₂||² = Σⱼ (x₁ⱼ - x₂ⱼ)²  (squared Euclidean distance)
             σ = parameter bandwidth (self.sigma)
           - Memetakan ke ruang fitur berdimensi tak hingga
           - Dapat menangani data non-linearly separable
           - Nilai K ∈ (0, 1]: semakin dekat x₁ dan x₂ → K mendekati 1
           - Kompleksitas: O(V)

        Args:
            x1 : Vektor fitur data pertama (ukuran V)
            x2 : Vektor fitur data kedua (ukuran V)

        Returns:
            float — nilai kernel K(x₁, x₂)
        """
        if self.kernel_type == 'linear':
            # K(x₁, x₂) = Σⱼ x₁ⱼ · x₂ⱼ  (dot product)
            return sum(a * b for a, b in zip(x1, x2))

        elif self.kernel_type == 'rbf':
            # ||x₁ - x₂||² = Σⱼ (x₁ⱼ - x₂ⱼ)²
            squared_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
            # K = exp(-||x₁-x₂||² / (2σ²))
            return math.exp(-squared_dist / (2 * (self.sigma ** 2)))

        return 0

    # ─── FUNGSI KEPUTUSAN (DECISION FUNCTION) ────────────────────────────

    def _decision_function(self, x_query):
        """
        Menghitung nilai fungsi keputusan f(x) untuk satu data query.

        Rumus:
          f(x) = Σᵢ₌₁ᴺ (αᵢ · yᵢ · K(xᵢ, x)) + b

        dimana:
          αᵢ    = Lagrange multiplier untuk data ke-i
          yᵢ    = label data ke-i (+1 atau -1)
          K(·,·)= fungsi kernel
          xᵢ    = data training ke-i
          x     = data query yang akan diprediksi
          b     = bias/threshold

        Optimasi: Hanya Support Vectors (αᵢ > 0) yang berkontribusi
        pada perhitungan. Data dengan αᵢ = 0 tidak mempengaruhi
        hyperplane → bisa di-skip untuk efisiensi.

        Kompleksitas: O(|SV| × V)
          |SV| = jumlah support vectors, V = dimensi fitur

        Args:
            x_query : Vektor fitur data yang akan dihitung (ukuran V)

        Returns:
            float — nilai f(x). Tanda (sign) menentukan kelas:
                    f(x) ≥ 0 → kelas +1
                    f(x) < 0  → kelas -1
        """
        result = self.b  # Mulai dari bias
        for i in range(self.n_samples):
            if self.alpha[i] > 0:  # Hanya Support Vectors yang dihitung
                # Akumulasi: αᵢ · yᵢ · K(xᵢ, x_query)
                result += self.alpha[i] * self.y[i] * self._kernel_function(self.X[i], x_query)
        return result

    # ─── PREDIKSI ────────────────────────────────────────────────────────

    def predict(self, X_test):
        """
        Menentukan kelas berdasarkan tanda (sign) dari f(x).

        Aturan Keputusan:
          Untuk setiap x ∈ X_test:
            ŷ = sign(f(x))
            ŷ = +1  jika f(x) ≥ 0  → Kelas Positif (misal: Hate Speech)
            ŷ = -1  jika f(x) < 0  → Kelas Negatif (misal: Non-Hate Speech)

        Args:
            X_test : List[List[float]] — matriks data test (M × V)

        Returns:
            List[int] — prediksi kelas untuk setiap data test
        """
        predictions = []
        for x in X_test:
            f_x = self._decision_function(x)
            predictions.append(1 if f_x >= 0 else -1)
        return predictions

    # ─── TRAINING DENGAN SMO (SEQUENTIAL MINIMAL OPTIMIZATION) ────────────

    def train(self, X, y):
        """
        Optimasi nilai Alpha menggunakan algoritma SMO (Simplified version).

        Referensi: Andrew Ng, CS229 Lecture Notes - SMO Algorithm
        ═══════════════════════════════════════════════════════════

        INPUT:
          X : Matriks TF-IDF (N × V) — setiap baris = vektor fitur dokumen
          y : Label (N × 1) — harus bernilai +1 atau -1

        TUJUAN:
          Mencari nilai αᵢ optimal yang memaksimalkan dual objective:
            W(α) = Σᵢ αᵢ - (1/2) Σᵢ Σⱼ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
          dengan constraint: 0 ≤ αᵢ ≤ C dan Σᵢ αᵢyᵢ = 0

        ALGORITMA SMO (per iterasi):
          1. Untuk setiap data i:
             a. Hitung error: Eᵢ = f(xᵢ) - yᵢ
             b. Cek kondisi KKT (Karush-Kuhn-Tucker)
             c. Jika melanggar KKT → pilih j secara acak
             d. Hitung batas L dan H untuk αⱼ
             e. Hitung eta (η) — second derivative
             f. Update αⱼ secara analitis dan clip ke [L, H]
             g. Update αᵢ berdasarkan perubahan αⱼ
             h. Update bias b

        KONDISI KKT yang dicek:
          - yᵢ·Eᵢ < -tol DAN αᵢ < C → αᵢ bisa naik
          - yᵢ·Eᵢ > tol  DAN αᵢ > 0 → αᵢ bisa turun
        """
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.n_features = len(X[0]) if self.n_samples > 0 else 0

        # ── Inisialisasi: semua α = 0, bias b = 0 ────────────────────────
        # Pada α = 0, semua data bukan support vector.
        # SMO akan secara bertahap menaikkan α untuk data yang relevan.
        self.alpha = [0.0] * self.n_samples
        self.b = 0.0

        passes = 0  # Counter iterasi tanpa perubahan

        # ── Loop Utama SMO ────────────────────────────────────────────────
        # Berhenti jika selama max_passes iterasi berturut-turut
        # tidak ada alpha yang berubah (konvergen)
        while passes < self.max_passes:
            num_changed_alphas = 0

            for i in range(self.n_samples):
                # ── Langkah 1: Hitung Error Eᵢ ───────────────────────────
                # Eᵢ = f(xᵢ) - yᵢ
                # Error = prediksi model saat ini dikurangi label sebenarnya
                # Jika model sempurna: Eᵢ = 0 untuk semua i
                E_i = self._decision_function(self.X[i]) - self.y[i]

                # ── Langkah 2: Cek Kondisi KKT ───────────────────────────
                # Kondisi KKT (Karush-Kuhn-Tucker) untuk optimality:
                #   αᵢ = 0   → yᵢf(xᵢ) ≥ 1  (data di luar margin, classified benar)
                #   0 < αᵢ < C → yᵢf(xᵢ) = 1  (data tepat di margin = support vector)
                #   αᵢ = C   → yᵢf(xᵢ) ≤ 1  (data melanggar margin atau salah klasifikasi)
                #
                # Pelanggaran KKT (data perlu di-update):
                #   yᵢEᵢ < -tol DAN αᵢ < C → margin terlalu kecil, αᵢ harus naik
                #   yᵢEᵢ > tol  DAN αᵢ > 0 → margin terlalu besar, αᵢ harus turun
                if (self.y[i] * E_i < -self.tol and self.alpha[i] < self.C) or \
                   (self.y[i] * E_i > self.tol and self.alpha[i] > 0):

                    # ── Langkah 3: Pilih j secara acak (j ≠ i) ───────────
                    j = i
                    while j == i:
                        j = random.randint(0, self.n_samples - 1)

                    # Hitung Error Eⱼ
                    E_j = self._decision_function(self.X[j]) - self.y[j]

                    # Simpan α lama untuk menghitung perubahan (Δα)
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    # ── Langkah 4: Hitung Batas L dan H untuk αⱼ ─────────
                    # Constraint: 0 ≤ αⱼ ≤ C DAN αᵢyᵢ + αⱼyⱼ = konstan
                    # (agar Σ αₖyₖ = 0 tetap terpenuhi setelah update)
                    #
                    # Jika yᵢ ≠ yⱼ (label berbeda):
                    #   L = max(0, αⱼ - αᵢ)
                    #   H = min(C, C + αⱼ - αᵢ)
                    # Jika yᵢ = yⱼ (label sama):
                    #   L = max(0, αᵢ + αⱼ - C)
                    #   H = min(C, αᵢ + αⱼ)
                    if self.y[i] != self.y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L == H:
                        continue  # Tidak ada ruang untuk update

                    # ── Langkah 5: Hitung Eta (η) ────────────────────────
                    # η = 2K(xᵢ,xⱼ) - K(xᵢ,xᵢ) - K(xⱼ,xⱼ)
                    # η adalah turunan kedua (second derivative) dari
                    # fungsi objektif terhadap αⱼ.
                    # η < 0 → fungsi objektif konkaf → ada maksimum unik
                    # η ≥ 0 → kasus degenerasi → skip
                    eta = 2.0 * self._kernel_function(self.X[i], self.X[j]) - \
                          self._kernel_function(self.X[i], self.X[i]) - \
                          self._kernel_function(self.X[j], self.X[j])

                    if eta >= 0:
                        continue  # Skip: tidak ada maksimum unik

                    # ── Langkah 6: Update αⱼ secara analitis ─────────────
                    # Rumus update (dari turunan parsial = 0):
                    #   αⱼ_baru = αⱼ_lama - yⱼ(Eᵢ - Eⱼ) / η
                    self.alpha[j] -= (self.y[j] * (E_i - E_j)) / eta

                    # ── Langkah 7: Clip αⱼ ke rentang [L, H] ────────────
                    # Memastikan constraint box terpenuhi: L ≤ αⱼ ≤ H
                    if self.alpha[j] > H:
                        self.alpha[j] = H
                    elif self.alpha[j] < L:
                        self.alpha[j] = L

                    # Jika perubahan αⱼ terlalu kecil → skip (numerik stabil)
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # ── Langkah 8: Update αᵢ ─────────────────────────────
                    # Dari constraint: αᵢyᵢ + αⱼyⱼ = konstan
                    # → αᵢ_baru = αᵢ_lama + yᵢyⱼ(αⱼ_lama - αⱼ_baru)
                    self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])

                    # ── Langkah 9: Update Bias b ─────────────────────────
                    # Dua kandidat bias dihitung dari αᵢ dan αⱼ:
                    #
                    # b₁ = b - Eᵢ - yᵢ(αᵢ_baru - αᵢ_lama)K(xᵢ,xᵢ)
                    #                - yⱼ(αⱼ_baru - αⱼ_lama)K(xᵢ,xⱼ)
                    b1 = self.b - E_i - self.y[i] * (self.alpha[i] - alpha_i_old) * \
                         self._kernel_function(self.X[i], self.X[i]) - \
                         self.y[j] * (self.alpha[j] - alpha_j_old) * \
                         self._kernel_function(self.X[i], self.X[j])

                    # b₂ = b - Eⱼ - yᵢ(αᵢ_baru - αᵢ_lama)K(xᵢ,xⱼ)
                    #                - yⱼ(αⱼ_baru - αⱼ_lama)K(xⱼ,xⱼ)
                    b2 = self.b - E_j - self.y[i] * (self.alpha[i] - alpha_i_old) * \
                         self._kernel_function(self.X[i], self.X[j]) - \
                         self.y[j] * (self.alpha[j] - alpha_j_old) * \
                         self._kernel_function(self.X[j], self.X[j])

                    # Pemilihan bias:
                    # Jika 0 < αᵢ < C → b = b₁ (αᵢ di bound → KKT exact)
                    # Jika 0 < αⱼ < C → b = b₂ (αⱼ di bound → KKT exact)
                    # Jika keduanya di bound (0 atau C) → b = rata-rata
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    num_changed_alphas += 1

            # ── Pengecekan Konvergensi ────────────────────────────────────
            # Jika tidak ada α yang berubah → increment passes
            # Jika ada perubahan → reset passes ke 0
            # Berhenti jika passes mencapai max_passes (konvergen)
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        # ── Hasil Training ────────────────────────────────────────────────
        # Support Vectors = data dengan αᵢ > 0
        # Hanya SV yang berkontribusi pada fungsi keputusan f(x)
        n_sv = sum(1 for a in self.alpha if a > 0)

        return {"status": "Success", "support_vectors": n_sv}