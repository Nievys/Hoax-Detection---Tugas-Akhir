import math
import random

class SVMScratch:
    def __init__(self, kernel='linear', C=1.0, tol=1e-3, max_passes=5, sigma=0.5, seed=42):
        self.kernel_type = kernel
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.sigma = sigma
        self.seed = seed

        # Variabel model
        self.alpha = []       # Lagrange multipliers αᵢ untuk setiap data
        self.b = 0            # Bias (threshold) hyperplane
        self.X = []           # Data training (disimpan untuk prediksi)
        self.y = []           # Label training (disimpan untuk prediksi)
        self.n_samples = 0    # N = jumlah sampel
        self.n_features = 0   # V = jumlah fitur (dimensi vocabulary TF-IDF)

    # FUNGSI KERNEL
    def _kernel_function(self, x1, x2):
        if self.kernel_type == 'linear':
            # K(x₁, x₂) = Σⱼ x₁ⱼ · x₂ⱼ  (dot product)
            return sum(a * b for a, b in zip(x1, x2))

        elif self.kernel_type == 'rbf':
            # ||x₁ - x₂||² = Σⱼ (x₁ⱼ - x₂ⱼ)²
            squared_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
            # K = exp(-||x₁-x₂||² / (2σ²))
            return math.exp(-squared_dist / (2 * (self.sigma ** 2)))

        return 0

    # FUNGSI KEPUTUSAN (DECISION FUNCTION)
    def _decision_function(self, x_query):
        result = self.b  # Mulai dari bias
        for i in range(self.n_samples):
            if self.alpha[i] > 0:  # Hanya Support Vectors yang dihitung
                # Akumulasi: αᵢ · yᵢ · K(xᵢ, x_query)
                result += self.alpha[i] * self.y[i] * self._kernel_function(self.X[i], x_query)
        return result

    # PREDIKSI
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            f_x = self._decision_function(x)
            predictions.append(1 if f_x >= 0 else -1)
        return predictions

    def predict_proba(self, X_test):
        all_proba = []
        for x in X_test:
            f_x = self._decision_function(x)
            # Cegah overflow pada math.exp
            if f_x > 100:
                p1 = 1.0
            elif f_x < -100:
                p1 = 0.0
            else:
                p1 = 1.0 / (1.0 + math.exp(-f_x))
            
            p_minus1 = 1.0 - p1
            all_proba.append({1: p1, -1: p_minus1})
        return all_proba

    def _decision_function_train(self, i):
        result = self.b
        for k in range(self.n_samples):
            if self.alpha[k] > 0:
                result += self.alpha[k] * self.y[k] * self.K_matrix[k][i]
        return result

    # TRAINING DENGAN SMO (SEQUENTIAL MINIMAL OPTIMIZATION)
    def train(self, X, y):
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.n_features = len(X[0]) if self.n_samples > 0 else 0

        # Reset seed PRNG agar hasil SELALU konsisten
        random.seed(self.seed)

        # Precompute Kernel Gram Matrix
        # Mencegah perhitungan K(x_i, x_j) berulang-ulang yang sangat mahal
        # Kompleksitas precompute: O(N² × V). Mengubah iterasi SMO menjadi O(N)
        self.K_matrix = [[0.0] * self.n_samples for _ in range(self.n_samples)]
        for i in range(self.n_samples):
            self.K_matrix[i][i] = self._kernel_function(self.X[i], self.X[i])
            for j in range(i + 1, self.n_samples):
                val = self._kernel_function(self.X[i], self.X[j])
                self.K_matrix[i][j] = val
                self.K_matrix[j][i] = val

        # Inisialisasi: semua α = 0, bias b = 0
        # Pada α = 0, semua data bukan support vector.
        # SMO akan secara bertahap menaikkan α untuk data yang relevan.
        self.alpha = [0.0] * self.n_samples
        self.b = 0.0

        passes = 0  # Counter iterasi tanpa perubahan

        # Loop Utama SMO
        # Berhenti jika selama max_passes iterasi berturut-turut
        # tidak ada alpha yang berubah (konvergen)
        while passes < self.max_passes:
            num_changed_alphas = 0

            for i in range(self.n_samples):
                # Langkah 1: Hitung Error Eᵢ
                # Eᵢ = f(xᵢ) - yᵢ
                # Error = prediksi model saat ini dikurangi label sebenarnya
                # Jika model sempurna: Eᵢ = 0 untuk semua i
                E_i = self._decision_function_train(i) - self.y[i]

                # Langkah 2: Cek Kondisi KKT
                if (self.y[i] * E_i < -self.tol and self.alpha[i] < self.C) or \
                   (self.y[i] * E_i > self.tol and self.alpha[i] > 0):

                    # Langkah 3: Pilih j secara acak (j ≠ i)
                    j = i
                    while j == i:
                        j = random.randint(0, self.n_samples - 1)

                    # Hitung Error Eⱼ
                    E_j = self._decision_function_train(j) - self.y[j]

                    # Simpan α lama untuk menghitung perubahan (Δα)
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    # Langkah 4: Hitung Batas L dan H untuk αⱼ
                    if self.y[i] != self.y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L == H:
                        continue  # Tidak ada ruang untuk update

                    # Langkah 5: Hitung Eta (η)
                    # η = 2K(xᵢ,xⱼ) - K(xᵢ,xᵢ) - K(xⱼ,xⱼ)
                    eta = 2.0 * self.K_matrix[i][j] - self.K_matrix[i][i] - self.K_matrix[j][j]

                    if eta >= 0:
                        continue  # Skip: tidak ada maksimum unik

                    # Langkah 6: Update αⱼ secara analitis
                    self.alpha[j] -= (self.y[j] * (E_i - E_j)) / eta

                    # Langkah 7: Clip αⱼ ke rentang [L, H]
                    if self.alpha[j] > H:
                        self.alpha[j] = H
                    elif self.alpha[j] < L:
                        self.alpha[j] = L

                    # Jika perubahan αⱼ terlalu kecil → skip (numerik stabil)
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # Langkah 8: Update αᵢ
                    self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])

                    # Langkah 9: Update Bias b
                    b1 = self.b - E_i - self.y[i] * (self.alpha[i] - alpha_i_old) * \
                         self.K_matrix[i][i] - \
                         self.y[j] * (self.alpha[j] - alpha_j_old) * \
                         self.K_matrix[i][j]

                    b2 = self.b - E_j - self.y[i] * (self.alpha[i] - alpha_i_old) * \
                         self.K_matrix[i][j] - \
                         self.y[j] * (self.alpha[j] - alpha_j_old) * \
                         self.K_matrix[j][j]

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    num_changed_alphas += 1

            # Pengecekan Konvergensi
            # Jika tidak ada α yang berubah → increment passes
            # Jika ada perubahan → reset passes ke 0
            # Berhenti jika passes mencapai max_passes (konvergen)
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        # Hasil Training
        # Support Vectors = data dengan αᵢ > 0
        # Hanya SV yang berkontribusi pada fungsi keputusan f(x)
        n_sv = sum(1 for a in self.alpha if a > 0)

        return {"status": "Success", "support_vectors": n_sv}