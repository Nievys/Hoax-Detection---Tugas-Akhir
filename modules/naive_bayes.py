"""
=============================================================================
MODUL 4: CLASSIFICATION — MULTINOMIAL NAIVE BAYES
=============================================================================
Judul TA : Analisis Pengaruh Teknik Normalisasi Kata Gaul (Slang) terhadap
           Akurasi Deteksi Ujaran Kebencian Berbahasa Indonesia
=============================================================================
Implementasi MURNI Python. Tidak menggunakan Scikit-Learn atau library
klasifikasi manapun. Setiap rumus dijelaskan secara matematis.
=============================================================================

LANDASAN MATEMATIS — MULTINOMIAL NAIVE BAYES
=============================================

Teorema Bayes (pondasi klasifikasi probabilistik):
  P(c|d) = P(d|c) · P(c) / P(d)

  dimana:
    P(c|d) = posterior  — probabilitas kelas c DIBERIKAN dokumen d
    P(d|c) = likelihood — probabilitas dokumen d JIKA kelas adalah c
    P(c)   = prior      — probabilitas kelas c secara umum (dari data training)
    P(d)   = evidence   — probabilitas dokumen d (konstan untuk semua kelas)

Karena P(d) sama untuk semua kelas, keputusan klasifikasi menjadi:
  ĉ = argmax_c  P(c) · P(d|c)

ASUMSI INDEPENDENSI NAIF (Naive Independence Assumption):
  Naive Bayes mengasumsikan bahwa setiap fitur (kata/term) dalam dokumen
  bersifat INDEPENDEN satu sama lain DIBERIKAN kelas:

    P(d|c) = P(t₁, t₂, ..., tₙ | c) = ∏ᵢ P(tᵢ|c)

  Ini adalah asumsi "naif" (naive) karena dalam realitas, kata-kata dalam
  kalimat TIDAK benar-benar independen (misal: "tidak suka" vs "sangat suka").

  NAMUN, meskipun asumsi ini secara teoritis salah, Naive Bayes terbukti
  efektif untuk klasifikasi teks karena:
    1. Estimasi parameter lebih stabil (menghindari curse of dimensionality)
    2. Tidak memerlukan estimasi joint probability berdimensi tinggi
    3. Bekerja baik untuk data dengan fitur sparse (seperti TF-IDF/BoW)

Model Multinomial:
  Dalam model Multinomial, dokumen direpresentasikan sebagai vektor frekuensi
  kata (Bag-of-Words atau TF-IDF weights). Likelihood setiap kata dihitung
  berdasarkan distribusi multinomial:

    P(tᵢ|c) = (count(tᵢ, c) + α) / (Σⱼ count(tⱼ, c) + α·|V|)

  dimana:
    count(tᵢ, c) = total bobot/frekuensi kata tᵢ di semua dokumen kelas c
    α            = parameter Laplace smoothing (default: 1)
    |V|          = ukuran vocabulary (jumlah kata unik)

Laplace Smoothing (Additive Smoothing):
  Tanpa smoothing, jika kata tᵢ tidak pernah muncul di kelas c:
    P(tᵢ|c) = 0  →  P(d|c) = 0 (karena ada perkalian)
  Ini membuat seluruh probabilitas menjadi 0, meskipun kata lain relevan.

  Laplace smoothing menambahkan α ke setiap count:
    P(tᵢ|c) = (count(tᵢ,c) + α) / (total_count(c) + α·|V|)
  Sehingga tidak ada probabilitas yang benar-benar 0.

Log-Space Computation:
  Karena perkalian banyak probabilitas kecil → numerik underflow,
  kita bekerja di log-space:

    log P(c|d) ∝ log P(c) + Σᵢ wᵢ · log P(tᵢ|c)

  dimana wᵢ adalah bobot TF-IDF dari term tᵢ dalam dokumen d.
  Ini mengubah perkalian menjadi penjumlahan (lebih stabil secara numerik).
=============================================================================
"""

import math


class MultinomialNBScratch:
    """
    Implementasi Multinomial Naive Bayes dari nol (scratch).

    Atribut setelah training:
      - classes        : List[int] — kelas unik yang ada ({+1, -1} atau {0, 1})
      - class_log_prior: Dict[int, float] — log prior probability per kelas
      - feature_log_prob: Dict[int, List[float]] — log likelihood per fitur per kelas
      - n_features     : int — jumlah fitur (ukuran vocabulary |V|)
      - alpha          : float — parameter Laplace smoothing
    """

    def __init__(self, alpha=1.0):
        """
        Inisialisasi parameter Multinomial Naive Bayes.

        Parameter:
        ──────────
        alpha (float):
            Parameter Laplace Smoothing (additive smoothing).
            - α = 1 → Laplace smoothing standar
            - α < 1 → Lidstone smoothing (lebih agresif)
            - α = 0 → tanpa smoothing (TIDAK direkomendasikan, risiko P=0)

            Rumus dengan smoothing:
              P(tᵢ|c) = (count(tᵢ, c) + α) / (Σⱼ count(tⱼ, c) + α·|V|)

            Efek: α semakin besar → distribusi semakin uniform (merata).
        """
        self.alpha = alpha

        # Variabel model (diisi saat training)
        self.classes = []               # Kelas unik: e.g. [0, 1] atau [-1, 1]
        self.class_log_prior = {}       # log P(c) untuk setiap kelas c
        self.feature_log_prob = {}      # log P(tᵢ|c) untuk setiap fitur i, kelas c
        self.class_count = {}           # Jumlah dokumen per kelas (untuk info)
        self.n_features = 0             # |V| = ukuran vocabulary
        self.feature_names = []         # Nama fitur (opsional, untuk interpretasi)

    # ─── TAHAP 1: HITUNG PRIOR PROBABILITY ───────────────────────────────

    def _compute_prior(self, y):
        """
        Menghitung Prior Probability P(c) untuk setiap kelas.

        Rumus:
          P(c) = Nₖ / N

        dimana:
          Nₖ = jumlah dokumen yang berlabel kelas c
          N  = total jumlah dokumen dalam training set

        Interpretasi:
          Prior merepresentasikan distribusi kelas SEBELUM melihat
          konten dokumen. Jika 60% data adalah kelas 1 (Hoax), maka:
            P(Hoax) = 0.6,  P(Fakta) = 0.4

        Penggunaan Log:
          log P(c) = log(Nₖ / N) = log(Nₖ) - log(N)
          Digunakan untuk menghindari numerik underflow saat perkalian.

        Kompleksitas: O(N) — iterasi seluruh label sekali

        Args:
            y : List[int] — label training untuk setiap dokumen

        Returns:
            Dict[int, float] — {kelas: log_prior_probability}
        """
        N = len(y)  # Total jumlah dokumen

        # Hitung frekuensi setiap kelas: Nₖ = |{i : yᵢ = c}|
        class_count = {}
        for label in y:
            class_count[label] = class_count.get(label, 0) + 1

        # Simpan untuk referensi
        self.class_count = class_count
        self.classes = sorted(class_count.keys())

        # Hitung log prior: log P(c) = log(Nₖ) - log(N)
        class_log_prior = {}
        for c in self.classes:
            # P(c) = Nₖ / N
            # log P(c) = log(Nₖ / N)
            class_log_prior[c] = math.log(class_count[c] / N)

        return class_log_prior

    # ─── TAHAP 2: HITUNG LIKELIHOOD DENGAN LAPLACE SMOOTHING ─────────────

    def _compute_likelihood(self, X, y):
        """
        Menghitung Likelihood P(tᵢ|c) untuk setiap fitur tᵢ dan kelas c.

        Rumus Multinomial dengan Laplace Smoothing:
          P(tᵢ|c) = (count(tᵢ, c) + α) / (Σⱼ count(tⱼ, c) + α·|V|)

        dimana:
          count(tᵢ, c) = Σ_{d ∈ kelas c} X[d][i]
                        = total bobot TF-IDF term tᵢ di semua dokumen kelas c
          Σⱼ count(tⱼ, c) = total bobot SEMUA term di kelas c
          α              = parameter smoothing (default: 1)
          |V|            = jumlah fitur/term unik dalam vocabulary

        Langkah Perhitungan:
          1. Untuk setiap kelas c:
             a. Jumlahkan bobot setiap fitur dari semua dokumen kelas c
                → vektor feature_count[c] berukuran |V|
             b. Hitung total bobot semua fitur di kelas c
                → total_count[c] = Σⱼ feature_count[c][j]
             c. Untuk setiap fitur i:
                log P(tᵢ|c) = log((feature_count[c][i] + α) / (total_count[c] + α·|V|))

        Mengapa Laplace Smoothing Penting:
          Tanpa smoothing (α=0):
            Jika kata "xyz" tidak pernah muncul di kelas c:
              P("xyz"|c) = 0/total = 0
              P(d|c) = P(t₁|c) × ... × P("xyz"|c) × ... = 0
            Seluruh probabilitas menjadi 0! Satu kata yang tidak ditemukan
            membatalkan semua bukti dari kata-kata lain.

          Dengan Laplace smoothing (α=1):
            P("xyz"|c) = (0 + 1) / (total + |V|) > 0
            Probabilitas kecil tapi BUKAN nol → kata lain masih berkontribusi.

        Kompleksitas: O(N × |V|) — iterasi seluruh matriks training

        Args:
            X : List[List[float]] — matriks TF-IDF (N × |V|)
            y : List[int] — label per dokumen

        Returns:
            Dict[int, List[float]] — {kelas: [log_prob_fitur_0, ..., log_prob_fitur_V-1]}
        """
        V = self.n_features  # |V| = ukuran vocabulary
        alpha = self.alpha

        # ── Langkah 1: Akumulasi bobot fitur per kelas ───────────────────
        # feature_count[c][i] = Σ_{d ∈ kelas c} X[d][i]
        # Artinya: total bobot TF-IDF term ke-i dari semua dokumen kelas c
        feature_count = {}
        for c in self.classes:
            feature_count[c] = [0.0] * V  # Inisialisasi vektor nol

        # Iterasi setiap dokumen dan akumulasi bobotnya ke kelas yang sesuai
        for doc_idx in range(len(y)):
            c = y[doc_idx]  # Kelas dokumen ini
            for feat_idx in range(V):
                # Akumulasi bobot TF-IDF term feat_idx ke kelas c
                feature_count[c][feat_idx] += X[doc_idx][feat_idx]

        # ── Langkah 2: Hitung log P(tᵢ|c) dengan Laplace Smoothing ──────
        feature_log_prob = {}
        for c in self.classes:
            # Total bobot semua fitur di kelas c:
            # total_count(c) = Σⱼ feature_count[c][j]
            total_count = sum(feature_count[c])

            # Denominator (penyebut) dengan smoothing:
            # Σⱼ count(tⱼ, c) + α·|V|
            denominator = total_count + alpha * V

            # Hitung log probability untuk setiap fitur
            log_probs = [0.0] * V
            for i in range(V):
                # Numerator (pembilang) dengan smoothing:
                # count(tᵢ, c) + α
                numerator = feature_count[c][i] + alpha

                # log P(tᵢ|c) = log(numerator / denominator)
                log_probs[i] = math.log(numerator / denominator)

            feature_log_prob[c] = log_probs

        return feature_log_prob

    # ─── TRAINING ────────────────────────────────────────────────────────

    def train(self, X, y, feature_names=None):
        """
        Melatih model Multinomial Naive Bayes.

        Proses Training:
          1. Hitung Prior: P(c) untuk setiap kelas c
          2. Hitung Likelihood: P(tᵢ|c) untuk setiap fitur tᵢ dan kelas c
             dengan menerapkan Laplace Smoothing

        Setelah training, model menyimpan:
          - class_log_prior   : log P(c) per kelas
          - feature_log_prob  : log P(tᵢ|c) per fitur per kelas

        Args:
            X             : List[List[float]] — matriks TF-IDF (N × V)
            y             : List[int] — label per dokumen
            feature_names : List[str] — nama fitur (opsional, untuk interpretasi)

        Returns:
            Dict — informasi hasil training
        """
        self.n_features = len(X[0]) if len(X) > 0 else 0
        self.feature_names = feature_names or []

        # ── Tahap 1: Hitung Prior P(c) ───────────────────────────────────
        self.class_log_prior = self._compute_prior(y)

        # ── Tahap 2: Hitung Likelihood P(tᵢ|c) ──────────────────────────
        self.feature_log_prob = self._compute_likelihood(X, y)

        return {
            "status": "Success",
            "n_classes": len(self.classes),
            "classes": self.classes,
            "class_distribution": {
                str(c): self.class_count[c] for c in self.classes
            },
            "n_features": self.n_features,
            "alpha": self.alpha,
        }

    # ─── PREDIKSI ────────────────────────────────────────────────────────

    def _predict_log_proba(self, x):
        """
        Menghitung log-posterior untuk SATU dokumen terhadap semua kelas.

        Rumus (dalam log-space):
          log P(c|d) ∝ log P(c) + Σᵢ wᵢ · log P(tᵢ|c)

        dimana:
          log P(c)    = log prior (dari training)
          log P(tᵢ|c) = log likelihood fitur ke-i untuk kelas c
          wᵢ          = bobot TF-IDF fitur ke-i dalam dokumen d

        Catatan Penting:
          Pada Multinomial NB standar dengan count data, rumusnya adalah:
            log P(c|d) ∝ log P(c) + Σᵢ count(tᵢ) · log P(tᵢ|c)

          Karena kita menggunakan bobot TF-IDF (bukan raw count), kita
          menggunakan bobot TF-IDF sebagai "count" efektif. Ini ekivalen
          dengan mengalikan log-likelihood dengan bobot TF-IDF.

          Fitur dengan bobot TF-IDF = 0 tidak berkontribusi pada skor
          (karena 0 × log P(tᵢ|c) = 0), sehingga kata yang tidak muncul
          di dokumen secara otomatis diabaikan.

        Args:
            x : List[float] — vektor TF-IDF satu dokumen (ukuran |V|)

        Returns:
            Dict[int, float] — {kelas: log_posterior_score}
        """
        log_scores = {}

        for c in self.classes:
            # Mulai dari log prior: log P(c)
            score = self.class_log_prior[c]

            # Tambahkan weighted log-likelihood setiap fitur:
            # Σᵢ wᵢ · log P(tᵢ|c)
            for i in range(self.n_features):
                if x[i] > 0:  # Hanya fitur yang aktif (bobot > 0)
                    score += x[i] * self.feature_log_prob[c][i]

            log_scores[c] = score

        return log_scores

    def predict(self, X_test):
        """
        Memprediksi kelas untuk satu atau banyak dokumen.

        Aturan Keputusan (Maximum A Posteriori / MAP):
          ĉ = argmax_c  log P(c|d)
            = argmax_c  [ log P(c) + Σᵢ wᵢ · log P(tᵢ|c) ]

        Kelas dengan log-posterior TERTINGGI dipilih sebagai prediksi.

        Mengapa Log-Space:
          - Perkalian probabilitas kecil (e.g., 0.001 × 0.002 × ...) → underflow
          - Log mengubah perkalian menjadi penjumlahan: log(a×b) = log(a) + log(b)
          - Argmax tidak berubah karena log adalah fungsi monoton naik

        Args:
            X_test : List[List[float]] — matriks TF-IDF data test (M × V)

        Returns:
            List[int] — prediksi kelas untuk setiap dokumen test
        """
        predictions = []
        for x in X_test:
            log_scores = self._predict_log_proba(x)

            # argmax: pilih kelas dengan log-posterior tertinggi
            best_class = max(log_scores, key=log_scores.get)
            predictions.append(best_class)

        return predictions

    def predict_proba(self, X_test):
        """
        Menghitung probabilitas posterior (approx) untuk setiap dokumen.

        Konversi log-score ke probabilitas menggunakan softmax:
          P(c|d) = exp(log_score(c)) / Σₖ exp(log_score(k))

        Ini adalah normalisasi agar Σ P(c|d) = 1 untuk setiap dokumen.

        Catatan:
          Untuk stabilitas numerik, kita kurangi nilai maksimum sebelum exp():
            P(c|d) = exp(s(c) - max_s) / Σₖ exp(s(k) - max_s)
          Ini disebut "log-sum-exp trick" dan tidak mengubah hasil.

        Args:
            X_test : List[List[float]] — matriks TF-IDF data test

        Returns:
            List[Dict[int, float]] — probabilitas per kelas per dokumen
        """
        all_proba = []

        for x in X_test:
            log_scores = self._predict_log_proba(x)

            # Log-sum-exp trick untuk stabilitas numerik
            max_score = max(log_scores.values())
            exp_scores = {
                c: math.exp(log_scores[c] - max_score)
                for c in self.classes
            }
            total_exp = sum(exp_scores.values())

            # Normalisasi: P(c|d) = exp(score) / total
            proba = {
                c: exp_scores[c] / total_exp
                for c in self.classes
            }
            all_proba.append(proba)

        return all_proba

    # ─── INTERPRETASI MODEL ──────────────────────────────────────────────

    def get_top_features_per_class(self, top_n=10):
        """
        Ambil fitur (kata) dengan probabilitas tertinggi per kelas.

        Berguna untuk interpretasi: kata apa yang paling "khas" untuk
        masing-masing kelas?

        Kata dengan log P(tᵢ|c) tinggi = kata yang sering muncul di kelas c
        dan jarang di kelas lain → fitur diskriminatif.

        Args:
            top_n : Jumlah fitur teratas per kelas

        Returns:
            Dict[int, List[Dict]] — {kelas: [{term, log_prob, rank}, ...]}
        """
        result = {}

        for c in self.classes:
            # Pasangkan setiap fitur dengan log-probabilitasnya
            features = []
            for i in range(self.n_features):
                name = self.feature_names[i] if i < len(self.feature_names) else f"f_{i}"
                features.append({
                    'term': name,
                    'log_prob': round(self.feature_log_prob[c][i], 6),
                    'prob': round(math.exp(self.feature_log_prob[c][i]), 8),
                })

            # Urutkan berdasarkan probabilitas (descending)
            features.sort(key=lambda x: x['log_prob'], reverse=True)

            # Beri rank
            for rank, item in enumerate(features[:top_n], start=1):
                item['rank'] = rank

            result[c] = features[:top_n]

        return result
