"""
=============================================================================
MODUL 5: CLASSIFICATION — RANDOM FOREST
=============================================================================
Judul TA : Analisis Pengaruh Teknik Normalisasi Kata Gaul (Slang) terhadap
           Akurasi Deteksi Ujaran Kebencian Berbahasa Indonesia
=============================================================================
Implementasi MURNI Python. Tidak menggunakan Scikit-Learn atau library
klasifikasi manapun. Setiap rumus dijelaskan secara matematis.
=============================================================================

LANDASAN MATEMATIS — RANDOM FOREST
====================================

Random Forest adalah metode ensemble learning yang mengkombinasikan
banyak Decision Tree untuk meningkatkan akurasi dan mengurangi overfitting.

Komponen Utama:
  1. DECISION TREE  — unit klasifikasi dasar (weak learner)
  2. BAGGING         — Bootstrap Aggregating untuk diversitas
  3. RANDOM FEATURE  — subset fitur acak per split
  4. MAJORITY VOTING — agregasi prediksi dari semua pohon

─────────────────────────────────────────────────────────────────────

DECISION TREE — GINI IMPURITY
══════════════════════════════

Gini Impurity mengukur "ketidakmurnian" sebuah node:
  Gini(S) = 1 - Σₖ pₖ²

  dimana:
    pₖ = proporsi kelas k dalam himpunan S
    K  = jumlah kelas

  Interpretasi:
    Gini = 0   → node murni (semua data satu kelas)
    Gini = 0.5 → node paling tidak murni (binary, 50:50)

Pemilihan Split Terbaik:
  Untuk setiap fitur f dan threshold t, hitung weighted Gini:
    Gini_split = (|S_left|/|S|) · Gini(S_left) + (|S_right|/|S|) · Gini(S_right)

  Pilih split yang MEMINIMALKAN Gini_split.

  Information Gain (Gini):
    ΔGini = Gini(parent) - Gini_split
  Split terbaik = yang memaksimalkan ΔGini.

─────────────────────────────────────────────────────────────────────

BAGGING (Bootstrap Aggregating)
═══════════════════════════════

Untuk setiap pohon ke-t (t = 1, ..., T):
  1. Ambil sampel bootstrap Bₜ dari dataset D:
     Bₜ = {(xᵢ, yᵢ) : i ~ Uniform(1, N) dengan pengembalian}
     |Bₜ| = N (ukuran sama dengan dataset asli)

  Catatan: ~63.2% data unik akan terpilih per bootstrap
  (karena P(tidak terpilih) = (1 - 1/N)^N ≈ 1/e ≈ 0.368)

Random Feature Selection (per split):
  Pada setiap node split, hanya subset acak dari fitur yang dipertimbangkan:
    m = √|V|  (untuk klasifikasi, rekomendasi umum)

  Ini menambah diversitas antar pohon dan mengurangi korelasi.

─────────────────────────────────────────────────────────────────────

MAJORITY VOTING (Agregasi)
══════════════════════════

Prediksi ensemble:
  ŷ(x) = mode({ h₁(x), h₂(x), ..., hₜ(x) })

  dimana hₜ(x) = prediksi pohon ke-t untuk input x.
  Mode = nilai yang paling sering muncul (suara terbanyak).

Probabilitas (opsional):
  P(c|x) = (1/T) · Σₜ 𝟙[hₜ(x) = c]
  = proporsi pohon yang memprediksi kelas c.
=============================================================================
"""

import math
import random


# =============================================================================
# BAGIAN 1: DECISION TREE NODE — STRUKTUR POHON
# =============================================================================

class TreeNode:
    """
    Representasi satu node dalam Decision Tree.

    Ada dua tipe node:
      1. Internal Node (node keputusan):
         - Memiliki feature_index dan threshold untuk split
         - Memiliki anak kiri (left) dan kanan (right)

      2. Leaf Node (node daun):
         - Memiliki label prediksi (kelas mayoritas)
         - Tidak memiliki anak

    Struktur Pohon:
      Setiap path dari root ke leaf merepresentasikan sebuah aturan:
        IF fitur[i₁] ≤ t₁ AND fitur[i₂] ≤ t₂ ... THEN kelas = c
    """

    def __init__(self):
        self.feature_index = None   # Indeks fitur yang digunakan untuk split
        self.threshold = None       # Nilai ambang batas split
        self.left = None            # Subtree kiri (fitur ≤ threshold)
        self.right = None           # Subtree kanan (fitur > threshold)
        self.label = None           # Label prediksi (hanya untuk leaf node)
        self.gini = None            # Gini impurity node ini
        self.n_samples = 0          # Jumlah sampel di node ini


# =============================================================================
# BAGIAN 2: DECISION TREE CLASSIFIER
# =============================================================================

class DecisionTreeScratch:
    """
    Implementasi Decision Tree dari nol menggunakan Gini Impurity.

    Parameter:
      max_depth    : Kedalaman maksimum pohon (mencegah overfitting)
      min_samples  : Jumlah minimum sampel untuk melakukan split
      max_features : Jumlah fitur yang dipertimbangkan per split
                     (None = semua fitur, 'sqrt' = √|V|)
    """

    def __init__(self, max_depth=5, min_samples=5, max_features=None):
        """
        Args:
            max_depth    : Kedalaman maksimum pohon.
                           Semakin dalam → semakin kompleks → risiko overfit.
            min_samples  : Minimum sampel di node untuk bisa di-split.
                           Jika < min_samples → jadikan leaf.
            max_features : Jumlah fitur acak per split.
                           None = semua, 'sqrt' = √|V|, int = angka tetap.
        """
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.root = None        # Root node pohon (diisi saat training)
        self.n_features = 0     # Jumlah fitur total

    # ─── GINI IMPURITY ───────────────────────────────────────────────────

    @staticmethod
    def _gini_impurity(y_subset):
        """
        Menghitung Gini Impurity untuk satu himpunan label.

        Rumus:
          Gini(S) = 1 - Σₖ pₖ²

        dimana:
          pₖ = |{i : yᵢ = k}| / |S| = proporsi kelas k
          K  = jumlah kelas unik dalam S

        Contoh:
          S = [1, 1, 0, 0]  → p₁ = 0.5, p₀ = 0.5
          Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.5  (paling tidak murni)

          S = [1, 1, 1, 1]  → p₁ = 1.0
          Gini = 1 - 1.0² = 0.0  (murni sempurna)

        Kompleksitas: O(|S|) — iterasi sekali untuk counting

        Args:
            y_subset : List[int] — label-label dalam himpunan ini

        Returns:
            float — nilai Gini Impurity [0, 0.5] untuk binary
        """
        n = len(y_subset)
        if n == 0:
            return 0.0

        # Hitung frekuensi setiap kelas
        class_count = {}
        for label in y_subset:
            class_count[label] = class_count.get(label, 0) + 1

        # Gini = 1 - Σₖ (nₖ/n)²
        gini = 1.0
        for count in class_count.values():
            p = count / n        # pₖ = proporsi kelas k
            gini -= p * p        # kurangi pₖ²

        return gini

    # ─── PENCARIAN SPLIT TERBAIK ─────────────────────────────────────────

    def _best_split(self, X, y, feature_indices):
        """
        Mencari split terbaik dari subset fitur yang diberikan.

        Algoritma (Dioptimasi dengan Running Counts):
          Untuk menghindari kompleksitas O(N²) saat mengecek threshold,
          kita mengurutkan sampel berdasarkan nilai fitur O(N log N).
          Lalu kita iterasi dari kiri ke kanan, memindahkan sampel dari
          'right' ke 'left' secara inkremental dan menghitung Gini O(1).
          
        Kompleksitas: O(m × N log N) — sangat efisien untuk dataset besar.

        Args:
            X               : Data fitur (subset bootstrap)
            y               : Label (subset bootstrap)
            feature_indices : List indeks fitur yang dipertimbangkan

        Returns:
            Tuple (best_feature, best_threshold, best_gini)
            atau (None, None, None) jika tidak ada split yang valid
        """
        n = len(y)
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        # Hitung total kemunculan setiap kelas di parent node
        total_counts = {}
        for label in y:
            total_counts[label] = total_counts.get(label, 0) + 1

        for feat_idx in feature_indices:
            # Urutkan pasangan (nilai_fitur, label) berdasarkan nilai_fitur
            sorted_pairs = sorted([(X[i][feat_idx], y[i]) for i in range(n)], key=lambda x: x[0])

            left_counts = {}
            right_counts = dict(total_counts)  # Copy of total counts

            for j in range(n - 1):
                val, label = sorted_pairs[j]

                # Pindahkan sampel dari himpunan 'right' ke 'left'
                left_counts[label] = left_counts.get(label, 0) + 1
                right_counts[label] -= 1
                
                # Kita hanya perlu mengevaluasi split jika nilai fitur saat ini
                # berbeda dengan nilai fitur berikutnya (kandidat threshold).
                next_val = sorted_pairs[j + 1][0]
                if val == next_val:
                    continue  # Tunggu sampai nilai fitur berubah

                # Kandidat threshold adalah midpoint
                threshold = (val + next_val) / 2.0

                n_left = j + 1
                n_right = n - n_left

                # Hitung Gini secara efisien O(K) dimana K adalah jumlah kelas
                gini_left = 1.0 - sum((c / n_left) ** 2 for c in left_counts.values())
                gini_right = 1.0 - sum((c / n_right) ** 2 for c in right_counts.values())

                weighted_gini = (n_left / n) * gini_left + (n_right / n) * gini_right

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feat_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gini

    # ─── MENENTUKAN KELAS MAYORITAS ──────────────────────────────────────

    @staticmethod
    def _majority_class(y):
        """
        Menentukan kelas mayoritas (mode) dari himpunan label.

        Rumus:
          ŷ = argmax_c |{i : yᵢ = c}|
          = kelas yang paling sering muncul

        Digunakan untuk menentukan label pada leaf node.

        Args:
            y : List[int] — himpunan label

        Returns:
            int — kelas mayoritas
        """
        class_count = {}
        for label in y:
            class_count[label] = class_count.get(label, 0) + 1

        return max(class_count, key=class_count.get)

    # ─── MEMBANGUN POHON (REKURSIF) ──────────────────────────────────────

    def _build_tree(self, X, y, depth=0):
        """
        Membangun Decision Tree secara rekursif (top-down, greedy).

        Algoritma CART (Classification and Regression Trees):
          1. Jika stopping criteria terpenuhi → buat leaf node
          2. Pilih subset fitur acak (untuk Random Forest)
          3. Cari split terbaik berdasarkan Gini Impurity
          4. Bagi data menjadi left dan right
          5. Rekursi untuk kedua sub-himpunan

        Stopping Criteria (kapan berhenti memecah):
          a. Kedalaman mencapai max_depth
          b. Jumlah sampel < min_samples
          c. Node sudah murni (semua label sama → Gini = 0)
          d. Tidak ada split yang valid ditemukan

        Args:
            X     : Data fitur
            y     : Label
            depth : Kedalaman saat ini (untuk pengecekan max_depth)

        Returns:
            TreeNode — root subtree
        """
        node = TreeNode()
        node.n_samples = len(y)
        node.gini = self._gini_impurity(y)

        # ── Stopping Criteria ────────────────────────────────────────────
        # Cek apakah harus berhenti dan membuat leaf
        unique_classes = set(y)

        if (depth >= self.max_depth or            # (a) Kedalaman maks
            len(y) < self.min_samples or          # (b) Sampel terlalu sedikit
            len(unique_classes) == 1):            # (c) Node sudah murni
            node.label = self._majority_class(y)
            return node

        # ── Random Feature Selection ─────────────────────────────────────
        # Pilih subset fitur acak untuk dipertimbangkan pada split ini.
        # Ini adalah kunci perbedaan Random Forest vs Bagged Trees:
        #   m = √|V| (rekomendasi untuk klasifikasi)
        # Tujuan: mengurangi korelasi antar pohon → meningkatkan diversitas
        all_features = list(range(self.n_features))

        if self.max_features == 'sqrt':
            m = max(1, int(math.sqrt(self.n_features)))
        elif isinstance(self.max_features, int):
            m = min(self.max_features, self.n_features)
        else:
            m = self.n_features  # Gunakan semua fitur

        # Pilih m fitur secara acak (tanpa pengembalian)
        if m < self.n_features:
            feature_indices = random.sample(all_features, m)
        else:
            feature_indices = all_features

        # ── Cari Split Terbaik ───────────────────────────────────────────
        best_feat, best_thresh, best_gini = self._best_split(X, y, feature_indices)

        if best_feat is None:  # (d) Tidak ada split valid
            node.label = self._majority_class(y)
            return node

        # ── Bagi Data dan Rekursi ────────────────────────────────────────
        node.feature_index = best_feat
        node.threshold = best_thresh

        # S_left = {xᵢ : xᵢ[f] ≤ t},  S_right = {xᵢ : xᵢ[f] > t}
        X_left, y_left = [], []
        X_right, y_right = [], []

        for i in range(len(y)):
            if X[i][best_feat] <= best_thresh:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])

        # Rekursi: bangun subtree kiri dan kanan
        node.left = self._build_tree(X_left, y_left, depth + 1)
        node.right = self._build_tree(X_right, y_right, depth + 1)

        return node

    # ─── TRAINING ────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Melatih Decision Tree dari data training.

        Args:
            X : List[List[float]] — matriks fitur (N × V)
            y : List[int] — label per sampel
        """
        self.n_features = len(X[0]) if len(X) > 0 else 0
        self.root = self._build_tree(X, y, depth=0)

    # ─── PREDIKSI SATU SAMPEL ────────────────────────────────────────────

    def _predict_one(self, x, node):
        """
        Traversal pohon untuk satu sampel (rekursif).

        Aturan:
          Pada setiap internal node:
            Jika x[feature_index] ≤ threshold → ke anak kiri
            Jika x[feature_index] > threshold → ke anak kanan
          Pada leaf node:
            Return label

        Args:
            x    : Vektor fitur satu sampel (ukuran V)
            node : Node saat ini dalam traversal

        Returns:
            int — prediksi kelas
        """
        # Leaf node → kembalikan label
        if node.label is not None:
            return node.label

        # Internal node → cek fitur dan threshold
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X_test):
        """
        Prediksi kelas untuk banyak sampel.

        Args:
            X_test : List[List[float]] — matriks data test

        Returns:
            List[int] — prediksi kelas per sampel
        """
        return [self._predict_one(x, self.root) for x in X_test]


# =============================================================================
# BAGIAN 3: RANDOM FOREST CLASSIFIER
# =============================================================================

class RandomForestScratch:
    """
    Implementasi Random Forest dari nol.

    Random Forest = Bagging + Random Feature Selection + Decision Trees

    Komponen:
      1. Bootstrap Sampling : ambil N sampel acak DENGAN pengembalian
      2. Decision Tree      : latih pohon pada bootstrap sample
      3. Random Features    : setiap split hanya pertimbangkan √|V| fitur
      4. Majority Voting    : agregasi prediksi semua pohon

    Keunggulan:
      - Mengurangi variance (overfitting) dari single decision tree
      - Robust terhadap noise dan outlier
      - Tidak memerlukan feature scaling (invariant terhadap monotonic transforms)
    """

    def __init__(self, n_trees=100, max_depth=5, min_samples=5, max_features='sqrt', seed=42):
        """
        Inisialisasi parameter Random Forest.

        Args:
            n_trees      : Jumlah pohon dalam forest (T).
                           Semakin banyak → semakin stabil, tapi lebih lambat.
            max_depth    : Kedalaman maksimum setiap pohon.
            min_samples  : Minimum sampel per node untuk split.
            max_features : Jumlah fitur acak per split.
                           'sqrt' = √|V| (rekomendasi untuk klasifikasi)
            seed         : Seed untuk reprodusibilitas (default: 42).
                           Seed di-set ulang setiap kali build_forest() dipanggil
                           agar hasil selalu konsisten.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.seed = seed
        self.trees = []             # List of DecisionTreeScratch
        self.oob_indices = []       # Indeks data Out-of-Bag per pohon
        self.feature_names = []

    # ─── BOOTSTRAP SAMPLING ──────────────────────────────────────────────

    @staticmethod
    def _bootstrap_sample(X, y):
        """
        Mengambil sampel bootstrap dari dataset.

        Metode:
          Dari dataset D berukuran N, ambil N sampel secara acak
          DENGAN pengembalian (with replacement).

        Statistik Bootstrap:
          - Setiap sampel memiliki probabilitas 1/N untuk terpilih per draw
          - Probabilitas TIDAK terpilih dalam N draws:
            P(tidak terpilih) = (1 - 1/N)^N ≈ 1/e ≈ 0.368  (untuk N besar)
          - Artinya ~63.2% data unik akan ada dalam bootstrap sample
          - Sisanya ~36.8% = Out-of-Bag (OOB) → bisa digunakan untuk validasi

        Args:
            X : List[List[float]] — matriks fitur
            y : List[int] — label

        Returns:
            Tuple (X_sample, y_sample, oob_indices)
              X_sample    : Data bootstrap
              y_sample    : Label bootstrap
              oob_indices : Set indeks data yang TIDAK terpilih (OOB)
        """
        N = len(y)
        # Ambil N indeks acak dengan pengembalian
        indices = [random.randint(0, N - 1) for _ in range(N)]

        X_sample = [X[i] for i in indices]
        y_sample = [y[i] for i in indices]

        # Out-of-Bag: indeks yang tidak terpilih
        selected = set(indices)
        oob = set(range(N)) - selected

        return X_sample, y_sample, oob

    # ─── BUILD FOREST (TRAINING) ─────────────────────────────────────────

    def build_forest(self, X, y, feature_names=None):
        """
        Membangun Random Forest — melatih T pohon pada bootstrap samples.

        Algoritma:
          FOR t = 1 TO T (jumlah pohon):
            1. Buat bootstrap sample Bₜ dari (X, y)
            2. Inisialisasi DecisionTree dengan max_features='sqrt'
            3. Latih pohon pada Bₜ
            4. Simpan pohon dan OOB indices

        Args:
            X             : List[List[float]] — matriks TF-IDF (N × V)
            y             : List[int] — label per dokumen
            feature_names : List[str] — nama fitur (opsional)

        Returns:
            Dict — informasi hasil training
        """
        # ── Reset seed PRNG agar hasil SELALU sama setiap kali dipanggil ──
        # Tanpa ini, global random state terus maju setiap kali
        # build_forest() dipanggil → hasil berbeda setiap run.
        random.seed(self.seed)

        self.trees = []
        self.oob_indices = []
        self.feature_names = feature_names or []

        for t in range(self.n_trees):
            # Langkah 1: Bootstrap sampling
            X_boot, y_boot, oob = self._bootstrap_sample(X, y)

            # Langkah 2-3: Buat dan latih Decision Tree
            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples=self.min_samples,
                max_features=self.max_features
            )
            tree.fit(X_boot, y_boot)

            # Langkah 4: Simpan
            self.trees.append(tree)
            self.oob_indices.append(oob)

        return {
            "status": "Success",
            "n_trees": self.n_trees,
            "max_depth": self.max_depth,
            "max_features": self.max_features,
            "min_samples": self.min_samples,
        }

    # ─── MAJORITY VOTING (PREDIKSI) ──────────────────────────────────────

    def predict(self, X_test):
        """
        Prediksi menggunakan majority voting dari semua pohon.

        Rumus:
          ŷ(x) = mode({ h₁(x), h₂(x), ..., hₜ(x) })

        Setiap pohon memberikan satu "suara" (vote) untuk kelasnya.
        Kelas dengan suara terbanyak menjadi prediksi final.

        Args:
            X_test : List[List[float]] — data test (M × V)

        Returns:
            List[int] — prediksi kelas per sampel
        """
        predictions = []

        for x in X_test:
            # Kumpulkan vote dari setiap pohon
            votes = []
            for tree in self.trees:
                vote = tree._predict_one(x, tree.root)
                votes.append(vote)

            # Majority voting: hitung frekuensi setiap kelas
            vote_count = {}
            for v in votes:
                vote_count[v] = vote_count.get(v, 0) + 1

            # Pilih kelas dengan suara terbanyak
            winner = max(vote_count, key=vote_count.get)
            predictions.append(winner)

        return predictions

    def predict_proba(self, X_test):
        """
        Hitung probabilitas prediksi berdasarkan proporsi vote.

        Rumus:
          P(c|x) = (1/T) · Σₜ 𝟙[hₜ(x) = c]
          = jumlah pohon yang memprediksi kelas c / total pohon

        Args:
            X_test : List[List[float]] — data test

        Returns:
            List[Dict[int, float]] — probabilitas per kelas per sampel
        """
        all_proba = []
        T = len(self.trees)

        for x in X_test:
            votes = [tree._predict_one(x, tree.root) for tree in self.trees]

            # Hitung proporsi vote per kelas
            vote_count = {}
            for v in votes:
                vote_count[v] = vote_count.get(v, 0) + 1

            proba = {c: count / T for c, count in vote_count.items()}
            all_proba.append(proba)

        return all_proba

    def get_vote_detail(self, x):
        """
        Ambil detail vote dari setiap pohon untuk satu sampel.

        Berguna untuk visualisasi: melihat bagaimana setiap pohon
        dalam forest "memilih" kelasnya.

        Args:
            x : List[float] — vektor fitur satu sampel

        Returns:
            List[Dict] — detail vote per pohon
        """
        details = []
        for t, tree in enumerate(self.trees):
            vote = tree._predict_one(x, tree.root)
            details.append({
                'tree_id': t + 1,
                'prediction': vote,
            })
        return details
