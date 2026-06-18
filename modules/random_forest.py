import math
import random

# BAGIAN 1: DECISION TREE NODE — STRUKTUR POHON
class TreeNode:
    def __init__(self):
        self.feature_index = None   # Indeks fitur yang digunakan untuk split
        self.threshold = None       # Nilai ambang batas split
        self.left = None            # Subtree kiri (fitur ≤ threshold)
        self.right = None           # Subtree kanan (fitur > threshold)
        self.label = None           # Label prediksi (hanya untuk leaf node)
        self.gini = None            # Gini impurity node ini
        self.n_samples = 0          # Jumlah sampel di node ini

# BAGIAN 2: DECISION TREE CLASSIFIER
class DecisionTreeScratch:
    def __init__(self, max_depth=5, min_samples=5, max_features=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.root = None        # Root node pohon
        self.n_features = 0     # Jumlah fitur total

    # GINI IMPURITY
    @staticmethod
    def _gini_impurity(y_subset):
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

    # PENCARIAN SPLIT TERBAIK
    def _best_split(self, X, y, feature_indices):
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

    # MENENTUKAN KELAS MAYORITAS
    @staticmethod
    def _majority_class(y):
        class_count = {}
        for label in y:
            class_count[label] = class_count.get(label, 0) + 1

        return max(class_count, key=class_count.get)

    # MEMBANGUN POHON (REKURSIF)
    def _build_tree(self, X, y, depth=0):
        node = TreeNode()
        node.n_samples = len(y)
        node.gini = self._gini_impurity(y)

        # Stopping Criteria
        # Cek apakah harus berhenti dan membuat leaf
        unique_classes = set(y)

        if (depth >= self.max_depth or            # (a) Kedalaman maks
            len(y) < self.min_samples or          # (b) Sampel terlalu sedikit
            len(unique_classes) == 1):            # (c) Node sudah murni
            node.label = self._majority_class(y)
            return node

        # Random Feature Selection
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

        # Cari Split Terbaik
        best_feat, best_thresh, best_gini = self._best_split(X, y, feature_indices)

        if best_feat is None:  # (d) Tidak ada split valid
            node.label = self._majority_class(y)
            return node

        # Bagi Data dan Rekursi
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

    # TRAINING
    def fit(self, X, y):
        self.n_features = len(X[0]) if len(X) > 0 else 0
        self.root = self._build_tree(X, y, depth=0)

    # PREDIKSI SATU SAMPEL

    def _predict_one(self, x, node):
        # Leaf node → kembalikan label
        if node.label is not None:
            return node.label

        # Internal node → cek fitur dan threshold
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X_test):
        return [self._predict_one(x, self.root) for x in X_test]


# BAGIAN 3: RANDOM FOREST CLASSIFIER
class RandomForestScratch:
    def __init__(self, n_trees=100, max_depth=5, min_samples=5, max_features='sqrt', seed=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.seed = seed
        self.trees = []             # List of DecisionTreeScratch
        self.oob_indices = []       # Indeks data Out-of-Bag per pohon
        self.feature_names = []

    # BOOTSTRAP SAMPLING
    @staticmethod
    def _bootstrap_sample(X, y):
        N = len(y)
        # Ambil N indeks acak dengan pengembalian
        indices = [random.randint(0, N - 1) for _ in range(N)]

        X_sample = [X[i] for i in indices]
        y_sample = [y[i] for i in indices]

        # Out-of-Bag: indeks yang tidak terpilih
        selected = set(indices)
        oob = set(range(N)) - selected

        return X_sample, y_sample, oob

    # BUILD FOREST (TRAINING)
    def build_forest(self, X, y, feature_names=None):
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

    # MAJORITY VOTING (PREDIKSI)
    def predict(self, X_test):
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
        details = []
        for t, tree in enumerate(self.trees):
            vote = tree._predict_one(x, tree.root)
            details.append({
                'tree_id': t + 1,
                'prediction': vote,
            })
        return details
