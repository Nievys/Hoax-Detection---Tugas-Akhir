import math

class MultinomialNBScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

        # Variabel model
        self.classes = []               # Kelas unik: e.g. [0, 1] atau [-1, 1]
        self.class_log_prior = {}       # log P(c) untuk setiap kelas c
        self.feature_log_prob = {}      # log P(tᵢ|c) untuk setiap fitur i, kelas c
        self.class_count = {}           # Jumlah dokumen per kelas (untuk info)
        self.n_features = 0             # |V| = ukuran vocabulary
        self.feature_names = []         # Nama fitur (opsional, untuk interpretasi)

    # TAHAP 1 HITUNG PRIOR PROBABILITY
    def _compute_prior(self, y):
        N = len(y)  # Total jumlah dokumen

        # Hitung frekuensi setiap kelas
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
            print("log prior", class_log_prior[c])

        return class_log_prior

    # TAHAP 2: HITUNG LIKELIHOOD DENGAN LAPLACE SMOOTHING
    def _compute_likelihood(self, X, y):
        V = self.n_features  # |V| = ukuran vocabulary
        alpha = self.alpha

        # Langkah 1 Akumulasi bobot fitur per kelas
        # feature_count[c][i] = Σ_{d ∈ kelas c} X[d][i]
        # Artinya total bobot TF-IDF term ke-i dari semua dokumen kelas c
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

    # TRAINING
    def train(self, X, y, feature_names=None):
        self.n_features = len(X[0]) if len(X) > 0 else 0
        self.feature_names = feature_names or []

        # Tahap 1 Hitung Prior P(c)
        self.class_log_prior = self._compute_prior(y)

        # Tahap 2 Hitung Likelihood P(tᵢ|c)
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

    # PREDIKSI
    def _predict_log_proba(self, x):
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
        predictions = []
        for x in X_test:
            log_scores = self._predict_log_proba(x)

            # argmax: pilih kelas dengan log-posterior tertinggi
            best_class = max(log_scores, key=log_scores.get)
            predictions.append(best_class)

        return predictions

    def predict_proba(self, X_test):
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

    # INTERPRETASI MODEL
    def get_top_features_per_class(self, top_n=10):
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
