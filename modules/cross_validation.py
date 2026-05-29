"""
=============================================================================
MODUL 7: K-FOLD CROSS VALIDATION ENGINE (FROM SCRATCH)
=============================================================================
Judul TA : Penerapan Metode Majority Voting Ensemble dengan Algoritma SVM,
           Random Forest, dan Naive Bayes untuk Deteksi Hoax Isu Energi
           pada Media Sosial X
=============================================================================
Implementasi MURNI Python. Tidak menggunakan Scikit-Learn atau library
evaluasi/validasi manapun. Setiap rumus dijelaskan secara matematis.
=============================================================================

LANDASAN MATEMATIS — K-FOLD CROSS VALIDATION
==============================================

K-Fold Cross Validation membagi dataset D menjadi K subset (fold) berukuran
sama. Pada setiap iterasi i (i = 1, ..., K):
  - Fold ke-i digunakan sebagai validation set (data uji)
  - K-1 fold lainnya digabung menjadi training set (data latih)

Tujuan:
  Mengevaluasi performa model secara lebih robust dan mengurangi bias
  yang muncul dari satu kali pembagian train-test split.

Rumus Rata-rata Metrik:
  mean(m) = (1/K) * Σᵢ₌₁ᴷ mᵢ
  dimana mᵢ = nilai metrik pada fold ke-i

Rumus Standar Deviasi (Population):
  σ = √[ (1/K) * Σᵢ₌₁ᴷ (mᵢ - mean(m))² ]

  Menggunakan population std (bukan sample std) karena kita mengukur
  seluruh K fold, bukan mengambil sampel dari populasi fold.

Data Shuffling (Fisher-Yates / Knuth Shuffle):
  Untuk mengacak urutan data sebelum pembagian fold.
  Algoritma:
    FOR i = N-1 DOWN TO 1:
      j = random_integer(0, i)     # 0 ≤ j ≤ i
      SWAP(arr[i], arr[j])
  Kompleksitas: O(N), menghasilkan permutasi acak seragam (uniform).
  Menggunakan seed internal untuk reprodusibilitas.

Linear Congruential Generator (LCG) — PRNG Internal:
  Xₙ₊₁ = (a * Xₙ + c) mod m
  dimana:
    a = 1664525      (multiplier, konstanta Numerical Recipes)
    c = 1013904223   (increment)
    m = 2³²          (modulus)
  Seed awal X₀ menentukan seluruh urutan → reproducible.
=============================================================================
"""

import math
import time

# Import modul internal proyek
from modules.tfidf import fit_transform, transform
from modules.svm import SVMScratch
from modules.naive_bayes import MultinomialNBScratch
from modules.random_forest import RandomForestScratch
from modules.ensemble import (
    ensemble_predict, weighted_soft_voting_predict, evaluate_model, compare_models
)


# =============================================================================
# BAGIAN 1: PSEUDO-RANDOM NUMBER GENERATOR (LCG) INTERNAL
# =============================================================================

class LCGRandom:
    """
    Linear Congruential Generator (LCG) — PRNG deterministik.

    Rumus:
      Xₙ₊₁ = (a * Xₙ + c) mod m

    Konstanta dari Numerical Recipes (Knuth):
      a = 1664525, c = 1013904223, m = 2³²

    Properti:
      - Periode maksimum = m = 4,294,967,296
      - Deterministik: seed yang sama → urutan yang sama
      - Cukup untuk keperluan shuffling data (bukan kriptografi)
    """

    def __init__(self, seed=42):
        """
        Args:
            seed : Nilai awal X₀ untuk generator
        """
        self._state = seed & 0xFFFFFFFF  # Pastikan 32-bit
        # Konstanta LCG (Numerical Recipes)
        self._a = 1664525
        self._c = 1013904223
        self._m = 2 ** 32  # 4,294,967,296

    def next_int(self):
        """
        Generate bilangan bulat acak berikutnya.

        Xₙ₊₁ = (a * Xₙ + c) mod m
        """
        self._state = (self._a * self._state + self._c) % self._m
        return self._state

    def randint(self, low, high):
        """
        Generate bilangan bulat acak dalam rentang [low, high] (inklusif).

        Metode: next_int() mod (high - low + 1) + low
        """
        if low > high:
            low, high = high, low
        return low + (self.next_int() % (high - low + 1))


# =============================================================================
# BAGIAN 2: DATA SHUFFLING (FISHER-YATES SHUFFLE)
# =============================================================================

def shuffle_indices(n, seed=42):
    """
    Mengacak indeks [0, 1, ..., n-1] menggunakan Fisher-Yates Shuffle
    dengan PRNG internal (LCG) untuk reprodusibilitas.

    Algoritma Fisher-Yates (Knuth Shuffle):
      INPUT : Array A = [0, 1, 2, ..., n-1]
      FOR i = n-1 DOWN TO 1:
        j = random(0, i)       ← bilangan acak seragam 0 ≤ j ≤ i
        SWAP(A[i], A[j])
      OUTPUT: A (permutasi acak seragam)

    Properti Matematis:
      - Menghasilkan permutasi acak seragam (uniform random permutation)
      - Setiap permutasi memiliki probabilitas yang sama: 1/n!
      - Kompleksitas: O(n) waktu, O(n) ruang

    Mengapa Fisher-Yates?
      Metode naif (sort dengan random key) menghasilkan distribusi TIDAK
      seragam dan memiliki kompleksitas O(n log n). Fisher-Yates adalah
      O(n) dan terbukti menghasilkan distribusi seragam sempurna.

    Args:
        n    : Jumlah elemen (ukuran dataset)
        seed : Seed untuk PRNG internal (default: 42)

    Returns:
        List[int] — indeks teracak [0..n-1] dalam urutan acak
    """
    rng = LCGRandom(seed=seed)

    # Inisialisasi array indeks: [0, 1, 2, ..., n-1]
    indices = list(range(n))

    # Fisher-Yates: iterasi dari belakang ke depan
    for i in range(n - 1, 0, -1):
        # j = bilangan acak dalam [0, i]
        j = rng.randint(0, i)
        # SWAP: tukar elemen di posisi i dan j
        indices[i], indices[j] = indices[j], indices[i]

    return indices


# =============================================================================
# BAGIAN 3: FOLD SPLITTING
# =============================================================================

def create_folds(data, k=5, seed=42):
    """
    Membagi dataset menjadi K fold berukuran (kurang lebih) sama.

    Algoritma Pembagian:
      1. Acak indeks dataset menggunakan Fisher-Yates Shuffle
      2. Hitung ukuran dasar per fold: base_size = ⌊N/K⌋
      3. Sisa pembagian: remainder = N mod K
      4. K fold pertama yang remainder > 0 mendapat 1 elemen ekstra

    Distribusi Ukuran Fold:
      Jika N = 103, K = 5:
        base_size = ⌊103/5⌋ = 20
        remainder = 103 mod 5 = 3
        Fold sizes: [21, 21, 21, 20, 20]  (3 fold pertama +1)
        Total: 21+21+21+20+20 = 103 ✓

    Mengapa Shuffle Sebelum Split?
      Tanpa shuffle, fold akan berisi data yang berurutan sesuai
      dataset asli. Jika dataset diurutkan berdasarkan label/waktu,
      maka distribusi kelas per fold akan TIDAK representatif
      (mis: fold 1 = semua hoax, fold 2 = semua fakta).
      Shuffle memastikan distribusi kelas lebih merata per fold.

    Args:
        data : List[Dict] — dataset (List of Dictionaries)
        k    : int — jumlah fold (default: 5)
        seed : int — seed untuk shuffling

    Returns:
        List[List[Dict]] — K buah fold, masing-masing berisi subset data
    """
    n = len(data)

    if k <= 0:
        raise ValueError("K harus > 0")
    if k > n:
        raise ValueError(f"K ({k}) tidak boleh lebih besar dari N ({n})")

    # ── Langkah 1: Acak indeks ────────────────────────────────────────────
    shuffled_indices = shuffle_indices(n, seed=seed)

    # ── Langkah 2: Hitung ukuran per fold ─────────────────────────────────
    # base_size = ⌊N/K⌋ = ukuran minimum setiap fold
    # remainder = N mod K = jumlah fold yang mendapat +1 elemen
    base_size = n // k
    remainder = n % k

    # ── Langkah 3: Bagi indeks teracak ke dalam K fold ────────────────────
    folds = []
    start = 0

    for i in range(k):
        # Fold ke-i mendapat base_size elemen
        # Jika i < remainder, tambah 1 elemen (distribusi sisa)
        fold_size = base_size + (1 if i < remainder else 0)
        end = start + fold_size

        # Ambil data sesuai indeks teracak pada rentang [start, end)
        fold_indices = shuffled_indices[start:end]
        fold_data = [data[idx] for idx in fold_indices]
        folds.append(fold_data)

        start = end

    return folds


# =============================================================================
# BAGIAN 4: CROSS VALIDATION LOOP
# =============================================================================

def run_cross_validation(dataset, k=5, seed=42,
                         text_col='normalized',
                         label_col='label',
                         svm_params=None,
                         nb_params=None,
                         rf_params=None,
                         tfidf_params=None,
                         ensemble_method='hard',
                         ensemble_weights=None,
                         progress_callback=None):
    """
    Menjalankan K-Fold Cross Validation lengkap.

    Pada setiap iterasi i (i = 0, 1, ..., K-1):
      1. Fold ke-i → validation set (data uji)
      2. K-1 fold lainnya → training set (data latih)
      3. Ekstraksi fitur TF-IDF terpisah (fit pada train, transform pada test)
      4. Latih 3 model individu: SVM, Naive Bayes, Random Forest
      5. Jalankan Majority Voting (Hard) ATAU Weighted Soft Voting Ensemble
      6. Evaluasi: Akurasi, Presisi, Recall, F1-Score

    Mengapa TF-IDF Dihitung Ulang Setiap Fold?
      Vocabulary dan IDF HARUS dihitung HANYA dari data training.
      Jika vocabulary/IDF dihitung dari seluruh data (termasuk test),
      terjadi DATA LEAKAGE — informasi dari test bocor ke model →
      estimasi performa terlalu optimistik (overly optimistic).

      Pada fold i:
        vocab_i = build_vocabulary(train_corpus_i)
        idf_i   = compute_idf(train_corpus_i, vocab_i)
        X_train = fit_transform(train_corpus_i)
        X_test  = transform(test_corpus_i, vocab_i, idf_i)

    Args:
        dataset     : List[Dict] — dataset lengkap
        k           : int — jumlah fold
        seed        : int — seed untuk reprodusibilitas
        text_col    : str — key untuk teks yang sudah dipreprocess
        label_col   : str — key untuk label klasifikasi
        svm_params  : Dict — parameter SVM (C, kernel, dll)
        nb_params   : Dict — parameter Naive Bayes (alpha)
        rf_params   : Dict — parameter Random Forest (n_trees, max_depth)
        tfidf_params: Dict — parameter TF-IDF (min_df, max_df_ratio)
        ensemble_method : str — 'hard' atau 'soft' (default: 'hard')
        ensemble_weights: Dict — bobot soft voting {'svm': 0.4, 'nb': 0.32, 'rf': 0.28}
        progress_callback : callable — callback(fold_idx, total_folds, msg)

    Returns:
        Dict berisi:
          'fold_results'    : List[Dict] — hasil evaluasi per fold
          'average_metrics' : Dict — rata-rata metrik semua fold
          'std_metrics'     : Dict — standar deviasi metrik
          'summary_table'   : List[Dict] — tabel ringkasan per model
          'k'               : int — jumlah fold
          'seed'            : int — seed yang digunakan
    """
    # ── Default parameter ─────────────────────────────────────────────────
    if svm_params is None:
        svm_params = {'kernel': 'linear', 'C': 1.0, 'tol': 1e-3, 'max_passes': 5}
    if nb_params is None:
        nb_params = {'alpha': 1.0}
    if rf_params is None:
        rf_params = {'n_trees': 10, 'max_depth': 10, 'min_samples': 2,
                     'max_features': 'sqrt'}
    if tfidf_params is None:
        tfidf_params = {'min_df': 1, 'max_df_ratio': 1.0, 'smooth_idf': True,
                        'normalize': True}
    if ensemble_weights is None:
        ensemble_weights = {'svm': 0.4, 'nb': 0.32, 'rf': 0.28}

    # ── Langkah 1: Bagi dataset menjadi K fold ───────────────────────────
    folds = create_folds(dataset, k=k, seed=seed)

    # Nama model yang akan dievaluasi (termasuk ensemble)
    model_names = ['SVM', 'Naive Bayes', 'Random Forest', 'Ensemble (Voting)']
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']

    # Struktur penyimpanan metrik per fold per model
    # all_metrics[model_name][metric_name] = [val_fold_0, ..., val_fold_K-1]
    all_metrics = {
        name: {m: [] for m in metric_names}
        for name in model_names
    }

    fold_results = []

    # Akumulator untuk Aggregated Confusion Matrix dan Total Execution Time (ms)
    total_cm = {name: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} for name in model_names}
    execution_times_ms = {name: [] for name in model_names}

    # ── Langkah 2: Loop K-Fold ────────────────────────────────────────────
    for i in range(k):
        if progress_callback:
            progress_callback(i, k, f"Memproses fold {i+1}/{k}...")

        # ── 2a: Pisahkan validation set dan training set ──────────────────
        # Fold ke-i = validation, sisanya = training
        val_data = folds[i]
        train_data = []
        for j in range(k):
            if j != i:
                # Gabungkan K-1 fold lainnya → training set
                train_data.extend(folds[j])

        # ── 2b: Ekstrak teks dan label ────────────────────────────────────
        train_corpus = [item.get(text_col, '') for item in train_data]
        val_corpus = [item.get(text_col, '') for item in val_data]

        # Konversi label ke integer
        train_labels = []
        for item in train_data:
            try:
                train_labels.append(int(item.get(label_col, 0)))
            except (ValueError, TypeError):
                train_labels.append(0)

        val_labels = []
        for item in val_data:
            try:
                val_labels.append(int(item.get(label_col, 0)))
            except (ValueError, TypeError):
                val_labels.append(0)

        # ── 2c: Ekstraksi TF-IDF (fit pada train, transform pada test) ───
        # fit_transform: bangun vocabulary + IDF dari training set SAJA
        tfidf_train = fit_transform(
            train_corpus,
            min_df=tfidf_params.get('min_df', 1),
            max_df_ratio=tfidf_params.get('max_df_ratio', 1.0),
            smooth_idf=tfidf_params.get('smooth_idf', True),
            normalize=tfidf_params.get('normalize', True),
        )

        X_train = tfidf_train['matrix']
        vocabulary = tfidf_train['vocabulary']
        idf_vector = tfidf_train['idf_vector']

        # transform: gunakan vocabulary & IDF dari training untuk test
        X_val = transform(
            val_corpus, vocabulary, idf_vector,
            normalize=tfidf_params.get('normalize', True)
        )

        # ── 2d: Konversi label SVM: 0 → -1 ───────────────────────────────
        # SVM menggunakan label {-1, +1}
        train_labels_svm = [1 if lbl == 1 else -1 for lbl in train_labels]

        # ── 2e: Latih model individu & Prediksi (dengan Pengukuran Waktu) ─
        # Seed deterministik per fold: seed_utama + nomor_fold
        # Memastikan hasil SELALU sama untuk seed dan fold yang sama,
        # berapapun kali tombol "Jalankan" ditekan.
        fold_seed = seed + i

        # --- SVM ---
        start_svm = time.time()
        svm_model = SVMScratch(
            kernel=svm_params.get('kernel', 'linear'),
            C=svm_params.get('C', 1.0),
            tol=svm_params.get('tol', 1e-3),
            max_passes=svm_params.get('max_passes', 5),
            sigma=svm_params.get('sigma', 0.5),
            seed=fold_seed,
        )
        svm_model.train(X_train, train_labels_svm)
        svm_preds_raw = svm_model.predict(X_val)
        svm_preds = [0 if p == -1 else 1 for p in svm_preds_raw]
        execution_times_ms['SVM'].append((time.time() - start_svm) * 1000)

        # --- Naive Bayes ---
        start_nb = time.time()
        nb_model = MultinomialNBScratch(
            alpha=nb_params.get('alpha', 1.0)
        )
        nb_model.train(X_train, train_labels)
        nb_preds = nb_model.predict(X_val)
        execution_times_ms['Naive Bayes'].append((time.time() - start_nb) * 1000)

        # --- Random Forest ---
        start_rf = time.time()
        rf_model = RandomForestScratch(
            n_trees=rf_params.get('n_trees', 200),
            max_depth=rf_params.get('max_depth', 15),
            min_samples=rf_params.get('min_samples', 2),
            max_features=rf_params.get('max_features', 'sqrt'),
            seed=fold_seed,
        )
        rf_model.build_forest(X_train, train_labels)
        rf_preds = rf_model.predict(X_val)
        execution_times_ms['Random Forest'].append((time.time() - start_rf) * 1000)

        # ── 2g: Ensemble Predict (Hard atau Soft) ─────────────────────────
        start_ens = time.time()
        ensemble_preds = []
        if ensemble_method == 'soft':
            # Print trace log hanya untuk fold pertama (agar tidak spam)
            verbose_trace = (i == 0)
            soft_res = weighted_soft_voting_predict(
                X_val, svm_model, nb_model, rf_model,
                w_svm=ensemble_weights.get('svm', 0.4),
                w_nb=ensemble_weights.get('nb', 0.32),
                w_rf=ensemble_weights.get('rf', 0.28),
                positive_label=1, negative_label=0,
                verbose=verbose_trace
            )
            ensemble_preds = soft_res['ensemble_predictions']
        else:
            # Majority Voting Ensemble (Hard)
            for idx in range(len(val_labels)):
                votes = [svm_preds[idx], nb_preds[idx], rf_preds[idx]]
                vote_count = {}
                for v in votes:
                    vote_count[v] = vote_count.get(v, 0) + 1
                winner = max(vote_count, key=vote_count.get)
                ensemble_preds.append(winner)
        execution_times_ms['Ensemble (Voting)'].append((time.time() - start_ens) * 1000)

        # ── 2h: Evaluasi setiap model ─────────────────────────────────────
        all_preds = {
            'SVM': svm_preds,
            'Naive Bayes': nb_preds,
            'Random Forest': rf_preds,
            'Ensemble (Voting)': ensemble_preds,
        }

        fold_eval = {}
        for model_name, preds in all_preds.items():
            eval_result = evaluate_model(val_labels, preds, positive_label=1)
            fold_eval[model_name] = eval_result
            
            # Akumulasi ke Aggregated Confusion Matrix
            cm = eval_result['confusion_matrix']
            total_cm[model_name]['TP'] += cm['TP']
            total_cm[model_name]['TN'] += cm['TN']
            total_cm[model_name]['FP'] += cm['FP']
            total_cm[model_name]['FN'] += cm['FN']

            # Catat metrik untuk averaging nanti
            for m in metric_names:
                all_metrics[model_name][m].append(eval_result[m])

        fold_results.append({
            'fold': i + 1,
            'train_size': len(train_data),
            'val_size': len(val_data),
            'n_features': len(vocabulary),
            'evaluations': fold_eval,
        })

    # ── Langkah 3: Hitung Rata-rata dan Standar Deviasi ───────────────────
    average_metrics = {}
    std_metrics = {}

    for model_name in model_names:
        average_metrics[model_name] = {}
        std_metrics[model_name] = {}

        for m in metric_names:
            values = all_metrics[model_name][m]
            avg = _compute_mean(values)
            std = _compute_std(values)
            average_metrics[model_name][m] = round(avg, 6)
            std_metrics[model_name][m] = round(std, 6)

    # ── Langkah 4: Buat Tabel Ringkasan ───────────────────────────────────
    
    # Hitung rata-rata waktu eksekusi
    avg_execution_time = {}
    for model_name in model_names:
        avg_execution_time[model_name] = _compute_mean(execution_times_ms[model_name])

    summary_table = _build_summary_table(
        model_names, metric_names, average_metrics, std_metrics,
        all_metrics, k, avg_execution_time
    )

    return {
        'fold_results': fold_results,
        'average_metrics': average_metrics,
        'std_metrics': std_metrics,
        'summary_table': summary_table,
        'aggregated_confusion_matrix': total_cm,
        'average_execution_time_ms': avg_execution_time,
        'all_fold_metrics': {
            name: {m: all_metrics[name][m] for m in metric_names}
            for name in model_names
        },
        'k': k,
        'seed': seed,
        'total_samples': len(dataset),
    }


# =============================================================================
# BAGIAN 5: FUNGSI STATISTIK — MEAN & STANDAR DEVIASI
# =============================================================================

def _compute_mean(values):
    """
    Menghitung rata-rata (mean) aritmetika.

    Rumus:
      mean(x) = (1/N) * Σᵢ₌₁ᴺ xᵢ

    dimana:
      N  = jumlah elemen dalam list
      xᵢ = elemen ke-i

    Kompleksitas: O(N)

    Args:
        values : List[float] — kumpulan nilai

    Returns:
        float — nilai rata-rata
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def _compute_std(values):
    """
    Menghitung Standar Deviasi Populasi.

    Rumus:
      σ = √[ (1/N) * Σᵢ₌₁ᴺ (xᵢ - μ)² ]

    dimana:
      μ  = mean(x) = rata-rata
      xᵢ = elemen ke-i
      N  = jumlah elemen

    Catatan:
      Menggunakan population standard deviation (pembagi N, bukan N-1)
      karena K fold bukan sampel dari populasi fold yang lebih besar,
      melainkan seluruh fold yang digunakan.

    Interpretasi:
      σ kecil → metrik stabil antar fold (model konsisten)
      σ besar → metrik fluktuatif (model sensitif terhadap pembagian data)

    Kompleksitas: O(N)

    Args:
        values : List[float] — kumpulan nilai metrik dari K fold

    Returns:
        float — standar deviasi populasi
    """
    if not values or len(values) <= 1:
        return 0.0

    n = len(values)
    mean = sum(values) / n

    # Variance: (1/N) * Σ(xᵢ - μ)²
    variance = sum((x - mean) ** 2 for x in values) / n

    # Standar deviasi = √variance
    return math.sqrt(variance)


# =============================================================================
# BAGIAN 6: TABEL RINGKASAN PERFORMA
# =============================================================================

def _build_summary_table(model_names, metric_names,
                         average_metrics, std_metrics,
                         all_metrics, k, avg_execution_time=None):
    """
    Membangun tabel ringkasan performa rata-rata K-Fold Cross Validation.

    Format Output (per model):
      {
        'model': 'SVM',
        'accuracy_mean': 0.85, 'accuracy_std': 0.02,
        'precision_mean': ...,  'precision_std': ...,
        'recall_mean': ...,     'recall_std': ...,
        'f1_score_mean': ...,   'f1_score_std': ...,
        'per_fold': { 'accuracy': [...], ... }
      }

    Args:
        model_names     : List[str]
        metric_names    : List[str]
        average_metrics : Dict[str, Dict[str, float]]
        std_metrics     : Dict[str, Dict[str, float]]
        all_metrics     : Dict — metrik per fold per model
        k               : int — jumlah fold

    Returns:
        List[Dict] — tabel ringkasan, diurutkan berdasarkan f1_score mean
    """
    table = []

    for model_name in model_names:
        row = {'model': model_name, 'k_folds': k}

        for m in metric_names:
            row[f'{m}_mean'] = average_metrics[model_name][m]
            row[f'{m}_std'] = std_metrics[model_name][m]
            
        if avg_execution_time:
            row['execution_time_ms'] = avg_execution_time[model_name]

        # Sertakan nilai per fold untuk detail
        row['per_fold'] = {
            m: [round(v, 6) for v in all_metrics[model_name][m]]
            for m in metric_names
        }

        table.append(row)

    # Urutkan berdasarkan F1-Score mean (descending)
    table.sort(key=lambda x: x.get('f1_score_mean', 0), reverse=True)

    return table


def format_cv_report(cv_result):
    """
    Format hasil cross validation menjadi string laporan yang mudah dibaca.

    Args:
        cv_result : Dict — output dari run_cross_validation()

    Returns:
        str — laporan dalam format tabel teks
    """
    k = cv_result['k']
    seed = cv_result['seed']
    total = cv_result['total_samples']

    lines = []
    lines.append("=" * 78)
    lines.append(f"  LAPORAN {k}-FOLD CROSS VALIDATION")
    lines.append(f"  Total Sampel: {total} | Seed: {seed}")
    lines.append("=" * 78)

    # Detail per fold
    for fr in cv_result['fold_results']:
        fold_num = fr['fold']
        lines.append(f"\n─── Fold {fold_num}/{k} "
                     f"(Train: {fr['train_size']}, Val: {fr['val_size']}, "
                     f"Fitur: {fr['n_features']}) ───")

        header = f"  {'Model':<22} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}"
        lines.append(header)
        lines.append("  " + "-" * 56)

        for model_name in ['SVM', 'Naive Bayes', 'Random Forest',
                           'Ensemble (Voting)']:
            ev = fr['evaluations'][model_name]
            lines.append(
                f"  {model_name:<22} "
                f"{ev['accuracy']:>8.4f} "
                f"{ev['precision']:>8.4f} "
                f"{ev['recall']:>8.4f} "
                f"{ev['f1_score']:>8.4f}"
            )

    # Ringkasan rata-rata
    lines.append("\n" + "=" * 78)
    lines.append(f"  RATA-RATA PERFORMA {k}-FOLD CROSS VALIDATION")
    lines.append("=" * 78)

    header = (f"  {'Model':<22} {'Acc':>12} {'Prec':>12} "
              f"{'Rec':>12} {'F1':>12}")
    lines.append(header)
    lines.append("  " + "-" * 72)

    avg = cv_result['average_metrics']
    std = cv_result['std_metrics']

    for model_name in ['SVM', 'Naive Bayes', 'Random Forest',
                       'Ensemble (Voting)']:
        a = avg[model_name]
        s = std[model_name]
        lines.append(
            f"  {model_name:<22} "
            f"{a['accuracy']:.4f}±{s['accuracy']:.4f} "
            f"{a['precision']:.4f}±{s['precision']:.4f} "
            f"{a['recall']:.4f}±{s['recall']:.4f} "
            f"{a['f1_score']:.4f}±{s['f1_score']:.4f}"
        )

    lines.append("=" * 78)

    return "\n".join(lines)
