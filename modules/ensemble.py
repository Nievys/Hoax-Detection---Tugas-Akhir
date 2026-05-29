"""
=============================================================================
MODUL 6: ENSEMBLE — MAJORITY VOTING
=============================================================================
Judul TA : Analisis Pengaruh Teknik Normalisasi Kata Gaul (Slang) terhadap
           Akurasi Deteksi Ujaran Kebencian Berbahasa Indonesia
=============================================================================
Implementasi MURNI Python. Tidak menggunakan Scikit-Learn atau library
evaluasi manapun. Setiap rumus dijelaskan secara matematis.
=============================================================================

LANDASAN MATEMATIS — ENSEMBLE MAJORITY VOTING
===============================================

Ensemble Learning menggabungkan prediksi dari beberapa model (classifier)
untuk menghasilkan prediksi yang lebih akurat dan robust.

Hard Voting (Majority Voting):
  Diberikan K model h₁, h₂, ..., hₖ, prediksi ensemble:

    ŷ_ensemble(x) = mode({ h₁(x), h₂(x), ..., hₖ(x) })

  Artinya: setiap model memberikan satu "suara" (vote), dan kelas
  dengan suara terbanyak menjadi prediksi final.

  Contoh (K=3: SVM, NB, RF):
    h_SVM(x) = 0,  h_NB(x) = 1,  h_RF(x) = 1
    → votes = {0: 1, 1: 2}
    → ŷ = 1 (menang 2 vs 1)

Mengapa Ensemble Lebih Baik?
  Condorcet's Jury Theorem (1785):
    Jika setiap classifier memiliki akurasi > 50% dan keputusan mereka
    independen, maka probabilitas keputusan mayoritas benar MENINGKAT
    seiring bertambahnya jumlah classifier.

    Untuk K classifier dengan akurasi p (p > 0.5):
      P(mayoritas benar) = Σ_{k=⌈K/2⌉}^{K} C(K,k) · p^k · (1-p)^{K-k}

    Contoh: 3 model dengan p=0.7 masing-masing
      P(≥2 benar) = C(3,2)·0.7²·0.3 + C(3,3)·0.7³ = 0.784

    → Akurasi ensemble (78.4%) > akurasi individu (70%)

─────────────────────────────────────────────────────────────────────

EVALUASI — CONFUSION MATRIX & METRIK
═══════════════════════════════════════

Confusion Matrix (untuk klasifikasi biner):
                     Prediksi
                   Positif  Negatif
  Aktual Positif [   TP   |   FN   ]
  Aktual Negatif [   FP   |   TN   ]

  TP (True Positive)  = benar positif (prediksi 1, aktual 1)
  TN (True Negative)  = benar negatif (prediksi 0, aktual 0)
  FP (False Positive)  = salah positif (prediksi 1, aktual 0) — Type I Error
  FN (False Negative) = salah negatif (prediksi 0, aktual 1) — Type II Error

Metrik Evaluasi:
  Akurasi   = (TP + TN) / (TP + TN + FP + FN)
  Presisi   = TP / (TP + FP)          — dari yang diprediksi positif, berapa yang benar?
  Recall    = TP / (TP + FN)          — dari yang aktual positif, berapa yang terdeteksi?
  F1-Score  = 2 · (Presisi · Recall) / (Presisi + Recall)  — harmonic mean
=============================================================================
"""


# =============================================================================
# BAGIAN 1: MAJORITY VOTING — ENSEMBLE PREDICT
# =============================================================================

def ensemble_predict(models, X_test):
    """
    Menjalankan prediksi dari semua model dan menggabungkan hasilnya
    menggunakan Hard Voting (Majority Voting).

    Rumus:
      ŷ_ensemble(x) = mode({ h₁(x), h₂(x), ..., hₖ(x) })

    Setiap model memberikan satu "suara". Kelas dengan suara terbanyak menang.

    Catatan tentang SVM:
      SVM menggunakan label {-1, +1}, sedangkan NB dan RF menggunakan {0, 1}.
      Sebelum voting, prediksi SVM dikonversi: -1 → 0, +1 → 1
      agar konsisten dengan model lain.

    Args:
        models : Dict[str, object] — {'svm': model, 'nb': model, 'rf': model}
                 Setiap model harus memiliki method predict(X_test)
        X_test : List[List[float]] — matriks fitur data test (M × V)

    Returns:
        Dict berisi:
          'ensemble_predictions' : List[int] — prediksi ensemble per sampel
          'individual_predictions': Dict[str, List[int]] — prediksi per model
          'vote_details' : List[Dict] — detail voting per sampel
    """
    n_samples = len(X_test)

    # ── Langkah 1: Jalankan prediksi dari setiap model ───────────────────
    individual_preds = {}
    for name, model in models.items():
        preds = model.predict(X_test)

        # Konversi label SVM: -1 → 0 agar konsisten
        if name == 'svm':
            preds = [0 if p == -1 else 1 for p in preds]

        individual_preds[name] = preds

    # ── Langkah 2: Majority Voting per sampel ────────────────────────────
    ensemble_preds = []
    vote_details = []

    for i in range(n_samples):
        # Kumpulkan vote dari setiap model untuk sampel ke-i
        votes = {}
        model_votes = {}
        for name in models:
            pred = individual_preds[name][i]
            votes[pred] = votes.get(pred, 0) + 1
            model_votes[name] = pred

        # Pilih kelas dengan suara terbanyak (majority)
        winner = max(votes, key=votes.get)
        ensemble_preds.append(winner)

        vote_details.append({
            'index': i,
            'votes': model_votes,
            'counts': votes,
            'ensemble': winner,
        })

    return {
        'ensemble_predictions': ensemble_preds,
        'individual_predictions': individual_preds,
        'vote_details': vote_details,
    }


# =============================================================================
# BAGIAN 1.5: WEIGHTED SOFT VOTING — ENSEMBLE PREDICT
# =============================================================================

def weighted_soft_voting_predict(X_test, svm_model, nb_model, rf_model, 
                                 w_svm=0.4, w_nb=0.32, w_rf=0.28, 
                                 positive_label=1, negative_label=0, verbose=False):
    """
    Menjalankan prediksi menggunakan Weighted Soft Voting Ensemble.
    
    Rumus Matematika:
      P_final(c) = (w_svm * P_svm(c) + w_nb * P_nb(c) + w_rf * P_rf(c)) / (w_svm + w_nb + w_rf)
      
    Aturan:
      - Validasi bobot: w_svm + w_nb + w_rf = 1.0 (jika tidak, dinormalisasi otomatis).
      - Menggunakan probabilitas prediksi (predict_proba) dari masing-masing model.
      - Jika P_final(positive_label) >= 0.5, prediksi = positive_label (Hoax).
      - Jika tidak, prediksi = negative_label (Valid).
      
    Args:
        X_test         : Matriks fitur (N x V)
        svm_model      : Model SVM yang sudah dilatih (harus memiliki predict_proba)
        nb_model       : Model Naive Bayes yang sudah dilatih
        rf_model       : Model Random Forest yang sudah dilatih
        w_svm          : Bobot untuk SVM (default: 0.4)
        w_nb           : Bobot untuk Naive Bayes (default: 0.32)
        w_rf           : Bobot untuk Random Forest (default: 0.28)
        positive_label : Kelas positif (default: 1)
        negative_label : Kelas negatif (default: 0)
        verbose        : Jika True, cetak trace log probabilitas untuk beberapa sampel pertama.
        
    Returns:
        Dict berisi:
          'ensemble_predictions' : List[int] — prediksi akhir
          'final_probabilities'  : List[float] — probabilitas kelas positif per sampel
    """
    n_samples = len(X_test)
    
    # ── Langkah 1: Validasi dan Normalisasi Bobot ─────────────────────────
    total_w = w_svm + w_nb + w_rf
    if abs(total_w - 1.0) > 1e-6:
        w_svm /= total_w
        w_nb /= total_w
        w_rf /= total_w
        
    # ── Langkah 2: Ekstraksi Probabilitas ─────────────────────────────────
    svm_probs = svm_model.predict_proba(X_test)
    nb_probs = nb_model.predict_proba(X_test)
    rf_probs = rf_model.predict_proba(X_test)
    
    ensemble_preds = []
    final_probs = []
    
    # ── Langkah 3: Kalkulasi Soft Voting per Sampel ───────────────────────
    for i in range(n_samples):
        # SVM menggunakan kunci kelas 1 dan -1
        # NB dan RF menggunakan kunci positive_label (biasanya 1) dan negative_label (0)
        p_svm = svm_probs[i].get(1, 0.0)
        p_nb = nb_probs[i].get(positive_label, 0.0)
        p_rf = rf_probs[i].get(positive_label, 0.0)
        
        # Hitung rata-rata probabilitas terbobot (weighted average probability)
        p_final = (w_svm * p_svm) + (w_nb * p_nb) + (w_rf * p_rf)
        
        # Thresholding
        if p_final >= 0.5:
            pred = positive_label
        else:
            pred = negative_label
            
        ensemble_preds.append(pred)
        final_probs.append(p_final)
        
        # Cetak trace log untuk 3 baris pertama (jika verbose = True)
        if verbose and i < 3:
            print(f"--- Trace Log Sampel {i+1} ---")
            print(f"P_SVM(1) = {p_svm:.4f} (w={w_svm:.4f})")
            print(f"P_NB(1)  = {p_nb:.4f} (w={w_nb:.4f})")
            print(f"P_RF(1)  = {p_rf:.4f} (w={w_rf:.4f})")
            print(f"P_Final  = ({w_svm:.4f}*{p_svm:.4f}) + ({w_nb:.4f}*{p_nb:.4f}) + ({w_rf:.4f}*{p_rf:.4f}) = {p_final:.4f}")
            print(f"Prediksi = {pred}\n")
            
    return {
        'ensemble_predictions': ensemble_preds,
        'final_probabilities': final_probs
    }


# =============================================================================
# BAGIAN 2: CONFUSION MATRIX — MANUAL
# =============================================================================

def compute_confusion_matrix(y_true, y_pred, positive_label=1):
    """
    Menghitung Confusion Matrix secara manual (tanpa library).

    Definisi:
      TP = Σ 𝟙[yᵢ = positif AND ŷᵢ = positif]  (benar positif)
      TN = Σ 𝟙[yᵢ = negatif AND ŷᵢ = negatif]  (benar negatif)
      FP = Σ 𝟙[yᵢ = negatif AND ŷᵢ = positif]  (salah positif — Type I)
      FN = Σ 𝟙[yᵢ = positif AND ŷᵢ = negatif]  (salah negatif — Type II)

    Kompleksitas: O(N) — satu iterasi pada semua sampel

    Args:
        y_true         : List[int] — label aktual
        y_pred         : List[int] — label prediksi
        positive_label : int — nilai yang dianggap kelas positif (default: 1)

    Returns:
        Dict {'TP': int, 'TN': int, 'FP': int, 'FN': int}
    """
    tp, tn, fp, fn = 0, 0, 0, 0

    for actual, predicted in zip(y_true, y_pred):
        if actual == positive_label and predicted == positive_label:
            tp += 1     # True Positive: aktual=positif, prediksi=positif ✓
        elif actual != positive_label and predicted != positive_label:
            tn += 1     # True Negative: aktual=negatif, prediksi=negatif ✓
        elif actual != positive_label and predicted == positive_label:
            fp += 1     # False Positive: aktual=negatif, prediksi=positif ✗
        elif actual == positive_label and predicted != positive_label:
            fn += 1     # False Negative: aktual=positif, prediksi=negatif ✗

    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


# =============================================================================
# BAGIAN 3: METRIK EVALUASI — MANUAL
# =============================================================================

def compute_accuracy(cm):
    """
    Menghitung Akurasi dari confusion matrix.

    Rumus:
      Akurasi = (TP + TN) / (TP + TN + FP + FN)

    Interpretasi:
      Proporsi prediksi yang benar dari total prediksi.
      Range: [0, 1] → 1 = semua prediksi benar.

    Args:
        cm : Dict — confusion matrix {'TP', 'TN', 'FP', 'FN'}

    Returns:
        float — nilai akurasi [0, 1]
    """
    total = cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']
    if total == 0:
        return 0.0
    return (cm['TP'] + cm['TN']) / total


def compute_precision(cm):
    """
    Menghitung Presisi dari confusion matrix.

    Rumus:
      Presisi = TP / (TP + FP)

    Interpretasi:
      Dari semua yang DIPREDIKSI positif, berapa proporsi yang BENAR positif?
      Presisi tinggi → sedikit false alarm (FP rendah).

    Args:
        cm : Dict — confusion matrix

    Returns:
        float — nilai presisi [0, 1]
    """
    denom = cm['TP'] + cm['FP']
    if denom == 0:
        return 0.0
    return cm['TP'] / denom


def compute_recall(cm):
    """
    Menghitung Recall (Sensitivity / True Positive Rate).

    Rumus:
      Recall = TP / (TP + FN)

    Interpretasi:
      Dari semua yang SEBENARNYA positif, berapa proporsi yang TERDETEKSI?
      Recall tinggi → sedikit kasus positif yang terlewat (FN rendah).

    Args:
        cm : Dict — confusion matrix

    Returns:
        float — nilai recall [0, 1]
    """
    denom = cm['TP'] + cm['FN']
    if denom == 0:
        return 0.0
    return cm['TP'] / denom


def compute_f1_score(cm):
    """
    Menghitung F1-Score dari confusion matrix.

    Rumus:
      F1 = 2 · (Presisi · Recall) / (Presisi + Recall)

    Interpretasi:
      Harmonic mean dari Presisi dan Recall.
      F1 tinggi hanya jika KEDUA presisi dan recall tinggi.
      F1 = 0 jika salah satu = 0.
      F1 = 1 hanya jika presisi = recall = 1 (sempurna).

    Mengapa Harmonic Mean?
      Harmonic mean memberikan bobot lebih pada nilai yang rendah.
      Jika presisi = 1.0 dan recall = 0.01:
        Arithmetic mean = (1.0 + 0.01) / 2 = 0.505 (menyesatkan!)
        Harmonic mean   = 2·(1.0·0.01)/(1.0+0.01) = 0.0198 (realistis)

    Args:
        cm : Dict — confusion matrix

    Returns:
        float — nilai F1-score [0, 1]
    """
    precision = compute_precision(cm)
    recall = compute_recall(cm)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def evaluate_model(y_true, y_pred, positive_label=1):
    """
    Evaluasi lengkap satu model: Confusion Matrix + semua metrik.

    Pipeline Evaluasi:
      1. Hitung Confusion Matrix (TP, TN, FP, FN)
      2. Hitung Akurasi   = (TP+TN) / total
      3. Hitung Presisi   = TP / (TP+FP)
      4. Hitung Recall    = TP / (TP+FN)
      5. Hitung F1-Score  = 2·P·R / (P+R)

    Args:
        y_true         : List[int] — label aktual
        y_pred         : List[int] — label prediksi
        positive_label : int — kelas positif (default: 1)

    Returns:
        Dict dengan semua metrik evaluasi
    """
    cm = compute_confusion_matrix(y_true, y_pred, positive_label)

    accuracy = compute_accuracy(cm)
    precision = compute_precision(cm)
    recall = compute_recall(cm)
    f1 = compute_f1_score(cm)

    return {
        'confusion_matrix': cm,
        'accuracy': round(accuracy, 6),
        'precision': round(precision, 6),
        'recall': round(recall, 6),
        'f1_score': round(f1, 6),
        'total_samples': cm['TP'] + cm['TN'] + cm['FP'] + cm['FN'],
        'correct': cm['TP'] + cm['TN'],
        'incorrect': cm['FP'] + cm['FN'],
    }


def compare_models(y_true, predictions_dict, positive_label=1):
    """
    Membandingkan akurasi beberapa model secara side-by-side.

    Args:
        y_true           : List[int] — label aktual
        predictions_dict : Dict[str, List[int]] — {nama_model: prediksi}
        positive_label   : int — kelas positif

    Returns:
        List[Dict] — evaluasi per model, diurutkan berdasarkan akurasi (desc)
    """
    results = []

    for model_name, y_pred in predictions_dict.items():
        eval_result = evaluate_model(y_true, y_pred, positive_label)
        eval_result['model'] = model_name
        results.append(eval_result)

    # Urutkan berdasarkan akurasi (descending)
    results.sort(key=lambda x: x['accuracy'], reverse=True)

    return results
