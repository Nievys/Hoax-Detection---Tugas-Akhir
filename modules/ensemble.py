# BAGIAN 1: MAJORITY VOTING — ENSEMBLE PREDICT
def ensemble_predict(models, X_test):
    n_samples = len(X_test)

    # Langkah 1 Jalankan prediksi dari setiap model
    individual_preds = {}
    for name, model in models.items():
        preds = model.predict(X_test)

        # Konversi label SVM -1 jadi 0 agar konsisten
        if name == 'svm':
            preds = [0 if p == -1 else 1 for p in preds]

        individual_preds[name] = preds

    # Langkah 2 Majority Voting per sampel
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

# BAGIAN 1.5 WEIGHTED SOFT VOTING — ENSEMBLE PREDICT
def weighted_soft_voting_predict(X_test, svm_model, nb_model, rf_model, 
                                 w_svm=0.4, w_nb=0.32, w_rf=0.28, 
                                 positive_label=1, negative_label=0, verbose=False):
    n_samples = len(X_test)
    
    # Langkah 1 Validasi dan Normalisasi Bobot
    total_w = w_svm + w_nb + w_rf
    if abs(total_w - 1.0) > 1e-6:
        w_svm /= total_w
        w_nb /= total_w
        w_rf /= total_w
        
    # Langkah 2 Ekstraksi Probabilitas
    svm_probs = svm_model.predict_proba(X_test)
    nb_probs = nb_model.predict_proba(X_test)
    rf_probs = rf_model.predict_proba(X_test)
    
    ensemble_preds = []
    final_probs = []
    
    # Langkah 3 Kalkulasi Soft Voting per Sampel
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
        
        # Cetak trace log untuk 3 baris pertama
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

# BAGIAN 2 CONFUSION MATRIX — MANUAL
def compute_confusion_matrix(y_true, y_pred, positive_label=1):
    tp, tn, fp, fn = 0, 0, 0, 0

    for actual, predicted in zip(y_true, y_pred):
        if actual == positive_label and predicted == positive_label:
            tp += 1     # True Positive: aktual=positif, prediksi=positif 
        elif actual != positive_label and predicted != positive_label:
            tn += 1     # True Negative: aktual=negatif, prediksi=negatif 
        elif actual != positive_label and predicted == positive_label:
            fp += 1     # False Positive: aktual=negatif, prediksi=positif 
        elif actual == positive_label and predicted != positive_label:
            fn += 1     # False Negative: aktual=positif, prediksi=negatif 

    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


# BAGIAN 3: METRIK EVALUASI — MANUAL
def compute_accuracy(cm):
    total = cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']
    if total == 0:
        return 0.0
    return (cm['TP'] + cm['TN']) / total


def compute_precision(cm):
    denom = cm['TP'] + cm['FP']
    if denom == 0:
        return 0.0
    return cm['TP'] / denom


def compute_recall(cm):
    denom = cm['TP'] + cm['FN']
    if denom == 0:
        return 0.0
    return cm['TP'] / denom


def compute_f1_score(cm):
    precision = compute_precision(cm)
    recall = compute_recall(cm)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def evaluate_model(y_true, y_pred, positive_label=1):
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
    results = []

    for model_name, y_pred in predictions_dict.items():
        eval_result = evaluate_model(y_true, y_pred, positive_label)
        eval_result['model'] = model_name
        results.append(eval_result)

    # Urutkan berdasarkan akurasi (descending)
    results.sort(key=lambda x: x['accuracy'], reverse=True)

    return results
