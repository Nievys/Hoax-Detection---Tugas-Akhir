from flask import Blueprint, request, jsonify
from core.state import _state
from modules.preprocessing import full_preprocessing_pipeline
from modules.tfidf import transform
from modules.ensemble import ensemble_predict, weighted_soft_voting_predict, compare_models

ensemble_bp = Blueprint('ensemble', __name__)

@ensemble_bp.route('/api/ensemble/evaluate', methods=['POST'])
def evaluate_ensemble():
    missing = []
    if not _state['svm_model']:  missing.append('SVM')
    if not _state['nb_model']:   missing.append('Naive Bayes')
    if not _state['rf_model']:   missing.append('Random Forest')
    if missing:
        return jsonify({'success': False,
                        'error': f'Model belum dilatih: {", ".join(missing)}'}), 400

    if not _state['tfidf_result']:
        return jsonify({'success': False, 'error': 'TF-IDF belum dihitung!'}), 400

    data = request.json or {}
    target_label = data.get('target_label', 'label')

    X = _state['tfidf_result']['matrix']

    try:
        y_true = [int(item['labels'].get(target_label, 0))
                  for item in _state['dataset']]
    except (ValueError, TypeError):
        return jsonify({'success': False,
                        'error': f'Nilai label "{target_label}" harus berupa angka!'}), 400

    models = {
        'SVM': _state['svm_model'],
        'Naive Bayes': _state['nb_model'],
        'Random Forest': _state['rf_model'],
    }
    ens_result = ensemble_predict(models, X)

    all_preds = dict(ens_result['individual_predictions'])
    all_preds['Ensemble (Voting)'] = ens_result['ensemble_predictions']

    comparison = compare_models(y_true, all_preds, positive_label=1)

    return jsonify({
        'success': True,
        'comparison': comparison,
        'total_samples': len(y_true),
    })

@ensemble_bp.route('/api/ensemble/predict', methods=['POST'])
def predict_ensemble():
    missing = []
    if not _state['svm_model']:  missing.append('SVM')
    if not _state['nb_model']:   missing.append('Naive Bayes')
    if not _state['rf_model']:   missing.append('Random Forest')
    if missing:
        return jsonify({'success': False,
                        'error': f'Model belum dilatih: {", ".join(missing)}'}), 400

    raw_text = request.json.get('text', '')
    if not raw_text.strip():
        return jsonify({'success': False, 'error': 'Teks tidak boleh kosong!'}), 400

    pp_result = full_preprocessing_pipeline(raw_text, _state['lexicon_merged'], verbose=False)
    normalized_text = pp_result['normalized']

    tfidf_vector = transform(
        [normalized_text],
        _state['tfidf_result']['vocabulary'],
        _state['tfidf_result']['idf_vector']
    )[0]

    svm_pred = _state['svm_model'].predict([tfidf_vector])[0]
    svm_pred_normalized = 0 if svm_pred == -1 else 1

    nb_pred = _state['nb_model'].predict([tfidf_vector])[0]
    rf_pred = _state['rf_model'].predict([tfidf_vector])[0]

    votes = {'SVM': svm_pred_normalized, 'Naive Bayes': nb_pred, 'Random Forest': rf_pred}
    vote_counts = {}
    for v in votes.values():
        vote_counts[v] = vote_counts.get(v, 0) + 1
    ensemble_pred = max(vote_counts, key=vote_counts.get)

    target = _state.get('nb_target', 'Target')
    label = f"{target} Terdeteksi" if ensemble_pred == 1 else f"Bukan {target}"

    return jsonify({
        'success': True,
        'normalized': normalized_text,
        'prediction': label,
        'code': int(ensemble_pred),
        'votes': votes,
        'vote_counts': {str(k): v for k, v in vote_counts.items()},
    })

@ensemble_bp.route('/api/ensemble/evaluate_soft', methods=['POST'])
def evaluate_ensemble_soft():
    missing = []
    if not _state['svm_model']:  missing.append('SVM')
    if not _state['nb_model']:   missing.append('Naive Bayes')
    if not _state['rf_model']:   missing.append('Random Forest')
    if missing:
        return jsonify({'success': False,
                        'error': f'Model belum dilatih: {", ".join(missing)}'}), 400

    if not _state['tfidf_result']:
        return jsonify({'success': False, 'error': 'TF-IDF belum dihitung!'}), 400

    data = request.json or {}
    target_label = data.get('target_label', 'label')
    weights = data.get('weights', {'svm': 0.4, 'nb': 0.32, 'rf': 0.28})
    
    X = _state['tfidf_result']['matrix']

    try:
        y_true = [int(item['labels'].get(target_label, 0))
                  for item in _state['dataset']]
    except (ValueError, TypeError):
        return jsonify({'success': False,
                        'error': f'Nilai label "{target_label}" harus berupa angka!'}), 400

    soft_res = weighted_soft_voting_predict(
        X, _state['svm_model'], _state['nb_model'], _state['rf_model'],
        w_svm=weights.get('svm', 0.4),
        w_nb=weights.get('nb', 0.32),
        w_rf=weights.get('rf', 0.28),
        positive_label=1, negative_label=0
    )

    # To do comparison, we need individual predictions
    models = {
        'SVM': _state['svm_model'],
        'Naive Bayes': _state['nb_model'],
        'Random Forest': _state['rf_model'],
    }
    hard_res = ensemble_predict(models, X)
    
    all_preds = dict(hard_res['individual_predictions'])
    all_preds['Ensemble (Soft Voting)'] = soft_res['ensemble_predictions']

    comparison = compare_models(y_true, all_preds, positive_label=1)

    return jsonify({
        'success': True,
        'comparison': comparison,
        'total_samples': len(y_true),
        'weights': weights
    })

@ensemble_bp.route('/api/ensemble/predict_soft', methods=['POST'])
def predict_ensemble_soft():
    missing = []
    if not _state['svm_model']:  missing.append('SVM')
    if not _state['nb_model']:   missing.append('Naive Bayes')
    if not _state['rf_model']:   missing.append('Random Forest')
    if missing:
        return jsonify({'success': False,
                        'error': f'Model belum dilatih: {", ".join(missing)}'}), 400

    data = request.json or {}
    raw_text = data.get('text', '')
    weights = data.get('weights', {'svm': 0.4, 'nb': 0.32, 'rf': 0.28})
    
    if not raw_text.strip():
        return jsonify({'success': False, 'error': 'Teks tidak boleh kosong!'}), 400

    pp_result = full_preprocessing_pipeline(raw_text, _state['lexicon_merged'], verbose=False)
    normalized_text = pp_result['normalized']

    tfidf_vector = transform(
        [normalized_text],
        _state['tfidf_result']['vocabulary'],
        _state['tfidf_result']['idf_vector']
    )[0]

    soft_res = weighted_soft_voting_predict(
        [tfidf_vector], _state['svm_model'], _state['nb_model'], _state['rf_model'],
        w_svm=weights.get('svm', 0.4),
        w_nb=weights.get('nb', 0.32),
        w_rf=weights.get('rf', 0.28),
        positive_label=1, negative_label=0
    )
    
    ensemble_pred = soft_res['ensemble_predictions'][0]
    final_prob = soft_res['final_probabilities'][0]

    target = _state.get('nb_target', 'Target')
    label = f"{target} Terdeteksi" if ensemble_pred == 1 else f"Bukan {target}"

    return jsonify({
        'success': True,
        'normalized': normalized_text,
        'prediction': label,
        'code': int(ensemble_pred),
        'probability': final_prob,
        'weights': weights
    })
