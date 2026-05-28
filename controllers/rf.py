from flask import Blueprint, request, jsonify
from core.state import _state
from modules.preprocessing import full_preprocessing_pipeline
from modules.tfidf import transform
from modules.random_forest import RandomForestScratch

rf_bp = Blueprint('rf', __name__)

@rf_bp.route('/api/rf/train', methods=['POST'])
def train_rf():
    if not _state['tfidf_result']:
        return jsonify({'success': False, 'error': 'Hitung TF-IDF terlebih dahulu!'}), 400

    data = request.json or {}
    n_trees = int(data.get('n_trees', 10))
    max_depth = int(data.get('max_depth', 10))
    min_samples = int(data.get('min_samples', 2))
    target_label = data.get('target_label', 'label')

    X = _state['tfidf_result']['matrix']
    feature_names = _state['tfidf_result']['feature_names']

    try:
        y = [int(item['labels'].get(target_label, 0))
             for item in _state['dataset']]
    except (ValueError, TypeError):
        return jsonify({'success': False,
                        'error': f'Nilai label "{target_label}" harus berupa angka!'}), 400

    if len(X) != len(y):
        return jsonify({'success': False,
                        'error': f'Jumlah fitur ({len(X)}) != jumlah label ({len(y)})'}), 400
    if len(set(y)) < 2:
        return jsonify({'success': False,
                        'error': 'Dataset harus memiliki minimal 2 kelas.'}), 400

    model = RandomForestScratch(
        n_trees=n_trees, max_depth=max_depth,
        min_samples=min_samples, max_features='sqrt'
    )
    train_info = model.build_forest(X, y, feature_names=feature_names)

    _state['rf_model'] = model
    _state['rf_target'] = target_label

    class_dist = {}
    for label in y:
        class_dist[label] = class_dist.get(label, 0) + 1

    return jsonify({
        'success': True,
        'message': f'Random Forest ({n_trees} pohon) berhasil dilatih',
        'training_info': train_info,
        'total_samples': len(y),
        'class_distribution': {str(k): v for k, v in class_dist.items()},
    })

@rf_bp.route('/api/rf/predict', methods=['POST'])
def predict_rf():
    if not _state['rf_model']:
        return jsonify({'success': False, 'error': 'Model belum dilatih!'}), 400

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

    prediction = _state['rf_model'].predict([tfidf_vector])[0]
    proba = _state['rf_model'].predict_proba([tfidf_vector])[0]
    vote_detail = _state['rf_model'].get_vote_detail(tfidf_vector)

    target = _state.get('rf_target', 'Target')
    label = f"{target} Terdeteksi" if prediction == 1 else f"Bukan {target}"

    return jsonify({
        'success': True,
        'normalized': normalized_text,
        'prediction': label,
        'code': int(prediction),
        'probabilities': {str(k): round(v, 6) for k, v in proba.items()},
        'vote_detail': vote_detail,
    })
