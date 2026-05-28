from flask import Blueprint, request, jsonify
from core.state import _state
from modules.preprocessing import full_preprocessing_pipeline
from modules.tfidf import transform
from modules.svm import SVMScratch

svm_bp = Blueprint('svm', __name__)

@svm_bp.route('/api/svm/train', methods=['POST'])
def train_svm():
    if not _state['tfidf_result']:
        return jsonify({'success': False, 'error': 'Hitung TF-IDF terlebih dahulu!'}), 400

    data = request.json
    kernel = data.get('kernel', 'linear')
    C = float(data.get('C', 1.0))
    target_label = data.get('target_label', 'label')

    X = _state['tfidf_result']['matrix']

    try:
        y = [1 if int(item['labels'].get(target_label, 0)) == 1 else -1
             for item in _state['dataset']]
    except (ValueError, TypeError):
        return jsonify({'success': False,
                        'error': f'Nilai label "{target_label}" harus berupa angka (0 atau 1)!'}), 400

    if len(X) != len(y):
        return jsonify({'success': False,
                        'error': f'Jumlah fitur ({len(X)}) != jumlah label ({len(y)})'}), 400

    if len(set(y)) < 2:
        return jsonify({'success': False,
                        'error': 'Dataset harus memiliki minimal 2 kelas (+1 dan -1). '
                                 'Pastikan kolom label memiliki nilai 0 dan 1.'}), 400

    model = SVMScratch(kernel=kernel, C=C, max_passes=10)
    train_status = model.train(X, y)

    _state['svm_model'] = model
    _state['svm_target'] = target_label

    n_pos = sum(1 for label in y if label == 1)
    n_neg = sum(1 for label in y if label == -1)

    return jsonify({
        'success': True,
        'message': 'Model SVM berhasil dilatih',
        'support_vectors': train_status['support_vectors'],
        'training_info': {
            'total_samples': len(y),
            'positive_class': n_pos,
            'negative_class': n_neg,
            'kernel': kernel,
            'C': C,
            'target_label': target_label,
        }
    })

@svm_bp.route('/api/svm/predict', methods=['POST'])
def predict_text():
    if not _state['svm_model']:
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

    prediction_code = _state['svm_model'].predict([tfidf_vector])[0]

    target = _state.get('svm_target', 'Target')
    label = f"{target} Terdeteksi" if prediction_code == 1 else f"Bukan {target}"

    return jsonify({
        'success': True,
        'normalized': normalized_text,
        'prediction': label,
        'code': int(prediction_code)
    })
