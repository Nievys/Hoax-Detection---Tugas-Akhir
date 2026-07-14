from flask import Blueprint, request, jsonify
from core.state import _state
from modules.preprocessing import full_preprocessing_pipeline, parse_label
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
        y = [1 if parse_label(item['labels'].get(target_label, 0)) == 1 else -1
             for item in _state['dataset']]
    except (ValueError, TypeError) as e:
        return jsonify({'success': False,
                        'error': f'Nilai label "{target_label}" gagal diproses: {str(e)}'}), 400

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

    # Data hasil train yang digunakan untuk prediksi
    sv_detail = []
    for i in range(model.n_samples):
        if model.alpha[i] > 0:
            doc_text = _state['dataset'][i].get('text', '')[:100] if i < len(_state['dataset']) else f"Doc #{i}"
            sv_detail.append({
                'index': i,
                'alpha': round(model.alpha[i], 6),
                'label': model.y[i],
                'text_preview': doc_text
            })

    feature_weights = {}
    if kernel == 'linear' and _state['tfidf_result'] and 'feature_names' in _state['tfidf_result']:
        fn = _state['tfidf_result']['feature_names']
        w = [0.0] * len(fn)
        for i in range(model.n_samples):
            if model.alpha[i] > 0:
                for j in range(len(fn)):
                    w[j] += model.alpha[i] * model.y[i] * model.X[i][j]
        sorted_w = sorted([(fn[j], round(w[j], 6)) for j in range(len(fn))], key=lambda x: abs(x[1]), reverse=True)
        feature_weights = {item[0]: item[1] for item in sorted_w if abs(item[1]) > 1e-5}

    data_for_prediction = {
        'bias_b': round(model.b, 6),
        'n_support_vectors': len(sv_detail),
        'support_vectors_list': sv_detail,
        'linear_weights_w': feature_weights if kernel == 'linear' else "Non-linear kernel (menggunakan dot product Support Vectors langsung)"
    }

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
            'bias': round(model.b, 6),
            'n_features': model.n_features,
        },
        'data_for_prediction': data_for_prediction,
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
