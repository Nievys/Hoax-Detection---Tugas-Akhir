import math
from flask import Blueprint, request, jsonify
from core.state import _state
from modules.preprocessing import full_preprocessing_pipeline, parse_label
from modules.tfidf import transform
from modules.naive_bayes import MultinomialNBScratch

nb_bp = Blueprint('nb', __name__)

@nb_bp.route('/api/nb/train', methods=['POST'])
def train_nb():
    if not _state['tfidf_result']:
        return jsonify({'success': False, 'error': 'Hitung TF-IDF terlebih dahulu!'}), 400

    data = request.json or {}
    alpha = float(data.get('alpha', 1.0))
    target_label = data.get('target_label', 'label')

    X = _state['tfidf_result']['matrix']
    feature_names = _state['tfidf_result']['feature_names']

    try:
        y = [parse_label(item['labels'].get(target_label, 0))
             for item in _state['dataset']]
    except (ValueError, TypeError) as e:
        return jsonify({'success': False,
                        'error': f'Nilai label "{target_label}" gagal diproses: {str(e)}'}), 400

    if len(X) != len(y):
        return jsonify({'success': False,
                        'error': f'Jumlah fitur ({len(X)}) != jumlah label ({len(y)})'}), 400

    if len(set(y)) < 2:
        return jsonify({'success': False,
                        'error': 'Dataset harus memiliki minimal 2 kelas.'}), 400

    model = MultinomialNBScratch(alpha=alpha)
    train_info = model.train(X, y, feature_names=feature_names)

    _state['nb_model'] = model
    _state['nb_target'] = target_label

    prior_info = {}
    for c in model.classes:
        prior_info[str(c)] = {
            'count': model.class_count[c],
            'log_prior': round(model.class_log_prior[c], 6),
            'prior': round(math.exp(model.class_log_prior[c]), 6),
        }

    top_features = model.get_top_features_per_class(top_n=10)
    top_features_str = {str(k): v for k, v in top_features.items()}

    # Data hasil train yang akan digunakan untuk prediksi
    vocab_likelihoods = {}
    for c in model.classes:
        vocab_likelihoods[str(c)] = {}
        for i in range(model.n_features):
            term = model.feature_names[i] if i < len(model.feature_names) else f"f_{i}"
            vocab_likelihoods[str(c)][term] = {
                'log_prob': round(model.feature_log_prob[c][i], 6),
                'prob': round(math.exp(model.feature_log_prob[c][i]), 8)
            }

    data_for_prediction = {
        'class_log_priors': prior_info,
        'vocabulary_feature_probabilities': vocab_likelihoods,
        'formula': 'score(c) = log_prior(c) + sum(tf_idf(term) * log_prob(term|c))'
    }

    return jsonify({
        'success': True,
        'message': 'Model Naive Bayes berhasil dilatih',
        'training_info': train_info,
        'prior': prior_info,
        'top_features': top_features_str,
        'data_for_prediction': data_for_prediction,
    })

@nb_bp.route('/api/nb/predict', methods=['POST'])
def predict_nb():
    if not _state['nb_model']:
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

    prediction = _state['nb_model'].predict([tfidf_vector])[0]
    proba = _state['nb_model'].predict_proba([tfidf_vector])[0]

    target = _state.get('nb_target', 'Target')
    label = f"{target} Terdeteksi" if prediction == 1 else f"Bukan {target}"

    proba_str = {str(k): round(v, 6) for k, v in proba.items()}

    return jsonify({
        'success': True,
        'normalized': normalized_text,
        'prediction': label,
        'code': int(prediction),
        'probabilities': proba_str,
    })
