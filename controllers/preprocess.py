from flask import Blueprint, request, jsonify
from core.state import _state
from modules.preprocessing import full_preprocessing_pipeline

preprocess_bp = Blueprint('preprocess', __name__)

@preprocess_bp.route('/api/preprocess/single', methods=['POST'])
def preprocess_single():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'success': False, 'error': 'Field "text" diperlukan'}), 400

    text         = data['text']
    verbose      = data.get('verbose', True)
    use_stemming = data.get('use_stemming', False)
    use_stopword = data.get('use_stopword', False)

    try:
        result = full_preprocessing_pipeline(
            text, _state['lexicon_merged'], verbose=verbose,
            use_stemming=use_stemming, use_stopword=use_stopword
        )
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@preprocess_bp.route('/api/preprocess/batch', methods=['POST'])
def preprocess_batch():
    if not _state['dataset']:
        return jsonify({'success': False,
                        'error': 'Belum ada dataset. Upload dataset terlebih dahulu.'}), 400

    data = request.get_json() or {}
    use_stemming = data.get('use_stemming', False)
    use_stopword = data.get('use_stopword', False)

    results = []
    preprocessed_corpus = []

    for item in _state['dataset']:
        processed = full_preprocessing_pipeline(
            item.get('text', ''),
            _state['lexicon_merged'],
            verbose=False,
            use_stemming=use_stemming,
            use_stopword=use_stopword
        )
        # Sertakan KEDUA format label untuk konsistensi:
        #   - 'label'  : string display untuk UI (e.g. "label: 1")
        #   - 'labels' : dict asli untuk SVM training (e.g. {'label': '1'})
        processed['label'] = item.get('label', '')
        processed['labels'] = item.get('labels', {})
        results.append(processed)
        preprocessed_corpus.append(processed['normalized'])

    # Simpan corpus yang sudah dipreprocess untuk digunakan TF-IDF
    _state['preprocessed_corpus'] = preprocessed_corpus
    _state['tfidf_result'] = None  # Reset TF-IDF karena corpus baru

    total = len(results)
    total_replacements = sum(r['stats']['replacements_made'] for r in results)

    return jsonify({
        'success': True,
        'total'  : total,
        'aggregate_stats': {
            'total_texts'       : total,
            'total_replacements': total_replacements,
            'avg_replacements'  : round(total_replacements / total, 2) if total else 0,
        },
        'results'          : results[:50],
        'corpus_ready'     : True,
    })
