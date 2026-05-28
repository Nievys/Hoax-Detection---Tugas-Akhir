import math
from flask import Blueprint, request, jsonify
from core.state import _state
from modules.preprocessing import full_preprocessing_pipeline
from modules.tfidf import (
    compute_tf, compute_tfidf_vector, fit_transform, transform,
    get_top_features, get_idf_ranking, matrix_to_preview
)

tfidf_bp = Blueprint('tfidf', __name__)

@tfidf_bp.route('/api/tfidf/fit', methods=['POST'])
def tfidf_fit():
    data = request.get_json() or {}

    if not _state['preprocessed_corpus']:
        if not _state['dataset']:
            return jsonify({
                'success': False,
                'error'  : 'Belum ada dataset. Upload dataset dan jalankan preprocessing terlebih dahulu.'
            }), 400

        corpus = []
        for item in _state['dataset']:
            result = full_preprocessing_pipeline(
                item.get('text', ''), _state['lexicon_merged'], verbose=False
            )
            corpus.append(result['normalized'])
        _state['preprocessed_corpus'] = corpus

    corpus = _state['preprocessed_corpus']

    min_df        = int(data.get('min_df', 1))
    max_df_ratio  = float(data.get('max_df_ratio', 1.0))
    max_features  = data.get('max_features', None)
    if max_features:
        max_features = int(max_features)
    smooth_idf    = bool(data.get('smooth_idf', True))
    normalize     = bool(data.get('normalize', True))

    try:
        result = fit_transform(
            corpus,
            min_df=min_df,
            max_df_ratio=max_df_ratio,
            max_features=max_features,
            smooth_idf=smooth_idf,
            normalize=normalize
        )

        _state['tfidf_result'] = result

        preview = matrix_to_preview(
            result['matrix'],
            result['feature_names'],
            max_rows=8,
            max_cols=12
        )

        top_idf = get_idf_ranking(result, top_n=15)

        return jsonify({
            'success'      : True,
            'stats'        : result['stats'],
            'preview'      : preview,
            'top_idf_terms': top_idf,
            'sample_vocab' : dict(list(result['vocabulary'].items())[:20]),
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@tfidf_bp.route('/api/tfidf/status', methods=['GET'])
def tfidf_status():
    corpus_ready = len(_state['preprocessed_corpus']) > 0
    tfidf_ready  = _state['tfidf_result'] is not None

    stats = {}
    if tfidf_ready:
        stats = _state['tfidf_result']['stats']

    return jsonify({
        'success'      : True,
        'corpus_ready' : corpus_ready,
        'corpus_size'  : len(_state['preprocessed_corpus']),
        'tfidf_ready'  : tfidf_ready,
        'stats'        : stats,
    })

@tfidf_bp.route('/api/tfidf/document/<int:doc_index>', methods=['GET'])
def tfidf_document(doc_index):
    if not _state['tfidf_result']:
        return jsonify({'success': False, 'error': 'TF-IDF belum dihitung'}), 400

    result  = _state['tfidf_result']
    top_n   = int(request.args.get('top_n', 10))

    if doc_index >= len(result['matrix']):
        return jsonify({'success': False, 'error': 'Indeks dokumen di luar range'}), 400

    top_features  = get_top_features(result, doc_index, top_n=top_n)
    doc_vector    = result['matrix'][doc_index]

    original_text = ''
    label = ''
    if doc_index < len(_state['dataset']):
        original_text = _state['dataset'][doc_index].get('text', '')
        label         = _state['dataset'][doc_index].get('label', '')

    normalized_text = ''
    if doc_index < len(_state['preprocessed_corpus']):
        normalized_text = _state['preprocessed_corpus'][doc_index]

    l2_norm = math.sqrt(sum(x**2 for x in doc_vector))

    return jsonify({
        'success'        : True,
        'doc_index'      : doc_index,
        'original_text'  : original_text,
        'normalized_text': normalized_text,
        'label'          : label,
        'top_features'   : top_features,
        'vector_length'  : len(doc_vector),
        'l2_norm'        : round(l2_norm, 6),
        'non_zero_count' : sum(1 for x in doc_vector if x > 0),
    })

@tfidf_bp.route('/api/tfidf/vocabulary', methods=['GET'])
def tfidf_vocabulary():
    if not _state['tfidf_result']:
        return jsonify({'success': False, 'error': 'TF-IDF belum dihitung'}), 400

    page     = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    search   = request.args.get('search', '').lower()
    sort_by  = request.args.get('sort', 'alpha')

    result       = _state['tfidf_result']
    feature_names = result['feature_names']
    idf_vector   = result['idf_vector']
    df           = result.get('document_frequency', {})

    items = [
        {
            'term' : feature_names[i],
            'index': i,
            'idf'  : round(idf_vector[i], 6),
            'df'   : df.get(feature_names[i], 0),
        }
        for i in range(len(feature_names))
    ]

    if search:
        items = [x for x in items if search in x['term']]

    if sort_by == 'idf_desc':
        items.sort(key=lambda x: x['idf'], reverse=True)
    elif sort_by == 'idf_asc':
        items.sort(key=lambda x: x['idf'])
    elif sort_by == 'df':
        items.sort(key=lambda x: x['df'], reverse=True)
    else:
        items.sort(key=lambda x: x['term'])

    total = len(items)
    start = (page - 1) * per_page
    page_items = items[start:start + per_page]

    return jsonify({
        'success' : True,
        'total'   : total,
        'page'    : page,
        'per_page': per_page,
        'items'   : page_items,
    })

@tfidf_bp.route('/api/tfidf/single', methods=['POST'])
def tfidf_single():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'success': False, 'error': 'Field "text" diperlukan'}), 400

    raw_text = data['text']

    pp_result = full_preprocessing_pipeline(
        raw_text, _state['lexicon_merged'], verbose=False
    )
    normalized = pp_result['normalized']

    if _state['tfidf_result']:
        result   = _state['tfidf_result']
        vocab    = result['vocabulary']
        idf_vec  = result['idf_vector']
        feat_names = result['feature_names']

        tf_vec    = compute_tf(normalized, vocab)
        tfidf_vec = compute_tfidf_vector(tf_vec, idf_vec, normalize=True)

        top_terms = sorted(
            [{'term': feat_names[i], 'tf': round(tf_vec[i], 6),
              'idf': round(idf_vec[i], 6), 'tfidf': round(tfidf_vec[i], 6)}
             for i in range(len(feat_names)) if tfidf_vec[i] > 0],
            key=lambda x: x['tfidf'], reverse=True
        )[:15]

        return jsonify({
            'success'        : True,
            'raw_text'       : raw_text,
            'normalized_text': normalized,
            'used_existing_vocab': True,
            'vocab_size'     : len(vocab),
            'top_terms'      : top_terms,
            'non_zero_count' : sum(1 for x in tfidf_vec if x > 0),
        })
    else:
        standalone = fit_transform([normalized], min_df=1, smooth_idf=True, normalize=True)

        top_terms = sorted(
            [{'term': standalone['feature_names'][i],
              'tf'  : round(compute_tf(normalized, standalone['vocabulary'])[i], 6),
              'idf' : round(standalone['idf_vector'][i], 6),
              'tfidf': round(standalone['matrix'][0][i], 6)}
             for i in range(len(standalone['feature_names']))
             if standalone['matrix'][0][i] > 0],
            key=lambda x: x['tfidf'], reverse=True
        )[:15]

        return jsonify({
            'success'            : True,
            'raw_text'           : raw_text,
            'normalized_text'    : normalized,
            'used_existing_vocab': False,
            'vocab_size'         : len(standalone['vocabulary']),
            'top_terms'          : top_terms,
            'non_zero_count'     : sum(1 for x in standalone['matrix'][0] if x > 0),
        })

@tfidf_bp.route('/api/tfidf/matrix/export', methods=['GET'])
def export_matrix():
    if not _state['tfidf_result']:
        return jsonify({'success': False, 'error': 'TF-IDF belum dihitung'}), 400

    result = _state['tfidf_result']
    labels = [item.get('labels', {}) for item in _state['dataset']]

    return jsonify({
        'success'      : True,
        'matrix'       : result['matrix'],
        'feature_names': result['feature_names'],
        'labels'       : labels,
        'stats'        : result['stats'],
    })
