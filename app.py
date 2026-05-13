"""
=============================================================================
BACKEND: Flask Web Server
  Modul 1 (Preprocessing) + Modul 2 (TF-IDF)
  Modul 3 (SVM) + Modul 4 (Naive Bayes) + Modul 5 (Random Forest)
  Modul 6 (Ensemble Majority Voting)
=============================================================================
"""

import os
import json
import tempfile
from flask import Flask, request, jsonify, send_from_directory

# Import modul preprocessing (Modul 1)
from modules.preprocessing import (
    read_csv_file, read_excel_file, read_dataset,
    read_lexicon_file, merge_lexicons,
    cleansing, normalize_text, full_preprocessing_pipeline,
    BUILTIN_INTERNAL_LEXICON, BUILTIN_EXTERNAL_LEXICON
)

# Import modul TF-IDF (Modul 2)
from modules.tfidf import (
    build_vocabulary, compute_tf, compute_idf,
    compute_tfidf_vector, fit_transform, transform,
    get_top_features, get_idf_ranking,
    compute_cosine_similarity, matrix_to_preview
)

from modules.svm import SVMScratch

# Import modul Naive Bayes (Modul 4)
from modules.naive_bayes import MultinomialNBScratch

# Import modul Random Forest (Modul 5)
from modules.random_forest import RandomForestScratch

# Import modul Ensemble (Modul 6)
from modules.ensemble import (
    ensemble_predict, compute_confusion_matrix,
    evaluate_model, compare_models
)

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# ─── State global ─────────────────────────────────────────────────────────────
_state = {
    'lexicon_internal': dict(BUILTIN_INTERNAL_LEXICON),
    'lexicon_external': dict(BUILTIN_EXTERNAL_LEXICON),
    'lexicon_merged'  : {},
    'merge_stats'     : {},
    'dataset'         : [],
    'preprocessed_corpus': [],  # Corpus setelah preprocessing (untuk TF-IDF)
    'tfidf_result'    : None,   # Output fit_transform()
    'svm_model': None,
    'classification_results': None,
    'svm_target': None,
    'nb_model': None,
    'nb_target': None,
    'rf_model': None,
    'rf_target': None,
}

# Inisialisasi kamus gabungan
_state['lexicon_merged'], _state['merge_stats'] = merge_lexicons(
    _state['lexicon_internal'],
    _state['lexicon_external'],
    conflict_strategy='internal_priority'
)

UPLOAD_FOLDER = tempfile.gettempdir()


# =============================================================================
# ROUTES — STATIC
# =============================================================================

@app.route('/')
def index():
    return app.send_static_file('index.html')


# =============================================================================
# ROUTES — MODUL 1: LEKSIKON & PREPROCESSING
# =============================================================================

@app.route('/api/lexicon/status', methods=['GET'])
def lexicon_status():
    stats = _state['merge_stats']
    return jsonify({
        'success'         : True,
        'internal_count'  : stats.get('total_internal', len(_state['lexicon_internal'])),
        'external_count'  : stats.get('total_external', len(_state['lexicon_external'])),
        'merged_count'    : stats.get('total_merged', len(_state['lexicon_merged'])),
        'conflict_count'  : stats.get('conflict_count', 0),
        'duplicate_count' : stats.get('true_duplicates', 0),
        'new_from_external': stats.get('new_from_external', 0),
        'strategy'        : stats.get('strategy_used', 'internal_priority'),
        'conflicts'       : stats.get('conflicts', {}),
        'using_builtin'   : True,
    })


@app.route('/api/lexicon/upload', methods=['POST'])
def upload_lexicon():
    lex_type = request.form.get('type', 'internal')
    strategy = request.form.get('strategy', 'internal_priority')

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Tidak ada file yang diupload'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'success': False, 'error': 'Nama file kosong'}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ('.csv', '.xlsx', '.xls'):
        return jsonify({'success': False,
                        'error': 'Format tidak didukung. Gunakan CSV atau Excel.'}), 400

    tmp_path = os.path.join(UPLOAD_FOLDER, f'lexicon_{lex_type}{ext}')
    f.save(tmp_path)

    try:
        lexicon_data = read_lexicon_file(tmp_path)
        if not lexicon_data:
            return jsonify({'success': False,
                            'error': 'Kamus kosong atau format kolom tidak dikenali.'}), 400

        if lex_type == 'internal':
            _state['lexicon_internal'] = lexicon_data
        else:
            _state['lexicon_external'] = lexicon_data

        _state['lexicon_merged'], _state['merge_stats'] = merge_lexicons(
            _state['lexicon_internal'],
            _state['lexicon_external'],
            conflict_strategy=strategy
        )

        return jsonify({
            'success'     : True,
            'count'       : len(lexicon_data),
            'merged_count': len(_state['lexicon_merged']),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route('/api/dataset/upload', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Tidak ada file'}), 400

    f = request.files['file']
    text_col  = request.form.get('text_col', 'text')
    label_cols_str = request.form.get('label_col', 'label')
    label_cols = [c.strip() for c in label_cols_str.split(',')]

    ext = os.path.splitext(f.filename)[1].lower()
    tmp_path = os.path.join(UPLOAD_FOLDER, f'dataset{ext}')
    f.save(tmp_path)

    try:
        dataset = read_dataset(tmp_path, text_col=text_col, label_cols=label_cols)
        if not dataset:
            return jsonify({'success': False,
                            'error': 'Dataset kosong atau kolom tidak ditemukan.'}), 400

        _state['dataset'] = dataset
        _state['preprocessed_corpus'] = []  # Reset corpus saat dataset baru diupload
        _state['tfidf_result'] = None

        return jsonify({
            'success': True,
            'total'  : len(dataset),
            'sample' : dataset[:5],
            'columns': {'text': text_col, 'label_cols': label_cols},
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.route('/api/dataset/info', methods=['GET'])
def dataset_info():
    if not _state['dataset']:
        return jsonify({'success': False, 'error': 'Dataset kosong'})
    # Get keys from the first row's 'labels' dictionary
    label_cols = list(_state['dataset'][0].get('labels', {}).keys())
    return jsonify({
        'success': True,
        'label_cols': label_cols
    })

@app.route('/api/preprocess/single', methods=['POST'])
def preprocess_single():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'success': False, 'error': 'Field "text" diperlukan'}), 400

    text    = data['text']
    verbose = data.get('verbose', True)

    try:
        result = full_preprocessing_pipeline(
            text, _state['lexicon_merged'], verbose=verbose
        )
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/preprocess/batch', methods=['POST'])
def preprocess_batch():
    if not _state['dataset']:
        return jsonify({'success': False,
                        'error': 'Belum ada dataset. Upload dataset terlebih dahulu.'}), 400

    results = []
    preprocessed_corpus = []

    for item in _state['dataset']:
        processed = full_preprocessing_pipeline(
            item.get('text', ''),
            _state['lexicon_merged'],
            verbose=False
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


@app.route('/api/lexicon/entries', methods=['GET'])
def get_lexicon_entries():
    page     = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    search   = request.args.get('search', '').lower()

    items = list(_state['lexicon_merged'].items())

    if search:
        items = [(k, v) for k, v in items
                 if search in k.lower() or search in v.lower()]

    total = len(items)
    start = (page - 1) * per_page
    page_items = items[start:start + per_page]

    return jsonify({
        'success' : True,
        'total'   : total,
        'page'    : page,
        'per_page': per_page,
        'entries' : [{'slang': k, 'formal': v} for k, v in page_items]
    })


@app.route('/api/merge', methods=['POST'])
def remerge():
    data     = request.get_json() or {}
    strategy = data.get('strategy', 'internal_priority')

    _state['lexicon_merged'], _state['merge_stats'] = merge_lexicons(
        _state['lexicon_internal'],
        _state['lexicon_external'],
        conflict_strategy=strategy
    )

    return jsonify({
        'success'       : True,
        'merged_count'  : len(_state['lexicon_merged']),
        'conflict_count': _state['merge_stats']['conflict_count'],
        'strategy'      : strategy,
    })


# =============================================================================
# ROUTES — MODUL 2: TF-IDF FEATURE EXTRACTION
# =============================================================================

@app.route('/api/tfidf/fit', methods=['POST'])
def tfidf_fit():
    """
    Fit dan transform seluruh corpus → matriks TF-IDF.
    Corpus diambil dari hasil batch preprocessing.
    """
    data = request.get_json() or {}

    # Cek ketersediaan corpus
    if not _state['preprocessed_corpus']:
        # Jika belum ada corpus dari batch, coba preprocess dataset on-the-fly
        if not _state['dataset']:
            return jsonify({
                'success': False,
                'error'  : 'Belum ada dataset. Upload dataset dan jalankan preprocessing terlebih dahulu.'
            }), 400

        # Auto-preprocess
        corpus = []
        for item in _state['dataset']:
            result = full_preprocessing_pipeline(
                item.get('text', ''), _state['lexicon_merged'], verbose=False
            )
            corpus.append(result['normalized'])
        _state['preprocessed_corpus'] = corpus

    corpus = _state['preprocessed_corpus']

    # Ambil parameter konfigurasi dari request
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

        # Simpan hasil untuk digunakan endpoint lain
        _state['tfidf_result'] = result

        # Buat preview matriks (subset kecil untuk UI)
        preview = matrix_to_preview(
            result['matrix'],
            result['feature_names'],
            max_rows=8,
            max_cols=12
        )

        # Top IDF terms
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


@app.route('/api/tfidf/status', methods=['GET'])
def tfidf_status():
    """Status modul TF-IDF: apakah sudah di-fit."""
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


@app.route('/api/tfidf/document/<int:doc_index>', methods=['GET'])
def tfidf_document(doc_index):
    """Analisis TF-IDF detail untuk dokumen tertentu."""
    if not _state['tfidf_result']:
        return jsonify({'success': False, 'error': 'TF-IDF belum dihitung'}), 400

    result  = _state['tfidf_result']
    top_n   = int(request.args.get('top_n', 10))

    if doc_index >= len(result['matrix']):
        return jsonify({'success': False, 'error': 'Indeks dokumen di luar range'}), 400

    top_features  = get_top_features(result, doc_index, top_n=top_n)
    doc_vector    = result['matrix'][doc_index]

    # Info dokumen asli
    original_text = ''
    label = ''
    if doc_index < len(_state['dataset']):
        original_text = _state['dataset'][doc_index].get('text', '')
        label         = _state['dataset'][doc_index].get('label', '')

    normalized_text = ''
    if doc_index < len(_state['preprocessed_corpus']):
        normalized_text = _state['preprocessed_corpus'][doc_index]

    # Hitung L2 norm (untuk verifikasi normalisasi)
    import math
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


@app.route('/api/tfidf/vocabulary', methods=['GET'])
def tfidf_vocabulary():
    """Ambil vocabulary dengan nilai IDF (paginasi)."""
    if not _state['tfidf_result']:
        return jsonify({'success': False, 'error': 'TF-IDF belum dihitung'}), 400

    page     = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    search   = request.args.get('search', '').lower()
    sort_by  = request.args.get('sort', 'alpha')  # 'alpha', 'idf_asc', 'idf_desc', 'df'

    result       = _state['tfidf_result']
    feature_names = result['feature_names']
    idf_vector   = result['idf_vector']
    df           = result.get('document_frequency', {})

    # Buat list item vocabulary
    items = [
        {
            'term' : feature_names[i],
            'index': i,
            'idf'  : round(idf_vector[i], 6),
            'df'   : df.get(feature_names[i], 0),
        }
        for i in range(len(feature_names))
    ]

    # Filter
    if search:
        items = [x for x in items if search in x['term']]

    # Sort
    if sort_by == 'idf_desc':
        items.sort(key=lambda x: x['idf'], reverse=True)
    elif sort_by == 'idf_asc':
        items.sort(key=lambda x: x['idf'])
    elif sort_by == 'df':
        items.sort(key=lambda x: x['df'], reverse=True)
    else:  # alpha
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


@app.route('/api/tfidf/single', methods=['POST'])
def tfidf_single():
    """
    Hitung TF-IDF untuk satu teks menggunakan vocabulary yang sudah di-fit.
    Jika belum di-fit, hitung secara standalone dengan teks ini saja.
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'success': False, 'error': 'Field "text" diperlukan'}), 400

    raw_text = data['text']

    # Preprocessing dulu
    pp_result = full_preprocessing_pipeline(
        raw_text, _state['lexicon_merged'], verbose=False
    )
    normalized = pp_result['normalized']

    if _state['tfidf_result']:
        # Gunakan vocabulary & IDF yang sudah ada
        result   = _state['tfidf_result']
        vocab    = result['vocabulary']
        idf_vec  = result['idf_vector']
        feat_names = result['feature_names']

        tf_vec    = compute_tf(normalized, vocab)
        tfidf_vec = compute_tfidf_vector(tf_vec, idf_vec, normalize=True)

        # Top terms
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
        # Standalone: fit hanya dari teks ini
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


@app.route('/api/tfidf/matrix/export', methods=['GET'])
def export_matrix():
    """Export matriks TF-IDF sebagai JSON (untuk integrasi modul klasifikasi)."""
    if not _state['tfidf_result']:
        return jsonify({'success': False, 'error': 'TF-IDF belum dihitung'}), 400

    result = _state['tfidf_result']
    # Gunakan dict labels asli (bukan display string) untuk integrasi SVM
    labels = [item.get('labels', {}) for item in _state['dataset']]

    return jsonify({
        'success'      : True,
        'matrix'       : result['matrix'],
        'feature_names': result['feature_names'],
        'labels'       : labels,
        'stats'        : result['stats'],
    })


# =============================================================================
# ROUTES — MODUL 3: KLASIFIKASI SVM (SUPPORT VECTOR MACHINE)
# =============================================================================
# Alur Klasifikasi SVM:
#   1. TRAINING: X (matriks TF-IDF N×V) + y (label {-1, +1}) → SMO → model
#   2. PREDIKSI: teks baru → preprocessing → TF-IDF transform → f(x) → sign
#
# Rumus Keputusan:
#   f(x) = Σᵢ (αᵢ · yᵢ · K(xᵢ, x)) + b
#   Kelas = sign(f(x))  → +1 (terdeteksi) atau -1 (tidak terdeteksi)
# =============================================================================

@app.route('/api/svm/train', methods=['POST'])
def train_svm():
    """
    Melatih model SVM menggunakan matriks TF-IDF sebagai fitur input.

    Alur:
      1. Ambil matriks TF-IDF (X) dari state → ukuran (N × V)
      2. Konversi label ke format biner SVM: {+1, -1}
         - Label asli 1 → +1 (kelas positif, misal: hate speech)
         - Label asli 0 → -1 (kelas negatif, misal: bukan hate speech)
      3. Jalankan SMO (Sequential Minimal Optimization) untuk optimasi α
      4. Simpan model (α, b, support vectors) ke state
    """
    if not _state['tfidf_result']:
        return jsonify({'success': False, 'error': 'Hitung TF-IDF terlebih dahulu!'}), 400

    data = request.json
    kernel = data.get('kernel', 'linear')
    C = float(data.get('C', 1.0))
    target_label = data.get('target_label', 'label')

    # ── Persiapkan data X (fitur) dan y (label) ──────────────────────────────
    # X = matriks TF-IDF, setiap baris = vektor fitur dokumen (ukuran V)
    X = _state['tfidf_result']['matrix']

    # Konversi label ke format SVM: {+1, -1}
    # SVM memerlukan label +1 dan -1 (bukan 0 dan 1)
    # Karena fungsi keputusan f(x) menghasilkan nilai riil,
    # dan kelas ditentukan oleh sign(f(x)).
    try:
        y = [1 if int(item['labels'].get(target_label, 0)) == 1 else -1
             for item in _state['dataset']]
    except (ValueError, TypeError):
        return jsonify({'success': False,
                        'error': f'Nilai label "{target_label}" harus berupa angka (0 atau 1)!'}), 400

    # ── Validasi data ────────────────────────────────────────────────────────
    if len(X) != len(y):
        return jsonify({'success': False,
                        'error': f'Jumlah fitur ({len(X)}) != jumlah label ({len(y)})'}), 400

    if len(set(y)) < 2:
        return jsonify({'success': False,
                        'error': 'Dataset harus memiliki minimal 2 kelas (+1 dan -1). '
                                 'Pastikan kolom label memiliki nilai 0 dan 1.'}), 400

    # ── Inisialisasi dan Training SMO ────────────────────────────────────────
    model = SVMScratch(kernel=kernel, C=C, max_passes=10)
    train_status = model.train(X, y)

    # Simpan model dan target label ke state
    _state['svm_model'] = model
    _state['svm_target'] = target_label

    # Hitung distribusi kelas untuk informasi
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


@app.route('/api/svm/predict', methods=['POST'])
def predict_text():
    """
    Prediksi kelas untuk teks baru menggunakan model SVM yang sudah dilatih.

    Alur Prediksi (3 tahap):
      1. PREPROCESSING : teks mentah → cleansing → normalisasi slang
      2. TF-IDF TRANSFORM : teks ternormalisasi → vektor TF-IDF
         menggunakan vocabulary & IDF dari training (BUKAN fit ulang)
      3. SVM PREDICT : vektor TF-IDF → f(x) → sign → kelas

    Rumus:
      f(x_baru) = Σᵢ (αᵢ · yᵢ · K(xᵢ, x_baru)) + b
      Kelas = +1 jika f(x) ≥ 0, else -1
    """
    if not _state['svm_model']:
        return jsonify({'success': False, 'error': 'Model belum dilatih!'}), 400

    raw_text = request.json.get('text', '')
    if not raw_text.strip():
        return jsonify({'success': False, 'error': 'Teks tidak boleh kosong!'}), 400

    # ── TAHAP 1: Preprocessing ───────────────────────────────────────────────
    # Pipeline: raw → lowercase → hapus URL/mention/hashtag/angka/simbol → normalisasi slang
    pp_result = full_preprocessing_pipeline(raw_text, _state['lexicon_merged'], verbose=False)
    normalized_text = pp_result['normalized']  # Ambil STRING dari dict hasil pipeline

    # ── TAHAP 2: Transform ke TF-IDF ────────────────────────────────────────
    # Gunakan vocabulary & IDF yang SUDAH dihitung dari data training.
    # Token baru yang tidak ada di vocabulary akan diabaikan (OOV).
    # Rumus: TF-IDF(t, d_baru) = TF(t, d_baru) × IDF(t)
    tfidf_vector = transform(
        [normalized_text],
        _state['tfidf_result']['vocabulary'],
        _state['tfidf_result']['idf_vector']
    )[0]  # Ambil vektor dokumen pertama (indeks 0)

    # ── TAHAP 3: Prediksi SVM ────────────────────────────────────────────────
    # f(x) = Σᵢ (αᵢ · yᵢ · K(xᵢ, x)) + b
    # sign(f(x)) = +1 → kelas positif (terdeteksi)
    # sign(f(x)) = -1 → kelas negatif (tidak terdeteksi)
    prediction_code = _state['svm_model'].predict([tfidf_vector])[0]

    target = _state.get('svm_target', 'Target')
    label = f"{target} Terdeteksi" if prediction_code == 1 else f"Bukan {target}"

    return jsonify({
        'success': True,
        'normalized': normalized_text,
        'prediction': label,
        'code': int(prediction_code)
    })


# =============================================================================
# ROUTES — MODUL 4: KLASIFIKASI MULTINOMIAL NAIVE BAYES
# =============================================================================
# Alur Klasifikasi Naive Bayes:
#   1. TRAINING: Hitung Prior P(c) dan Likelihood P(tᵢ|c) dari data training
#   2. PREDIKSI: teks baru → preprocessing → TF-IDF → argmax log-posterior
#
# Rumus Keputusan (log-space):
#   ĉ = argmax_c [ log P(c) + Σᵢ wᵢ · log P(tᵢ|c) ]
# =============================================================================

@app.route('/api/nb/train', methods=['POST'])
def train_nb():
    """
    Melatih model Multinomial Naive Bayes dari matriks TF-IDF.

    Alur:
      1. Ambil matriks TF-IDF (X) → fitur input
      2. Ambil label dari dataset → konversi ke integer
      3. Hitung Prior P(c) = Nₖ/N untuk setiap kelas
      4. Hitung Likelihood P(tᵢ|c) dengan Laplace Smoothing
    """
    if not _state['tfidf_result']:
        return jsonify({'success': False, 'error': 'Hitung TF-IDF terlebih dahulu!'}), 400

    data = request.json or {}
    alpha = float(data.get('alpha', 1.0))
    target_label = data.get('target_label', 'label')

    # ── Persiapkan data X (fitur) dan y (label) ──────────────────────────
    X = _state['tfidf_result']['matrix']
    feature_names = _state['tfidf_result']['feature_names']

    # Konversi label ke integer
    try:
        y = [int(item['labels'].get(target_label, 0))
             for item in _state['dataset']]
    except (ValueError, TypeError):
        return jsonify({'success': False,
                        'error': f'Nilai label "{target_label}" harus berupa angka!'}), 400

    # ── Validasi ─────────────────────────────────────────────────────────
    if len(X) != len(y):
        return jsonify({'success': False,
                        'error': f'Jumlah fitur ({len(X)}) != jumlah label ({len(y)})'}), 400

    if len(set(y)) < 2:
        return jsonify({'success': False,
                        'error': 'Dataset harus memiliki minimal 2 kelas.'}), 400

    # ── Training ─────────────────────────────────────────────────────────
    model = MultinomialNBScratch(alpha=alpha)
    train_info = model.train(X, y, feature_names=feature_names)

    _state['nb_model'] = model
    _state['nb_target'] = target_label

    # ── Prior probabilities untuk response ───────────────────────────────
    import math
    prior_info = {}
    for c in model.classes:
        prior_info[str(c)] = {
            'count': model.class_count[c],
            'log_prior': round(model.class_log_prior[c], 6),
            'prior': round(math.exp(model.class_log_prior[c]), 6),
        }

    # ── Top features per kelas ───────────────────────────────────────────
    top_features = model.get_top_features_per_class(top_n=10)
    # Konversi key int → str untuk JSON
    top_features_str = {str(k): v for k, v in top_features.items()}

    return jsonify({
        'success': True,
        'message': 'Model Naive Bayes berhasil dilatih',
        'training_info': train_info,
        'prior': prior_info,
        'top_features': top_features_str,
    })


@app.route('/api/nb/predict', methods=['POST'])
def predict_nb():
    """
    Prediksi kelas untuk teks baru menggunakan Multinomial Naive Bayes.

    Alur Prediksi (3 tahap):
      1. PREPROCESSING : teks mentah → cleansing → normalisasi slang
      2. TF-IDF TRANSFORM : teks ternormalisasi → vektor TF-IDF
      3. NB PREDICT : vektor → log P(c) + Σ wᵢ·log P(tᵢ|c) → argmax
    """
    if not _state['nb_model']:
        return jsonify({'success': False, 'error': 'Model belum dilatih!'}), 400

    raw_text = request.json.get('text', '')
    if not raw_text.strip():
        return jsonify({'success': False, 'error': 'Teks tidak boleh kosong!'}), 400

    # ── TAHAP 1: Preprocessing ───────────────────────────────────────────
    pp_result = full_preprocessing_pipeline(raw_text, _state['lexicon_merged'], verbose=False)
    normalized_text = pp_result['normalized']

    # ── TAHAP 2: Transform ke TF-IDF ────────────────────────────────────
    tfidf_vector = transform(
        [normalized_text],
        _state['tfidf_result']['vocabulary'],
        _state['tfidf_result']['idf_vector']
    )[0]

    # ── TAHAP 3: Prediksi Naive Bayes ────────────────────────────────────
    prediction = _state['nb_model'].predict([tfidf_vector])[0]
    proba = _state['nb_model'].predict_proba([tfidf_vector])[0]

    target = _state.get('nb_target', 'Target')
    label = f"{target} Terdeteksi" if prediction == 1 else f"Bukan {target}"

    # Konversi proba keys ke string untuk JSON
    proba_str = {str(k): round(v, 6) for k, v in proba.items()}

    return jsonify({
        'success': True,
        'normalized': normalized_text,
        'prediction': label,
        'code': int(prediction),
        'probabilities': proba_str,
    })



# =============================================================================
# ROUTES — MODUL 5: KLASIFIKASI RANDOM FOREST
# =============================================================================
# Alur:
#   1. TRAINING: Bootstrap Sampling + Decision Tree (Gini) × T pohon
#   2. PREDIKSI: teks → preprocessing → TF-IDF → vote semua pohon → majority
# =============================================================================

@app.route('/api/rf/train', methods=['POST'])
def train_rf():
    """
    Melatih Random Forest dari matriks TF-IDF.
    Membangun T pohon, masing-masing dilatih pada bootstrap sample.
    """
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

    # Distribusi kelas
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


@app.route('/api/rf/predict', methods=['POST'])
def predict_rf():
    """
    Prediksi menggunakan Random Forest (majority voting dari T pohon).
    """
    if not _state['rf_model']:
        return jsonify({'success': False, 'error': 'Model belum dilatih!'}), 400

    raw_text = request.json.get('text', '')
    if not raw_text.strip():
        return jsonify({'success': False, 'error': 'Teks tidak boleh kosong!'}), 400

    # Tahap 1: Preprocessing
    pp_result = full_preprocessing_pipeline(raw_text, _state['lexicon_merged'], verbose=False)
    normalized_text = pp_result['normalized']

    # Tahap 2: TF-IDF transform
    tfidf_vector = transform(
        [normalized_text],
        _state['tfidf_result']['vocabulary'],
        _state['tfidf_result']['idf_vector']
    )[0]

    # Tahap 3: Prediksi Random Forest (majority voting)
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



# =============================================================================
# ROUTES — MODUL 6: ENSEMBLE MAJORITY VOTING
# =============================================================================
# Alur:
#   1. EVALUASI: jalankan prediksi SVM + NB + RF pada data training,
#      hitung confusion matrix & metrik untuk setiap model + ensemble
#   2. PREDIKSI: teks baru → preprocessing → TF-IDF → 3 model → majority vote
# =============================================================================

@app.route('/api/ensemble/evaluate', methods=['POST'])
def evaluate_ensemble():
    """
    Evaluasi semua model individu + ensemble pada data training.
    Menghasilkan tabel perbandingan akurasi.
    """
    # Cek apakah semua model sudah dilatih
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

    # Label aktual (ground truth)
    try:
        y_true = [int(item['labels'].get(target_label, 0))
                  for item in _state['dataset']]
    except (ValueError, TypeError):
        return jsonify({'success': False,
                        'error': f'Nilai label "{target_label}" harus berupa angka!'}), 400

    # Jalankan ensemble predict
    models = {
        'SVM': _state['svm_model'],
        'Naive Bayes': _state['nb_model'],
        'Random Forest': _state['rf_model'],
    }
    ens_result = ensemble_predict(models, X)

    # Kumpulkan semua prediksi (individual + ensemble)
    all_preds = dict(ens_result['individual_predictions'])
    all_preds['Ensemble (Voting)'] = ens_result['ensemble_predictions']

    # Bandingkan semua model
    comparison = compare_models(y_true, all_preds, positive_label=1)

    return jsonify({
        'success': True,
        'comparison': comparison,
        'total_samples': len(y_true),
    })


@app.route('/api/ensemble/predict', methods=['POST'])
def predict_ensemble():
    """
    Prediksi teks baru menggunakan Ensemble Majority Voting.
    Menjalankan SVM + NB + RF dan mengambil suara terbanyak.
    """
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

    # Tahap 1: Preprocessing
    pp_result = full_preprocessing_pipeline(raw_text, _state['lexicon_merged'], verbose=False)
    normalized_text = pp_result['normalized']

    # Tahap 2: TF-IDF transform
    tfidf_vector = transform(
        [normalized_text],
        _state['tfidf_result']['vocabulary'],
        _state['tfidf_result']['idf_vector']
    )[0]

    # Tahap 3: Prediksi dari ketiga model
    svm_pred = _state['svm_model'].predict([tfidf_vector])[0]
    svm_pred_normalized = 0 if svm_pred == -1 else 1  # Konversi SVM: -1→0

    nb_pred = _state['nb_model'].predict([tfidf_vector])[0]
    rf_pred = _state['rf_model'].predict([tfidf_vector])[0]

    # Tahap 4: Majority Voting
    votes = {'SVM': svm_pred_normalized, 'Naive Bayes': nb_pred, 'Random Forest': rf_pred}
    vote_counts = {}
    for v in votes.values():
        vote_counts[v] = vote_counts.get(v, 0) + 1
    ensemble_pred = max(vote_counts, key=vote_counts.get)

    target = _state.get('nb_target', 'Target')  # Ambil dari NB target
    label = f"{target} Terdeteksi" if ensemble_pred == 1 else f"Bukan {target}"

    return jsonify({
        'success': True,
        'normalized': normalized_text,
        'prediction': label,
        'code': int(ensemble_pred),
        'votes': votes,
        'vote_counts': {str(k): v for k, v in vote_counts.items()},
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
