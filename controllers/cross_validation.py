"""
Controller: K-Fold Cross Validation API Endpoint
"""

from flask import Blueprint, request, jsonify
from core.state import _state
from modules.cross_validation import run_cross_validation, format_cv_report

cv_bp = Blueprint('cross_validation', __name__)


@cv_bp.route('/api/cv/run', methods=['POST'])
def run_cv():
    """
    Menjalankan K-Fold Cross Validation pada dataset yang sudah di-preprocess.

    Request Body (JSON):
        k            : int   — jumlah fold (default: 5)
        seed         : int   — seed untuk reprodusibilitas (default: 42)
        target_label : str   — nama kolom label (default: 'label')
        svm_params   : Dict  — parameter SVM (opsional)
        nb_params    : Dict  — parameter NB (opsional)
        rf_params    : Dict  — parameter RF (opsional)
        tfidf_params : Dict  — parameter TF-IDF (opsional)
    """
    # Validasi: dataset harus sudah ada dan sudah dipreprocess
    if not _state.get('dataset'):
        return jsonify({
            'success': False,
            'error': 'Dataset belum dimuat! Upload dataset terlebih dahulu.'
        }), 400

    if not _state.get('preprocessed_corpus'):
        return jsonify({
            'success': False,
            'error': 'Dataset belum dipreprocess! Jalankan preprocessing terlebih dahulu.'
        }), 400

    data = request.json or {}
    k = data.get('k', 5)
    seed = data.get('seed', 42)
    target_label = data.get('target_label', 'label')

    # Validasi K
    if not isinstance(k, int) or k < 2:
        return jsonify({
            'success': False,
            'error': 'Nilai K harus bilangan bulat >= 2.'
        }), 400

    if k > len(_state['dataset']):
        return jsonify({
            'success': False,
            'error': f'K ({k}) tidak boleh lebih besar dari jumlah data ({len(_state["dataset"])}).'
        }), 400

    # Siapkan dataset untuk cross validation
    # Setiap item harus memiliki 'normalized' (teks) dan 'label' (kelas)
    cv_dataset = []
    for i, item in enumerate(_state['dataset']):
        text = ''
        if i < len(_state['preprocessed_corpus']):
            text = _state['preprocessed_corpus'][i]

        # Ambil label dari dataset
        try:
            label_val = int(item.get('labels', {}).get(target_label, 0))
        except (ValueError, TypeError):
            label_val = 0

        cv_dataset.append({
            'normalized': text,
            'label': label_val,
        })

    # Parameter opsional untuk model
    svm_params = data.get('svm_params', None)
    nb_params = data.get('nb_params', None)
    rf_params = data.get('rf_params', None)
    tfidf_params = data.get('tfidf_params', None)

    try:
        result = run_cross_validation(
            dataset=cv_dataset,
            k=k,
            seed=seed,
            text_col='normalized',
            label_col='label',
            svm_params=svm_params,
            nb_params=nb_params,
            rf_params=rf_params,
            tfidf_params=tfidf_params,
        )

        # Generate text report
        report_text = format_cv_report(result)

        return jsonify({
            'success': True,
            'k': k,
            'seed': seed,
            'total_samples': len(cv_dataset),
            'summary_table': result['summary_table'],
            'fold_results': result['fold_results'],
            'average_metrics': result['average_metrics'],
            'std_metrics': result['std_metrics'],
            'report_text': report_text,
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error saat menjalankan cross validation: {str(e)}'
        }), 500
