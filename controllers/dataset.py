import os
from flask import Blueprint, request, jsonify
from core.state import _state, UPLOAD_FOLDER
from modules.preprocessing import read_dataset

dataset_bp = Blueprint('dataset', __name__)

@dataset_bp.route('/api/dataset/upload', methods=['POST'])
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

@dataset_bp.route('/api/dataset/info', methods=['GET'])
def dataset_info():
    if not _state['dataset']:
        return jsonify({'success': False, 'error': 'Dataset kosong'})
    # Get keys from the first row's 'labels' dictionary
    label_cols = list(_state['dataset'][0].get('labels', {}).keys())
    return jsonify({
        'success': True,
        'label_cols': label_cols
    })
