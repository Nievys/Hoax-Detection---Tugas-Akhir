import os
from flask import Blueprint, request, jsonify
from core.state import _state, UPLOAD_FOLDER
from modules.preprocessing import read_lexicon_file, merge_lexicons

lexicon_bp = Blueprint('lexicon', __name__)

@lexicon_bp.route('/api/lexicon/status', methods=['GET'])
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

@lexicon_bp.route('/api/lexicon/upload', methods=['POST'])
def upload_lexicon():
    lex_type = request.form.get('type', 'internal')
    strategy = request.form.get('strategy', 'internal_priority')

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Tidak ada file yang diupload'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'success': False, 'error': 'Nama file kosong'}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ('.csv', '.xlsx', '.xls', '.json'):
        return jsonify({'success': False,
                        'error': 'Format tidak didukung. Gunakan CSV, Excel, atau JSON.'}), 400

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

@lexicon_bp.route('/api/lexicon/entries', methods=['GET'])
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

@lexicon_bp.route('/api/merge', methods=['POST'])
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
