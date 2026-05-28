import tempfile
from modules.preprocessing import BUILTIN_INTERNAL_LEXICON, BUILTIN_EXTERNAL_LEXICON, merge_lexicons

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
