"""
=============================================================================
BACKEND: Flask Web Server
  Modul 1 (Preprocessing) + Modul 2 (TF-IDF)
  Modul 3 (SVM) + Modul 4 (Naive Bayes) + Modul 5 (Random Forest)
  Modul 6 (Ensemble Majority Voting)
=============================================================================
"""

import os
from flask import Flask

# Import controllers (Blueprints)
from controllers.lexicon import lexicon_bp
from controllers.dataset import dataset_bp
from controllers.preprocess import preprocess_bp
from controllers.tfidf import tfidf_bp
from controllers.svm import svm_bp
from controllers.nb import nb_bp
from controllers.rf import rf_bp
from controllers.ensemble import ensemble_bp

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# =============================================================================
# ROUTES — STATIC
# =============================================================================

@app.route('/')
def index():
    return app.send_static_file('index.html')

# =============================================================================
# REGISTER BLUEPRINTS
# =============================================================================

app.register_blueprint(lexicon_bp)
app.register_blueprint(dataset_bp)
app.register_blueprint(preprocess_bp)
app.register_blueprint(tfidf_bp)
app.register_blueprint(svm_bp)
app.register_blueprint(nb_bp)
app.register_blueprint(rf_bp)
app.register_blueprint(ensemble_bp)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
