import math
import re

# BAGIAN 1: PEMBANGUNAN VOCABULARY
def build_vocabulary(
    corpus: list[str],
    min_df: int = 1,
    max_df_ratio: float = 1.0,
    max_features: int = None
) -> dict:
    if not corpus:
        return {}

    N = len(corpus)  # Total dokumen dalam corpus

    # Hitung Document Frequency per Term
    # document_frequency[t] = |{d ∈ D : t ∈ d}|
    # Menggunakan set per dokumen untuk menghindari penghitungan ganda
    # (jika t muncul 3x di d, tetap df += 1, bukan +3)
    document_frequency = {}  # {term: df_count}

    for doc in corpus:
        # Tokenisasi sederhana: split whitespace (sudah di-preprocess)
        tokens_in_doc = set(doc.split())  # set → hitung kemunculan unik per dok
        for token in tokens_in_doc:
            if token:  # Lewati token kosong
                document_frequency[token] = document_frequency.get(token, 0) + 1

    # Filter Berdasarkan min_df dan max_df_ratio
    # min_df : Hapus kata yang terlalu jarang (noise, typo, kata sangat spesifik)
    # max_df_ratio: Hapus kata yang terlalu umum (tidak informatif secara global)
    max_df_count = int(N * max_df_ratio)  # Konversi ratio → count absolut

    filtered_terms = {
        term for term, df in document_frequency.items()
        if min_df <= df <= max_df_count
    }

    # Batasi Ukuran Vocabulary (opsional: top-N berdasarkan df)
    if max_features and len(filtered_terms) > max_features:
        # Urutkan berdasarkan df (descending) → ambil top max_features
        # Kata dengan df tinggi = lebih representatif corpus
        sorted_by_df = sorted(
            filtered_terms,
            key=lambda t: document_frequency[t],
            reverse=True
        )
        filtered_terms = set(sorted_by_df[:max_features])

    # Buat Mapping Term → Indeks (urutan alfabetis untuk determinisme)
    # Urutan alfabetis memastikan hasil reproducible (tidak acak tiap run)
    sorted_terms = sorted(filtered_terms)  # Urutkan O(|V| log|V|)

    # Pemetaan: { term: integer_index }
    # Indeks ini menentukan posisi kolom dalam matriks TF-IDF
    # Contoh: vocab = {'aku': 0, 'benci': 1, 'cinta': 2, ...}
    vocabulary = {term: idx for idx, term in enumerate(sorted_terms)}

    return vocabulary


def get_document_frequency(corpus: list[str], vocabulary: dict) -> dict:
    df = {term: 0 for term in vocabulary}  # Inisialisasi semua 0

    for doc in corpus:
        # Gunakan set untuk menghitung kemunculan unik per dokumen
        tokens_in_doc = set(doc.split())
        for token in tokens_in_doc:
            if token in df:  # Hanya term yang ada di vocabulary
                df[token] += 1

    return df

# BAGIAN 2: TERM FREQUENCY (TF)
def compute_tf(document: str, vocabulary: dict) -> list[float]:
    V = len(vocabulary)

    # Inisialisasi vektor TF dengan 0.0 untuk setiap term di vocabulary
    tf_vector = [0.0] * V

    # Tokenisasi dokumen: split pada whitespace
    tokens = document.split()
    total_tokens = len(tokens)  # |d| = panjang dokumen

    if total_tokens == 0:
        return tf_vector  # Dokumen kosong → semua TF = 0

    # Hitung Frekuensi Absolut per Token
    # term_count[t] = f(t, d) = berapa kali t muncul di d
    term_count = {}
    for token in tokens:
        if token in vocabulary:  # Hanya token yang ada di vocabulary
            term_count[token] = term_count.get(token, 0) + 1

    # Konversi ke TF Relatif dan Isi Vektor
    # TF(t, d) = f(t, d) / |d|
    for term, count in term_count.items():
        idx = vocabulary[term]  # Dapatkan indeks kolom dari vocabulary
        tf_vector[idx] = count / total_tokens  # TF relatif

    return tf_vector

# BAGIAN 3: INVERSE DOCUMENT FREQUENCY (IDF)
def compute_idf(
    vocabulary: dict,
    document_frequency: dict,
    N: int,
    smooth: bool = True
) -> list[float]:

    V = len(vocabulary)
    idf_vector = [0.0] * V

    for term, idx in vocabulary.items():
        df = document_frequency.get(term, 0)  # Document frequency term ini

        if smooth:
            # Smooth IDF: log((N+1)/(df+1)) + 1
            # Konsisten dengan sklearn.feature_extraction.text.TfidfVectorizer
            # dengan smooth_idf=True (default sklearn)
            idf_vector[idx] = math.log((N + 1) / (df + 1)) + 1
        else:
            # Standard IDF: log(N/df)
            # Pastikan df > 0 untuk menghindari ZeroDivisionError
            # (seharusnya selalu > 0 karena term dari vocabulary)
            if df > 0:
                idf_vector[idx] = math.log(N / df)
            else:
                idf_vector[idx] = 0.0  # Fallback safety

    return idf_vector

# BAGIAN 4: TF-IDF WEIGHTING
def compute_tfidf_vector(
    tf_vector: list[float],
    idf_vector: list[float],
    normalize: bool = True
) -> list[float]:

    V = len(tf_vector)

    # TF-IDF[i] = TF[i] × IDF[i]  untuk setiap dimensi i
    tfidf = [tf_vector[i] * idf_vector[i] for i in range(V)]

    if normalize:
        # Hitung L2 norm: ||v||₂ = √(Σᵢ tfidf[i]²)
        l2_norm = math.sqrt(sum(x ** 2 for x in tfidf))

        if l2_norm > 0:  # Hindari pembagian nol untuk vektor nol
            # Normalisasi: setiap elemen dibagi L2 norm
            tfidf = [x / l2_norm for x in tfidf]
        # Jika l2_norm = 0 → vektor nol, tidak bisa dinormalisasi → biarkan

    return tfidf


# BAGIAN 5: MATRIKS TF-IDF (PIPELINE UTAMA)
def fit_transform(
    corpus: list[str],
    min_df: int = 1,
    max_df_ratio: float = 1.0,
    max_features: int = None,
    smooth_idf: bool = True,
    normalize: bool = True
) -> dict:

    N = len(corpus)
    if N == 0:
        return {
            'matrix': [], 'vocabulary': {}, 'idf_vector': [],
            'feature_names': [], 'stats': {}
        }

    # LANGKAH 1: Bangun Vocabulary
    vocabulary = build_vocabulary(
        corpus, min_df=min_df,
        max_df_ratio=max_df_ratio,
        max_features=max_features
    )
    V = len(vocabulary)  # |V| = ukuran vocabulary

    # feature_names: list kata terurut sesuai indeks kolom
    # Contoh: ['bagus', 'benci', 'cinta'] → term di kolom 0, 1, 2
    feature_names = [''] * V
    for term, idx in vocabulary.items():
        feature_names[idx] = term

    # LANGKAH 2: Hitung Document Frequency
    document_frequency = get_document_frequency(corpus, vocabulary)

    # LANGKAH 3: Hitung IDF (satu kali untuk seluruh corpus)
    idf_vector = compute_idf(
        vocabulary, document_frequency, N, smooth=smooth_idf
    )

    # LANGKAH 4: Bangun Matriks TF-IDF
    # X = [] → akan menjadi matriks (N × V)
    matrix = []

    for doc in corpus:
        # 4a. Hitung TF untuk dokumen ini
        tf_vec = compute_tf(doc, vocabulary)

        # 4b+4c. Hitung TF-IDF dan normalisasi
        tfidf_vec = compute_tfidf_vector(tf_vec, idf_vector, normalize=normalize)

        # 4d. Tambahkan vektor dokumen ke matriks
        matrix.append(tfidf_vec)

    # LANGKAH 5: Kumpulkan Statistik
    # Hitung sparsity: persentase elemen nol dalam matriks
    total_elements = N * V
    zero_elements  = sum(1 for row in matrix for val in row if val == 0.0)
    sparsity = (zero_elements / total_elements * 100) if total_elements > 0 else 0

    stats = {
        'n_documents'     : N,
        'n_features'      : V,
        'total_elements'  : total_elements,
        'zero_elements'   : zero_elements,
        'sparsity_percent': round(sparsity, 2),
        'min_df'          : min_df,
        'max_df_ratio'    : max_df_ratio,
        'smooth_idf'      : smooth_idf,
        'normalized'      : normalize,
    }

    return {
        'matrix'      : matrix,
        'vocabulary'  : vocabulary,
        'idf_vector'  : idf_vector,
        'feature_names': feature_names,
        'document_frequency': document_frequency,
        'stats'       : stats,
    }


def transform(
    corpus: list[str],
    vocabulary: dict,
    idf_vector: list[float],
    normalize: bool = True
) -> list[list[float]]:

    matrix = []
    for doc in corpus:
        tf_vec    = compute_tf(doc, vocabulary)
        tfidf_vec = compute_tfidf_vector(tf_vec, idf_vector, normalize=normalize)
        matrix.append(tfidf_vec)
    return matrix

# BAGIAN 6: UTILITIES — Analisis & Visualisasi
def get_top_features(
    tfidf_result: dict,
    doc_index: int,
    top_n: int = 10
) -> list[dict]:

    matrix       = tfidf_result['matrix']
    feature_names = tfidf_result['feature_names']

    if doc_index >= len(matrix):
        return []

    doc_vector = matrix[doc_index]

    # Pasangkan setiap term dengan nilai TF-IDF-nya
    term_scores = [
        {'term': feature_names[i], 'tfidf': doc_vector[i], 'rank': 0}
        for i in range(len(feature_names))
    ]

    # Urutkan berdasarkan TF-IDF descending → ambil top_n
    term_scores.sort(key=lambda x: x['tfidf'], reverse=True)

    # Beri rank
    for rank, item in enumerate(term_scores[:top_n], start=1):
        item['rank'] = rank

    return term_scores[:top_n]


def get_idf_ranking(
    tfidf_result: dict,
    top_n: int = 20
) -> list[dict]:

    vocabulary  = tfidf_result['vocabulary']
    idf_vector  = tfidf_result['idf_vector']
    df          = tfidf_result.get('document_frequency', {})
    feature_names = tfidf_result['feature_names']

    rankings = [
        {
            'term': feature_names[idx],
            'idf' : round(idf_vector[idx], 6),
            'df'  : df.get(feature_names[idx], 0),
            'rank': 0
        }
        for idx in range(len(feature_names))
    ]

    # Urutkan IDF descending (kata paling unik di atas)
    rankings.sort(key=lambda x: x['idf'], reverse=True)
    for rank, item in enumerate(rankings[:top_n], start=1):
        item['rank'] = rank

    return rankings[:top_n]


def compute_cosine_similarity(
    vec_a: list[float],
    vec_b: list[float]
) -> float:

    if len(vec_a) != len(vec_b):
        raise ValueError("Panjang vektor harus sama")

    # Dot product: A · B = Σᵢ(Aᵢ × Bᵢ)
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    # L2 norm masing-masing vektor
    norm_a = math.sqrt(sum(a ** 2 for a in vec_a))
    norm_b = math.sqrt(sum(b ** 2 for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0  # Salah satu atau keduanya vektor nol

    return dot_product / (norm_a * norm_b)


def matrix_to_preview(
    matrix: list[list[float]],
    feature_names: list[str],
    max_rows: int = 5,
    max_cols: int = 10
) -> dict:

    N = len(matrix)
    V = len(feature_names)

    preview_rows  = min(max_rows, N)
    preview_cols  = min(max_cols, V)

    # Ambil subset matriks
    preview_matrix = [
        [round(matrix[i][j], 6) for j in range(preview_cols)]
        for i in range(preview_rows)
    ]

    return {
        'full_shape'    : [N, V],
        'preview_shape' : [preview_rows, preview_cols],
        'preview_cols'  : feature_names[:preview_cols],
        'preview_matrix': preview_matrix,
        'truncated'     : N > max_rows or V > max_cols,
    }
