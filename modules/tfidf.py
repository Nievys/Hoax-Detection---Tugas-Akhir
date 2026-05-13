"""
=============================================================================
MODUL 2: FEATURE EXTRACTION — TF-IDF
=============================================================================
Judul TA : Analisis Pengaruh Teknik Normalisasi Kata Gaul (Slang) terhadap
           Akurasi Deteksi Ujaran Kebencian Berbahasa Indonesia
=============================================================================
Implementasi MURNI Python. Tidak menggunakan TfidfVectorizer atau library
feature extraction manapun. Setiap rumus dijelaskan secara matematis.
=============================================================================

LANDASAN MATEMATIS
==================

TF (Term Frequency) — Frekuensi Relatif Kata dalam Dokumen:
  TF(t, d) = f(t, d) / |d|
  dimana:
    f(t, d) = jumlah kemunculan term t dalam dokumen d
    |d|     = total token dalam dokumen d
  Interpretasi: Seberapa sering kata t muncul secara proporsional di d.

IDF (Inverse Document Frequency) — Keunikan Kata di Seluruh Corpus:
  IDF(t, D) = log( N / df(t) )
  dimana:
    N      = total jumlah dokumen dalam corpus D
    df(t)  = jumlah dokumen yang mengandung term t (document frequency)
    log    = logaritma natural (math.log) atau log basis 10/2
  Interpretasi: Kata yang muncul di banyak dokumen → IDF rendah (tidak
  informatif). Kata langka → IDF tinggi (sangat informatif).
  Catatan: Digunakan smooth IDF opsional: log((N+1)/(df+1)) + 1 untuk
           menghindari pembagian nol.

TF-IDF Weighting — Bobot Akhir:
  TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
  Interpretasi: Kata yang sering muncul di satu dokumen TAPI jarang di
  dokumen lain mendapat bobot tinggi → fitur diskriminatif.

Cosine Normalization (opsional):
  v_norm = v / ||v||₂   dimana ||v||₂ = √(Σ vᵢ²)
  Menstandarkan panjang vektor agar tidak bias terhadap dokumen panjang.
=============================================================================
"""

import math
import re


# =============================================================================
# BAGIAN 1: PEMBANGUNAN VOCABULARY (KOSA KATA)
# =============================================================================

def build_vocabulary(corpus: list[str],
                     min_df: int = 1,
                     max_df_ratio: float = 1.0,
                     max_features: int = None) -> dict:
    """
    Membangun kosa kata (vocabulary) dari seluruh corpus.

    Vocabulary adalah pemetaan: { kata → indeks_kolom }
    Ini adalah fondasi representasi Bag-of-Words dan TF-IDF.

    Proses Pembangunan:
      1. Iterasi setiap dokumen di corpus → tokenisasi
      2. Hitung document_frequency[t] = jumlah dokumen yang mengandung t
      3. Filter berdasarkan min_df dan max_df_ratio
      4. Urutkan alfabetis untuk konsistensi indeks
      5. Buat mapping: { kata: indeks } → O(|V|) dimana V = vocabulary

    Catatan Indeks:
      Indeks vocab menentukan KOLOM pada matriks TF-IDF.
      vocab['tidak'] = 5 → kolom ke-5 di setiap vektor dokumen.

    Kompleksitas:
      Waktu: O(N × k̄) dimana N=jumlah dokumen, k̄=rata-rata panjang dokumen
      Ruang: O(|V|) dimana |V| = ukuran vocabulary

    Args:
        corpus        : List of strings (teks yang sudah dinormalisasi)
        min_df        : Minimum document frequency — kata harus muncul
                        setidaknya di min_df dokumen (hapus kata sangat jarang)
        max_df_ratio  : Maximum df ratio [0..1] — hapus kata terlalu umum
                        (stopword de facto, misal: muncul di 95%+ dokumen)
        max_features  : Batasi ukuran vocabulary (ambil top-N berdasarkan df)

    Returns:
        Dict { kata: indeks_integer } — vocabulary terurut
    """
    if not corpus:
        return {}

    N = len(corpus)  # Total dokumen dalam corpus

    # ── Hitung Document Frequency per Term ───────────────────────────────────
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

    # ── Filter Berdasarkan min_df dan max_df_ratio ────────────────────────────
    # min_df : Hapus kata yang terlalu jarang (noise, typo, kata sangat spesifik)
    # max_df_ratio: Hapus kata yang terlalu umum (tidak informatif secara global)
    max_df_count = int(N * max_df_ratio)  # Konversi ratio → count absolut

    filtered_terms = {
        term for term, df in document_frequency.items()
        if min_df <= df <= max_df_count
    }

    # ── Batasi Ukuran Vocabulary (opsional: top-N berdasarkan df) ────────────
    if max_features and len(filtered_terms) > max_features:
        # Urutkan berdasarkan df (descending) → ambil top max_features
        # Kata dengan df tinggi = lebih representatif corpus
        sorted_by_df = sorted(
            filtered_terms,
            key=lambda t: document_frequency[t],
            reverse=True
        )
        filtered_terms = set(sorted_by_df[:max_features])

    # ── Buat Mapping Term → Indeks (urutan alfabetis untuk determinisme) ──────
    # Urutan alfabetis memastikan hasil reproducible (tidak acak tiap run)
    sorted_terms = sorted(filtered_terms)  # Urutkan O(|V| log|V|)

    # Pemetaan: { term: integer_index }
    # Indeks ini menentukan posisi kolom dalam matriks TF-IDF
    # Contoh: vocab = {'aku': 0, 'benci': 1, 'cinta': 2, ...}
    vocabulary = {term: idx for idx, term in enumerate(sorted_terms)}

    return vocabulary


def get_document_frequency(corpus: list[str], vocabulary: dict) -> dict:
    """
    Hitung Document Frequency untuk setiap term dalam vocabulary.

    df(t) = |{d ∈ D : t ∈ d}|
    = Jumlah dokumen yang mengandung term t (minimal 1 kali)

    Digunakan untuk menghitung IDF selanjutnya.

    Args:
        corpus     : List of strings
        vocabulary : Dict {term: indeks} dari build_vocabulary()

    Returns:
        Dict {term: df_count}
    """
    df = {term: 0 for term in vocabulary}  # Inisialisasi semua 0

    for doc in corpus:
        # Gunakan set untuk menghitung kemunculan unik per dokumen
        tokens_in_doc = set(doc.split())
        for token in tokens_in_doc:
            if token in df:  # Hanya term yang ada di vocabulary
                df[token] += 1

    return df


# =============================================================================
# BAGIAN 2: TERM FREQUENCY (TF)
# =============================================================================

def compute_tf(document: str, vocabulary: dict) -> list[float]:
    """
    Menghitung Term Frequency (TF) untuk SATU dokumen.

    Rumus:
      TF(t, d) = f(t, d) / |d|

      dimana:
        f(t, d) = frekuensi absolut term t dalam dokumen d
                  (berapa kali t muncul di d)
        |d|     = total jumlah token dalam d (panjang dokumen)

    Representasi Output:
      Vektor TF berukuran |V| (panjang vocabulary).
      Setiap elemen ke-i = TF dari term vocabulary[i] pada dokumen ini.
      Elemen = 0.0 jika term tidak muncul di dokumen.

    Contoh:
      vocab = {'bagus': 0, 'buruk': 1, 'tidak': 2}
      doc   = "tidak bagus tidak"
      TF    = [0.333, 0.0, 0.667]  → [1/3, 0/3, 2/3]

    Kompleksitas: O(|d| + |V|) — tokenisasi + inisialisasi vektor

    Args:
        document   : String teks (sudah dipreprocess & dinormalisasi)
        vocabulary : Dict {term: indeks}

    Returns:
        List of float berukuran len(vocabulary)
    """
    V = len(vocabulary)

    # Inisialisasi vektor TF dengan 0.0 untuk setiap term di vocabulary
    tf_vector = [0.0] * V

    # Tokenisasi dokumen: split pada whitespace
    tokens = document.split()
    total_tokens = len(tokens)  # |d| = panjang dokumen

    if total_tokens == 0:
        return tf_vector  # Dokumen kosong → semua TF = 0

    # ── Hitung Frekuensi Absolut per Token ───────────────────────────────────
    # term_count[t] = f(t, d) = berapa kali t muncul di d
    term_count = {}
    for token in tokens:
        if token in vocabulary:  # Hanya token yang ada di vocabulary
            term_count[token] = term_count.get(token, 0) + 1

    # ── Konversi ke TF Relatif dan Isi Vektor ────────────────────────────────
    # TF(t, d) = f(t, d) / |d|
    for term, count in term_count.items():
        idx = vocabulary[term]  # Dapatkan indeks kolom dari vocabulary
        tf_vector[idx] = count / total_tokens  # TF relatif

    return tf_vector


# =============================================================================
# BAGIAN 3: INVERSE DOCUMENT FREQUENCY (IDF)
# =============================================================================

def compute_idf(vocabulary: dict,
                document_frequency: dict,
                N: int,
                smooth: bool = True) -> list[float]:
    """
    Menghitung IDF (Inverse Document Frequency) untuk seluruh vocabulary.

    Dua Varian Rumus:
      1. Standard IDF (smooth=False):
            IDF(t) = log( N / df(t) )
            Masalah: Jika df(t) = 0 → ZeroDivisionError
                     Jika df(t) = N → IDF = 0 (term muncul di semua dok)

      2. Smooth IDF (smooth=True, default — sama dengan sklearn):
            IDF(t) = log( (N + 1) / (df(t) + 1) ) + 1
            Keuntungan:
              a) Menghindari pembagian nol (df tidak pernah 0 setelah +1)
              b) +1 di luar log mencegah IDF = 0 untuk term universaL
              c) Konsisten dengan implementasi sklearn TfidfVectorizer

    Catatan Pemilihan log:
      math.log() menggunakan logaritma natural (base e ≈ 2.718).
      Ini tidak mempengaruhi ranking relatif antar term, hanya scaling.
      Beberapa referensi menggunakan log₂ atau log₁₀ — hasilnya equivalent
      dalam perbandingan relative (ranking tidak berubah).

    Representasi Output:
      Vektor IDF berukuran |V|.
      idf_vector[i] = IDF dari term vocabulary[i].

    Kompleksitas: O(|V|) — iterasi sekali pada vocabulary

    Args:
        vocabulary         : Dict {term: indeks}
        document_frequency : Dict {term: df_count}
        N                  : Total jumlah dokumen dalam corpus
        smooth             : Gunakan smooth IDF (default: True)

    Returns:
        List of float berukuran len(vocabulary)
    """
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


# =============================================================================
# BAGIAN 4: TF-IDF WEIGHTING
# =============================================================================

def compute_tfidf_vector(tf_vector: list[float],
                         idf_vector: list[float],
                         normalize: bool = True) -> list[float]:
    """
    Menghitung vektor TF-IDF dari vektor TF dan IDF.

    Rumus per dimensi ke-i:
      TFIDF[i] = TF[i] × IDF[i]

    Ini adalah element-wise multiplication (perkalian Hadamard):
      TFIDF = TF ⊙ IDF

    Normalisasi L2 (opsional, sangat direkomendasikan):
      Tanpa normalisasi: dokumen panjang cenderung memiliki skor lebih tinggi
      karena lebih banyak term yang muncul. Normalisasi L2 menyeragamkan
      panjang semua vektor.

      v_norm[i] = v[i] / ||v||₂
      ||v||₂ = √(Σᵢ v[i]²)   (Euclidean norm / L2 norm)

      Setelah normalisasi: ||v_norm||₂ = 1 (unit vector)

    Args:
        tf_vector  : Vektor TF hasil compute_tf()
        idf_vector : Vektor IDF hasil compute_idf()
        normalize  : Terapkan L2 normalization (default: True)

    Returns:
        List of float — vektor TF-IDF untuk satu dokumen
    """
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


# =============================================================================
# BAGIAN 5: MATRIKS TF-IDF (PIPELINE UTAMA)
# =============================================================================

def fit_transform(corpus: list[str],
                  min_df: int = 1,
                  max_df_ratio: float = 1.0,
                  max_features: int = None,
                  smooth_idf: bool = True,
                  normalize: bool = True) -> dict:
    """
    Pipeline lengkap: Corpus → Matriks TF-IDF.

    Algoritma:
      1. BUILD VOCABULARY
         vocab = build_vocabulary(corpus, min_df, max_df_ratio, max_features)
         Hasil: { term: col_index } mapping

      2. HITUNG DOCUMENT FREQUENCY
         df[t] = |{d ∈ D : t ∈ d}| untuk t ∈ vocab

      3. HITUNG IDF (SEKALI untuk seluruh corpus)
         idf[t] = log((N+1)/(df[t]+1)) + 1   [smooth]

      4. UNTUK SETIAP DOKUMEN dᵢ:
         a. Hitung TF(t, dᵢ) → vektor ukuran |V|
         b. Hitung TF-IDF[j] = TF[j] × IDF[j]  untuk j = 0...|V|-1
         c. Normalisasi L2 (opsional)
         d. Tambahkan ke matriks

      5. OUTPUT: Matriks X ukuran (N × |V|)
         X[i][j] = TF-IDF bobot term ke-j pada dokumen ke-i

    Representasi Matriks:
      X = [
        [doc₀_term₀, doc₀_term₁, ..., doc₀_term_{|V|-1}],  ← dokumen 0
        [doc₁_term₀, doc₁_term₁, ..., doc₁_term_{|V|-1}],  ← dokumen 1
        ...
        [doc_{N-1}_term₀, ..., doc_{N-1}_term_{|V|-1}],     ← dokumen N-1
      ]

      Baris = dokumen, Kolom = term/fitur

    Kompleksitas:
      Waktu: O(N × |V|) — dominasi oleh konstruksi matriks
      Ruang: O(N × |V|) — menyimpan matriks

    Args:
        corpus        : List of strings (setelah preprocessing/normalisasi)
        min_df        : Min document frequency untuk masuk vocabulary
        max_df_ratio  : Max document frequency ratio (0..1)
        max_features  : Batasi jumlah fitur (None = tidak dibatasi)
        smooth_idf    : Gunakan smooth IDF formula
        normalize     : Normalisasi L2 per vektor dokumen

    Returns:
        Dict berisi:
          'matrix'     : List[List[float]] — matriks TF-IDF (N × |V|)
          'vocabulary' : Dict[str, int] — mapping term → indeks kolom
          'idf_vector' : List[float] — nilai IDF per term
          'feature_names': List[str] — nama term terurut sesuai indeks kolom
          'stats'      : Dict — statistik proses
    """
    N = len(corpus)
    if N == 0:
        return {
            'matrix': [], 'vocabulary': {}, 'idf_vector': [],
            'feature_names': [], 'stats': {}
        }

    # ── LANGKAH 1: Bangun Vocabulary ─────────────────────────────────────────
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

    # ── LANGKAH 2: Hitung Document Frequency ─────────────────────────────────
    document_frequency = get_document_frequency(corpus, vocabulary)

    # ── LANGKAH 3: Hitung IDF (satu kali untuk seluruh corpus) ───────────────
    idf_vector = compute_idf(
        vocabulary, document_frequency, N, smooth=smooth_idf
    )

    # ── LANGKAH 4: Bangun Matriks TF-IDF ─────────────────────────────────────
    # X = [] → akan menjadi matriks (N × V)
    matrix = []

    for doc in corpus:
        # 4a. Hitung TF untuk dokumen ini
        tf_vec = compute_tf(doc, vocabulary)

        # 4b+4c. Hitung TF-IDF dan normalisasi
        tfidf_vec = compute_tfidf_vector(tf_vec, idf_vector, normalize=normalize)

        # 4d. Tambahkan vektor dokumen ke matriks
        matrix.append(tfidf_vec)

    # ── LANGKAH 5: Kumpulkan Statistik ───────────────────────────────────────
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


def transform(corpus: list[str],
              vocabulary: dict,
              idf_vector: list[float],
              normalize: bool = True) -> list[list[float]]:
    """
    Transformasi dokumen BARU menggunakan vocabulary & IDF yang sudah ada.

    Berbeda dengan fit_transform() yang menghitung vocabulary & IDF dari awal,
    transform() menggunakan parameter yang SUDAH dihitung sebelumnya.

    Digunakan untuk: transform data test/validasi menggunakan vocabulary
    yang difit dari data training.

    ⚠ PENTING: Token di dokumen baru yang tidak ada di vocabulary akan
    diabaikan (out-of-vocabulary / OOV tokens). Ini perilaku standar TF-IDF.

    Args:
        corpus     : List of strings dokumen baru
        vocabulary : Dict {term: indeks} dari fit_transform()
        idf_vector : List[float] dari fit_transform()
        normalize  : Normalisasi L2

    Returns:
        Matriks TF-IDF List[List[float]] ukuran (len(corpus) × len(vocabulary))
    """
    matrix = []
    for doc in corpus:
        tf_vec    = compute_tf(doc, vocabulary)
        tfidf_vec = compute_tfidf_vector(tf_vec, idf_vector, normalize=normalize)
        matrix.append(tfidf_vec)
    return matrix


# =============================================================================
# BAGIAN 6: UTILITIES — Analisis & Visualisasi
# =============================================================================

def get_top_features(tfidf_result: dict,
                     doc_index: int,
                     top_n: int = 10) -> list[dict]:
    """
    Ambil top-N term dengan TF-IDF tertinggi untuk dokumen tertentu.

    Berguna untuk interpretasi: kata apa yang paling merepresentasikan
    dokumen ke-i?

    Args:
        tfidf_result : Output dari fit_transform()
        doc_index    : Indeks dokumen (baris matriks)
        top_n        : Jumlah term teratas yang diambil

    Returns:
        List of dict: [{'term': str, 'tfidf': float, 'rank': int}, ...]
    """
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


def get_idf_ranking(tfidf_result: dict, top_n: int = 20) -> list[dict]:
    """
    Ambil term dengan IDF tertinggi (kata paling unik/langka di corpus).

    Term dengan IDF tinggi = kata langka → informatif untuk klasifikasi.
    Term dengan IDF rendah = kata umum → seperti stopword.

    Args:
        tfidf_result : Output dari fit_transform()
        top_n        : Jumlah term yang ditampilkan

    Returns:
        List of dict: [{'term': str, 'idf': float, 'df': int}, ...]
    """
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


def compute_cosine_similarity(vec_a: list[float],
                               vec_b: list[float]) -> float:
    """
    Hitung Cosine Similarity antara dua vektor TF-IDF.

    Rumus:
      sim(A, B) = (A · B) / (||A||₂ × ||B||₂)
               = Σᵢ(Aᵢ × Bᵢ) / (√Σᵢ Aᵢ² × √Σᵢ Bᵢ²)

    Nilai range: [0, 1] untuk vektor non-negatif (TF-IDF selalu ≥ 0)
      1 = identik, 0 = ortogonal (tidak ada kesamaan)

    Catatan: Jika vektor sudah dinormalisasi L2 (||v|| = 1),
    maka cosine similarity = dot product saja: Σᵢ(Aᵢ × Bᵢ)

    Args:
        vec_a : Vektor TF-IDF dokumen A
        vec_b : Vektor TF-IDF dokumen B

    Returns:
        Float cosine similarity [0..1]
    """
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


def matrix_to_preview(matrix: list[list[float]],
                      feature_names: list[str],
                      max_rows: int = 5,
                      max_cols: int = 10) -> dict:
    """
    Buat preview matriks TF-IDF untuk ditampilkan di UI.

    Karena matriks penuh bisa sangat besar (N × |V|),
    fungsi ini memotong untuk tampilan: max_rows × max_cols.

    Args:
        matrix        : Matriks TF-IDF penuh
        feature_names : Nama kolom/term
        max_rows      : Maksimum baris yang ditampilkan
        max_cols      : Maksimum kolom yang ditampilkan

    Returns:
        Dict dengan data preview dan metadata dimensi
    """
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
