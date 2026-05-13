"""
=============================================================================
MODUL 1: DATA & PREPROCESSING
=============================================================================
Judul TA : Analisis Pengaruh Teknik Normalisasi Kata Gaul (Slang) terhadap
           Akurasi Deteksi Ujaran Kebencian Berbahasa Indonesia
=============================================================================
Implementasi murni Python (Pure Python) tanpa library tingkat tinggi.
Setiap fungsi dilengkapi komentar matematis/logika untuk keperluan sidang.
=============================================================================
"""

import re
import csv
import os

# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 1: PEMBACAAN FILE (CSV & EXCEL)
# ─────────────────────────────────────────────────────────────────────────────

def read_csv_file(filepath: str, delimiter: str = ',') -> list[dict]:
    """
    Membaca file CSV menggunakan library `csv` bawaan Python.

    Representasi Data:
      Hasil akhir adalah List of Dictionaries (LoD):
        [ {'kolom1': 'nilai', 'kolom2': 'nilai'}, ... ]

    Kompleksitas: O(n) — linear terhadap jumlah baris n.

    Args:
        filepath  : Path lengkap ke file .csv
        delimiter : Pemisah kolom, default koma ','

    Returns:
        List of Dictionaries yang merepresentasikan setiap baris data.
    """
    result = []

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

    with open(filepath, mode='r', encoding='utf-8-sig', newline='') as f:
        # csv.DictReader otomatis menjadikan baris pertama sebagai header (key dict)
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            # Konversi OrderedDict → dict biasa, strip whitespace per nilai
            result.append({k: v.strip() if isinstance(v, str) else v
                           for k, v in dict(row).items()})

    return result


def read_excel_file(filepath: str, sheet_name: str = None) -> list[dict]:
    """
    Membaca file Excel (.xlsx) menggunakan `openpyxl` HANYA untuk ekstraksi
    data mentah ke dalam List of Dictionaries Python.

    Logika Pembacaan:
      1. Buka workbook → pilih sheet (aktif atau berdasarkan nama).
      2. Baris pertama (row index 1) dijadikan header/key dictionary.
      3. Setiap baris berikutnya (row index 2..n) dipetakan ke dict
         menggunakan zip(headers, values) → O(k) per baris, k = jumlah kolom.
      4. Sel kosong (None) dikonversi ke string kosong ''.

    Kompleksitas: O(n × k) — n baris, k kolom.

    Args:
        filepath   : Path ke file .xlsx
        sheet_name : Nama sheet. Jika None, gunakan active sheet.

    Returns:
        List of Dictionaries merepresentasikan isi sheet.
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl diperlukan. Install: pip install openpyxl")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)

    # Pilih worksheet
    if sheet_name and sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
    else:
        ws = wb.active

    result = []
    headers = []

    for i, row in enumerate(ws.iter_rows(values_only=True)):
        # Baris pertama (i=0) → jadikan header
        if i == 0:
            # Normalisasi header: lowercase & strip spasi
            headers = [str(cell).strip().lower() if cell is not None else f'col_{j}'
                       for j, cell in enumerate(row)]
            continue

        # Baris data: zip header dengan nilai sel
        # zip() → pasangkan setiap header[j] dengan row[j]
        row_values = [str(cell).strip() if cell is not None else ''
                      for cell in row]

        # Lewati baris yang seluruhnya kosong
        if all(v == '' for v in row_values):
            continue

        row_dict = dict(zip(headers, row_values))
        result.append(row_dict)

    wb.close()
    return result


def read_dataset(filepath: str, text_col: str = 'text',
                 label_cols: list[str] = None) -> list[dict]:
    """
    Wrapper umum untuk membaca dataset (CSV atau Excel).
    Mengembalikan list dengan kolom 'text', dictionary 'labels', dan string 'label' untuk display.

    Args:
        filepath  : Path file (.csv / .xlsx / .xls)
        text_col  : Nama kolom teks pada file
        label_cols: List nama kolom label pada file

    Returns:
        List of dict: [{'text': '...', 'labels': {...}, 'label': '...'}, ...]
    """
    if label_cols is None:
        label_cols = ['label']
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.csv':
        raw = read_csv_file(filepath)
    elif ext in ('.xlsx', '.xls'):
        raw = read_excel_file(filepath)
    else:
        raise ValueError(f"Format file tidak didukung: {ext}")

    # Ekstrak hanya kolom yang relevan
    dataset = []
    for row in raw:
        # Cari kolom teks
        text_val = next((v for k, v in row.items()
                         if k.lower() == text_col.lower()), '')
        
        # Cari semua kolom label
        labels_dict = {}
        for lc in label_cols:
            val = next((v for k, v in row.items() if k.lower() == lc.lower()), '')
            labels_dict[lc] = val
        
        # Buat string representasi untuk display
        label_str = " | ".join(f"{k}: {v}" for k, v in labels_dict.items())
        
        dataset.append({'text': text_val, 'labels': labels_dict, 'label': label_str})

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 2: TEXT CLEANSING
# ─────────────────────────────────────────────────────────────────────────────

# Pola Regex yang dikompilasi sekali untuk efisiensi (re.compile → O(1) lookup)
_RE_URL     = re.compile(r'https?://\S+|www\.\S+')
_RE_MENTION = re.compile(r'@\w+')
_RE_HASHTAG = re.compile(r'#\w+')
_RE_HTML    = re.compile(r'<[^>]+>')
_RE_NUMBER  = re.compile(r'\d+')
_RE_SYMBOL  = re.compile(r'[^a-zA-Z\s]')       # Hapus semua non-huruf non-spasi
_RE_SPACES  = re.compile(r'\s+')               # Normalisasi multi-spasi → satu spasi


def cleansing(text: str, steps_log: bool = False) -> str | tuple:
    """
    Fungsi text cleansing manual menggunakan `re` dan manipulasi string.

    Pipeline Cleansing (urutan penting):
      1. Lowercase          : Σ(c) → c.lower()  — Normalisasi kapitalisasi
      2. Hapus URL          : re.sub(_RE_URL, '')
      3. Hapus Mention (@)  : re.sub(_RE_MENTION, '')
      4. Hapus Hashtag (#)  : re.sub(_RE_HASHTAG, '')
      5. Hapus HTML Tag     : re.sub(_RE_HTML, '')
      6. Hapus Angka        : re.sub(_RE_NUMBER, '')
      7. Hapus Simbol/Puncts: re.sub(_RE_SYMBOL, ' ')
      8. Strip & Normalize  : re.sub(_RE_SPACES, ' ').strip()

    Catatan Matematis:
      Setiap operasi re.sub adalah O(n) dimana n = len(text).
      Total pipeline: O(8n) ≈ O(n) (linear).

    Args:
        text      : String teks mentah
        steps_log : Jika True, kembalikan tuple (hasil, log_langkah)

    Returns:
        Teks bersih (str), atau (str, list) jika steps_log=True
    """
    if not isinstance(text, str):
        text = str(text)

    steps = []

    # Langkah 1: Lowercase — standarisasi huruf kapital
    t = text.lower()
    if steps_log: steps.append(('1. Lowercase', t))

    # Langkah 2: Hapus URL (http://, https://, www.)
    t = _RE_URL.sub('', t)
    if steps_log: steps.append(('2. Hapus URL', t))

    # Langkah 3: Hapus mention Twitter/sosmed (@username)
    t = _RE_MENTION.sub('', t)
    if steps_log: steps.append(('3. Hapus Mention', t))

    # Langkah 4: Hapus hashtag (#topik)
    t = _RE_HASHTAG.sub('', t)
    if steps_log: steps.append(('4. Hapus Hashtag', t))

    # Langkah 5: Hapus tag HTML (<br>, <p>, dll.)
    t = _RE_HTML.sub('', t)
    if steps_log: steps.append(('5. Hapus HTML', t))

    # Langkah 6: Hapus angka (digit [0-9])
    t = _RE_NUMBER.sub('', t)
    if steps_log: steps.append(('6. Hapus Angka', t))

    # Langkah 7: Hapus simbol & tanda baca → ganti dengan spasi
    # [^a-zA-Z\s] = karakter yang BUKAN huruf dan BUKAN whitespace
    t = _RE_SYMBOL.sub(' ', t)
    if steps_log: steps.append(('7. Hapus Simbol', t))

    # Langkah 8: Normalisasi spasi ganda → satu spasi, lalu strip tepi
    # re.sub(\s+) → menggabungkan semua whitespace berturut-turut
    t = _RE_SPACES.sub(' ', t).strip()
    if steps_log: steps.append(('8. Normalize Spasi', t))

    return (t, steps) if steps_log else t


# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 3: SISTEM NORMALISASI LEKSIKON DINAMIS
# ─────────────────────────────────────────────────────────────────────────────

def read_lexicon_file(filepath: str,
                      slang_col: str = 'slang',
                      formal_col: str = 'formal') -> dict:
    """
    Membaca file kamus slang (CSV atau Excel) dan mengembalikan dictionary.

    Struktur Output:
      { 'kata_slang': 'kata_baku', ... }
      Contoh: { 'gak': 'tidak', 'bgt': 'banget', 'yg': 'yang' }

    Logika Lookup:
      Dictionary Python menggunakan hash table → O(1) average untuk lookup.
      Ini jauh lebih efisien daripada linear search O(n) di list.

    Args:
        filepath  : Path file kamus
        slang_col : Nama kolom kata slang
        formal_col: Nama kolom kata baku/formal

    Returns:
        Dict {slang: formal}
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.csv':
        rows = read_csv_file(filepath)
    elif ext in ('.xlsx', '.xls'):
        rows = read_excel_file(filepath)
    else:
        raise ValueError(f"Format tidak didukung: {ext}")

    lexicon = {}
    for row in rows:
        # Case-insensitive column matching
        slang_val = next((v for k, v in row.items()
                          if k.lower() == slang_col.lower()), None)
        formal_val = next((v for k, v in row.items()
                           if k.lower() == formal_col.lower()), None)

        if slang_val and formal_val:
            # Normalisasi key: lowercase & strip untuk konsistensi lookup
            lexicon[slang_val.lower().strip()] = formal_val.lower().strip()

    return lexicon


def merge_lexicons(lexicon_internal: dict,
                   lexicon_external: dict,
                   conflict_strategy: str = 'internal_priority') -> dict:
    """
    Menggabungkan dua kamus slang, menghapus duplikat, dan menangani konflik.

    Strategi Penggabungan (Set-theoretic Union dengan resolusi konflik):
      Misalkan:
        L_i = Kamus Internal  (prioritas utama penelitian)
        L_e = Kamus Eksternal (Saka-NLP, sebagai suplemen)

      Kamus Gabungan: L_g = L_i ∪ L_e

      Untuk setiap kata w yang ada di L_i ∩ L_e (irisan / konflik):
        - 'internal_priority' : L_g[w] = L_i[w]  (internal menang)
        - 'external_priority' : L_g[w] = L_e[w]  (eksternal menang)
        - 'keep_both'         : simpan L_i[w] (default ke internal)

    Statistik yang dicatat:
      |L_i| = total entri internal
      |L_e| = total entri eksternal
      |L_i ∩ L_e| = jumlah konflik (kata yang sama, nilai berbeda)
      |L_g| = total entri gabungan (≤ |L_i| + |L_e|)

    Args:
        lexicon_internal   : Dict kamus internal peneliti
        lexicon_external   : Dict kamus Saka-NLP / sumber luar
        conflict_strategy  : Strategi resolusi konflik

    Returns:
        Tuple (merged_dict, stats_dict)
    """
    conflicts = {}   # {slang: {'internal': val, 'external': val}}
    duplicates = []  # kata yang sama DAN nilai yang sama (benar-benar duplikat)

    # Mulai dari salinan kamus internal (baseline)
    merged = dict(lexicon_internal)

    for slang, formal in lexicon_external.items():
        if slang in merged:
            if merged[slang] == formal:
                # Duplikat murni: kata & makna sama → abaikan
                duplicates.append(slang)
            else:
                # Konflik: kata sama tapi makna berbeda → catat & resolusi
                conflicts[slang] = {
                    'internal': merged[slang],
                    'external': formal,
                    'resolved': merged[slang]  # default: internal menang
                }
                if conflict_strategy == 'external_priority':
                    merged[slang] = formal
                    conflicts[slang]['resolved'] = formal
                # 'internal_priority' & 'keep_both' → pertahankan internal (sudah ada)
        else:
            # Kata baru dari eksternal → tambahkan
            merged[slang] = formal

    stats = {
        'total_internal'  : len(lexicon_internal),
        'total_external'  : len(lexicon_external),
        'total_merged'    : len(merged),
        'true_duplicates' : len(duplicates),
        'conflicts'       : conflicts,
        'conflict_count'  : len(conflicts),
        'new_from_external': len(merged) - len(lexicon_internal),
        'strategy_used'   : conflict_strategy
    }

    return merged, stats


def tokenize(text: str) -> list[str]:
    """
    Tokenisasi sederhana: memecah kalimat menjadi daftar token (kata).

    Metode: split berdasarkan whitespace.
    Definisi formal:
      T = text.split()
      → Token ke-i: T[i] untuk i ∈ {0, 1, ..., |T|-1}

    Kompleksitas: O(n) — n = panjang string.

    Args:
        text : String teks (sudah di-cleanse)

    Returns:
        List of string tokens
    """
    # str.split() tanpa argumen: split pada satu atau lebih whitespace
    return text.split()


def normalize_text(text: str,
                   lexicon: dict,
                   return_log: bool = False) -> str | tuple:
    """
    Normalisasi teks dengan lookup ke kamus gabungan.

    Algoritma Normalisasi Per-Token:
      Input : Kalimat S = [t_0, t_1, ..., t_{n-1}]
      Proses: Untuk setiap token t_i:
                jika t_i ∈ Kamus_Gabungan:
                  t_i' = Kamus_Gabungan[t_i]   # O(1) hash lookup
                else:
                  t_i' = t_i                   # Token dipertahankan
      Output: S' = ' '.join([t_0', t_1', ..., t_{n-1}'])

    Kompleksitas Total: O(n) — n = jumlah token.
    Setiap lookup dict adalah O(1) amortized (hash table).

    Args:
        text       : Teks yang sudah di-cleanse
        lexicon    : Kamus gabungan {slang: formal}
        return_log : Jika True, kembalikan log setiap penggantian

    Returns:
        Teks ternormalisasi (str), atau (str, list_log) jika return_log=True
    """
    tokens = tokenize(text)  # Tokenisasi: O(n)
    normalized_tokens = []
    replacement_log = []

    for token in tokens:
        # Lookup O(1): cek apakah token ada di kamus
        if token in lexicon:
            replacement = lexicon[token]
            normalized_tokens.append(replacement)
            if return_log:
                replacement_log.append({
                    'original' : token,
                    'replaced' : replacement
                })
        else:
            # Token tidak ditemukan di kamus → pertahankan
            normalized_tokens.append(token)

    # Gabungkan token kembali menjadi kalimat
    result = ' '.join(normalized_tokens)

    return (result, replacement_log) if return_log else result


# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 4: FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def full_preprocessing_pipeline(text: str,
                                 lexicon: dict,
                                 verbose: bool = False) -> dict:
    """
    Pipeline lengkap: Raw Text → Cleansed → Normalized.

    Urutan Proses:
      Raw Text
        ↓ cleansing()     → hilangkan noise (URL, simbol, angka)
        ↓ normalize_text() → ganti slang → kata baku
      Final Text

    Args:
        text    : Teks mentah
        lexicon : Kamus gabungan
        verbose : Tampilkan log detail

    Returns:
        Dictionary hasil dengan semua tahapan
    """
    # Tahap 1: Cleansing
    cleansed, cleansing_steps = cleansing(text, steps_log=True)

    # Tahap 2: Normalisasi
    normalized, replacement_log = normalize_text(
        cleansed, lexicon, return_log=True
    )

    result = {
        'raw'            : text,
        'cleansed'       : cleansed,
        'normalized'     : normalized,
        'cleansing_steps': cleansing_steps if verbose else [],
        'replacements'   : replacement_log,
        'stats': {
            'raw_length'       : len(text),
            'cleansed_length'  : len(cleansed),
            'normalized_length': len(normalized),
            'tokens_raw'       : len(tokenize(cleansed)),
            'replacements_made': len(replacement_log)
        }
    }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# BAGIAN 5: KAMUS BAWAAN (FALLBACK JIKA TIDAK ADA FILE UPLOAD)
# ─────────────────────────────────────────────────────────────────────────────

BUILTIN_INTERNAL_LEXICON = {
    'gak'   : 'tidak', 'ga'    : 'tidak', 'nggak' : 'tidak',
    'bgt'   : 'banget','udah'  : 'sudah', 'udh'   : 'sudah',
    'yg'    : 'yang',  'yg'    : 'yang',  'dgn'   : 'dengan',
    'tdk'   : 'tidak', 'hrs'   : 'harus', 'blm'   : 'belum',
    'krn'   : 'karena','karna' : 'karena','bkn'   : 'bukan',
    'sdh'   : 'sudah', 'skrg'  : 'sekarang', 'utk': 'untuk',
    'gmn'   : 'bagaimana', 'gimana': 'bagaimana',
    'emang' : 'memang','emg'   : 'memang',
    'jgn'   : 'jangan','jgn'   : 'jangan',
    'tp'    : 'tapi',  'tapi'  : 'tetapi',
    'sm'    : 'sama',  'ama'   : 'sama',
    'klo'   : 'kalau', 'kl'    : 'kalau', 'kalo'  : 'kalau',
    'org'   : 'orang', 'orng'  : 'orang',
    'bro'   : 'saudara','sob'  : 'sahabat',
    'gue'   : 'saya',  'gw'    : 'saya',  'aku'   : 'saya',
    'lo'    : 'kamu',  'lu'    : 'kamu',  'elo'   : 'kamu',
    'mereka': 'mereka','dia'   : 'dia',
    'nih'   : 'ini',   'tuh'   : 'itu',
    'nyebelin': 'menjengkelkan', 'nyebalin': 'menjengkelkan',
    'bangsat': 'bajingan', 'brengsek': 'brengsek',
    'kampret': 'celaka', 'sial' : 'celaka',
    'bodo'  : 'bodoh', 'tolol' : 'tolol',
    'gila'  : 'gila',  'gile'  : 'gila',
    'aing'  : 'saya',  'maneh' : 'kamu',
    'wkwk'  : '',      'haha'  : '', 'hehe': '', 'xixi': '',
    'lol'   : '',      'omg'   : '',
    'msh'   : 'masih', 'masi'  : 'masih',
    'jd'    : 'jadi',  'lbh'   : 'lebih',
    'byk'   : 'banyak','sdikit': 'sedikit',
    'trs'   : 'terus', 'abis'  : 'habis', 'abish': 'habis',
}

BUILTIN_EXTERNAL_LEXICON = {
    'gak'   : 'tidak', 'nggak' : 'tidak', 'kagak' : 'tidak',
    'enggak': 'tidak', 'ngga'  : 'tidak',
    'bgt'   : 'banget','baget' : 'banget',
    'yg'    : 'yang',  'krn'   : 'karena',
    'kpd'   : 'kepada','thd'   : 'terhadap',
    'dll'   : 'dan lain-lain', 'dsb'   : 'dan sebagainya',
    'dlm'   : 'dalam', 'dpt'   : 'dapat',
    'tsb'   : 'tersebut', 'spt' : 'seperti',
    'sdgkan': 'sedangkan', 'jk' : 'jika',
    'mau'   : 'ingin', 'pengen': 'ingin', 'pgn'   : 'ingin',
    'dah'   : 'sudah', 'da'    : 'sudah',
    'pd'    : 'pada',  'spy'   : 'supaya', 'biar'  : 'supaya',
    'ttg'   : 'tentang',
    'lagi'  : 'sedang','lg'    : 'sedang', 'lagi'  : 'lagi',
    'aja'   : 'saja',  'aj'    : 'saja',
    'doang' : 'saja',  'doank' : 'saja',
    'sih'   : '',      'deh'   : '', 'dong'  : '',
    'kan'   : 'bukan', 'kok'   : '',
    'makanya': 'maka dari itu',
    'soalnya': 'karena', 'soale': 'karena',
    'temen' : 'teman', 'tmn'   : 'teman',
    'cowok' : 'laki-laki', 'cewek': 'perempuan',
    'bokap' : 'ayah',  'nyokap': 'ibu',
    'mantap': 'bagus', 'mantul': 'mantap betul',
    'kepo'  : 'ingin tahu', 'baper': 'terbawa perasaan',
    'gabut' : 'tidak ada kegiatan',
    'nolep' : 'tidak punya kehidupan sosial',
    'receh' : 'tidak penting', 'garing': 'tidak lucu',
    'lebay' : 'berlebihan', 'alay'  : 'norak',
}
