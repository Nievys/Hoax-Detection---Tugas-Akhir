# Modul 1: Data & Preprocessing — Tugas Akhir Ujaran Kebencian

## Cara Menjalankan

```bash
# 1. Install dependensi
pip install flask openpyxl

# 2. Jalankan server
python app.py

# 3. Buka browser
http://localhost:5000
```

## Struktur Project

```
hate_speech_app/
├── app.py                          ← Flask web server (backend)
├── requirements.txt
├── modules/
│   └── preprocessing.py            ← MODUL UTAMA (Pure Python)
├── templates/
│   └── index.html                  ← Frontend UI
└── data/
    ├── lexicons/
    │   └── kamus_internal_sample.csv   ← Contoh kamus (CSV)
    └── dataset_sample.csv              ← Contoh dataset
```

## Format File Kamus (CSV/Excel)

| slang | formal |
|-------|--------|
| gak   | tidak  |
| bgt   | banget |

## Format File Dataset (CSV/Excel)

| text                    | label |
|-------------------------|-------|
| Dasar brengsek lo!      | 1     |
| Terima kasih bantuannya | 0     |

## Library yang Digunakan (HANYA Pure Python + minimal)
- `re` — regex bawaan Python (cleansing)
- `csv` — bawaan Python (baca CSV)
- `openpyxl` — HANYA untuk ekstraksi data mentah Excel
- `flask` — web server

## TIDAK menggunakan: Pandas, Scikit-Learn, NLTK, Sastrawi
