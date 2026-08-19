# Language Models & Agents — Individual Project

## Phase 1: Data Collection & Tokenizer Construction

**Author:** [Raj k jain] | [2025201036]

---   

## Language Selection

| Model | Language | Script | Tier | Justification |
|-------|----------|--------|------|---------------|
| **Model H** | Hindi | Devanagari | Higher-resource | Hindi is one of the most widely spoken Indian languages with abundant public text corpora available (e.g., AI4Bharat Sangraha). Large-scale verified datasets enable reaching the ~500M token target comfortably. |
| **Model L** | Nepali | Devanagari | Lower-resource | Nepali is on the allowed lower-resource list. While Nepali shares the Devanagari script with Hindi, its linguistic structure, vocabulary, and morphology are distinct. Public data from Sangraha (verified Nepali split) provides sufficient coverage to meet the token target. |

---

## Repository Structure

```
.
├── README.md                          # This file
├── .gitignore
│
├── hindi/
│   ├── scripts/
│   │   ├── hindi_data_collect.py      # Resumable, chunked data collection script
│   │   └── hindi_tokenize.py          # Parallelized tokenizer & split export script
│   ├── tokenizer/
│   │   ├── hindi_tokenizer.model      # Trained SentencePiece model
│   │   └── hindi_tokenizer.vocab      # Vocabulary file (32,000 tokens)
│   └── data/                          # Output directory for dataset splits
│       ├── hindi_train.txt
│       ├── hindi_val.txt
│       └── hindi_test.txt
│
├── nepali/
│   ├── scripts/
│   │   ├── nepali_data_collect.py     # Resumable, chunked data collection script
│   │   └── nepali_tokenize.py         # Parallelized tokenizer & split export script
│   ├── tokenizer/
│   │   ├── nepali_tokenizer.model     # Trained SentencePiece model
│   │   └── nepali_tokenizer.vocab     # Vocabulary file (16,000 tokens)
│   └── data/                          # Output directory for dataset splits
│       ├── nepali_train.txt
│       ├── nepali_val.txt
│       └── nepali_test.txt
│
└── report/
    ├── phase1_report.md               # Detailed Phase 1 report with all stats & analysis
    └── phase1_report.tex              # LaTeX equivalent of the report
```

---

## Google Drive Links

| Artifact | Language | Link |
|----------|----------|------|
| Train/Val/Test splits (`.txt`) | Hindi | https://drive.google.com/drive/folders/1WXVGFmtd4bqrctrEArb3Y-tlOfAA55dF?usp=sharing |
| Train/Val/Test splits (`.txt`) | Nepali | https://drive.google.com/drive/folders/1Lq7IYyVIRJTbLPG1fdc2if_RiI713HS7?usp=sharing |
| SQLite database (`hindi_state.db`, 8.0 GB) | Hindi | https://drive.google.com/file/d/1EaZsr74aLL0g70TVOkbckQr62NDk90CH/view?usp=sharing |
| SQLite database (`nepali_state.db`, 11.0 GB) | Nepali | https://drive.google.com/file/d/1avTD7KQ92QcKJojcmqSV3jb4bW1q3vHH/view?usp=sharing |

---

Books collection : https://drive.google.com/drive/folders/1j4u5S7glsMkOuiD8I-74cXO1q8Sy6Ajx?usp=sharing

## Reproduction Steps

### Prerequisites
```bash
pip install sentencepiece pymupdf pdf2image pytesseract datasets trafilatura
```
System dependency (for OCR): `sudo apt install tesseract-ocr tesseract-ocr-hin tesseract-ocr-nep`

### Stage 1: Data Collection
From the root directory, run the highly optimized, memory-safe data collection scripts:
```bash
python hindi/scripts/hindi_data_collect.py
python nepali/scripts/nepali_data_collect.py
```
*These scripts will handle downloading from Hugging Face, OCRing PDFs in `books/`, web-scraping, and streaming them all directly into `hindi_state.db` and `nepali_state.db` safely.*

### Stage 2: Tokenizer Training & Split Export
After the data is collected, generate your tokenizers and `.txt` splits by running:
```bash
python hindi/scripts/hindi_tokenize.py
python nepali/scripts/nepali_tokenize.py
```
*These scripts train a SentencePiece model, parallel-encode millions of documents across CPU cores, and export the datasets cleanly into `hindi/data/` and `nepali/data/`.*

---

## Dataset Summary

| Metric | Hindi (Model H) | Nepali (Model L) |
|--------|-----------------|-------------------|
| **Total tokens (all splits)** | 681,200,436 | 775,992,967 |
| **Train tokens** | 613,059,508 | 698,466,193 |
| **Total documents** | 1,323,072 | 1,836,292 |
| **Manual tokens (train)** | 6,777,503 (1.11%) | 322,028 (0.05%) |
| **Downloaded tokens (train)** | 606,282,005 (98.89%) | 698,144,165 (99.95%) |

> **Note on manual data fraction:** The current manual data fraction is below the 20% target for both languages due to the overwhelming volume of the Sangraha corpus. See the `report/phase1_report.md` for a detailed discussion.

---

## Tokenizer Summary

| Metric | Hindi | Nepali |
|--------|-------|--------|
| Algorithm | SentencePiece (BPE) | SentencePiece (BPE) |
| Vocabulary size | 32,000 | 16,000 |
| Avg chars per token | 4.12 | 4.59 |
| Fertility (tokens/word) | 1.24 | 1.40 |
| Unknown-token rate (val) | 0.0485% | 0.1585% |

> **Known Limitations:** Redistribution rights for the Sangraha corpus and several manual sources are currently unverified. The database logs these as "unknown — verify redistribution rights". Strict open-source redistribution licenses have not been fully confirmed for all subsets.

---

## Contact

Name - Raj k jain
Roll No. - 2025201036
Email -  raj.jain@students.iiit.ac.in
