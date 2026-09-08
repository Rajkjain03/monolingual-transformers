# Language Models & Agents — Individual Project

Complete submission for the three-phase monolingual Transformer project. Model H is Hindi and Model L is Nepali; each model has its own corpus, tokenizer, vocabulary, weights, pretraining run, and reasoning-finetuning data.

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
│   │   ├── model.py                   # Decoder-only Transformer architecture
│   │   ├── train.py                   # Resumable pretraining with AMP/accumulation
│   │   ├── evaluate.py                # PPL/BPB/generation/attention evaluation
│   │   ├── dataloader.py              # Memory-mapped dataset loader
│   │   ├── prepare_data.py            # Converts text to binary memmaps
│   │   ├── hindi_data_collect.py      # Resumable data collection
│   │   ├── hindi_tokenize.py          # Tokenizer training and split export
│   │   └── verify_causal_mask.py     # Empirical causal-mask test
│   ├── configs/
│   │   └── config.json                # Model and training hyperparameters
│   ├── tokenizer/
│   │   ├── hindi_tokenizer.model      # Trained SentencePiece model
│   │   └── hindi_tokenizer.vocab      # Vocabulary file (32,000 tokens)
│   ├── data/                          # Text and tokenized train/val/test splits
│       ├── hindi_train.txt
│       ├── hindi_val.txt
│       └── hindi_test.txt
│   └── checkpoints/                    # Local checkpoint; Drive link above
│
│
├── nepali/
│   ├── scripts/
│   │   ├── model.py                   # Decoder-only Transformer architecture
│   │   ├── train.py                   # Pretraining script with AMP and gradient accum
│   │   ├── evaluate.py                # Evaluation script (metrics & attention heatmaps)
│   │   ├── dataloader.py              # Memory-mapped dataset loader
│   │   ├── prepare_data.py            # Converts .txt to binary memmap format
│   │   ├── nepali_data_collect.py     # Resumable, chunked data collection script
│   │   └── nepali_tokenize.py         # Parallelized tokenizer & split export script
│   ├── tokenizer/
│   │   ├── nepali_tokenizer.model     # Trained SentencePiece model
│   │   └── nepali_tokenizer.vocab     # Vocabulary file (16,000 tokens)
│   ├── configs/
│   │   └── config.json                # Model architecture & training hyperparameters
│   ├── data/                          # Text and tokenized train/val/test splits
│   │   ├── nepali_test.bin            # Tokenized evaluation data
│   │   ├── nepali_train.bin           # Tokenized training data
│   │   └── nepali_train.txt
│   └── checkpoints/                   # Local checkpoint; Drive link above
│
├── phase3/
│   ├── configs/                       # Reasoning-finetuning run configs
│   ├── data/<language>/                # Leakage-controlled JSONL splits
│   ├── scripts/                       # Generation, finetuning, evaluation, plotting
│   ├── results/                       # Metrics and per-example predictions
│   └── checkpoints/<language>/         # Local best/last resumable checkpoints
│
└── report/
    ├── phase1_report.md               # Detailed Phase 1 report with all stats & analysis
    ├── phase2_report.md               # Detailed Phase 2 report (architecture, evaluation)
    ├── phase3_report.md               # Consolidated final report and Phase 3 analysis
    └── images/                        # Generated plots and heatmaps
```

The Phase 3 paths above are rooted at `phase3/`: `phase3/configs/`,
`phase3/data/`, `phase3/scripts/`, `phase3/results/`, and
`phase3/checkpoints/`.

---

## Google Drive Links

| Artifact | Language | Link |
|----------|----------|------|
| Train/Val/Test splits (`.txt`) | Hindi | https://drive.google.com/drive/folders/1WXVGFmtd4bqrctrEArb3Y-tlOfAA55dF?usp=sharing |
| Train/Val/Test splits (`.txt`) | Nepali | https://drive.google.com/drive/folders/1Lq7IYyVIRJTbLPG1fdc2if_RiI713HS7?usp=sharing |
| SQLite database (`hindi_state.db`, 8.0 GB) | Hindi | https://drive.google.com/file/d/1EaZsr74aLL0g70TVOkbckQr62NDk90CH/view?usp=sharing |
| SQLite database (`nepali_state.db`, 11.0 GB) | Nepali | https://drive.google.com/file/d/1avTD7KQ92QcKJojcmqSV3jb4bW1q3vHH/view?usp=sharing |
| Pretrained Checkpoint (`final.pt`) | Hindi | https://drive.google.com/file/d/1QpTcnaJEnypQP3c9NFM790h3enfT7xlm/view?usp=sharing |
| Pretrained Checkpoint (`final.pt`) | Nepali | https://drive.google.com/file/d/1a7cVgfeJVn6-yiZWHhG2N4YwfBfJmdwh/view?usp=sharing |
| Phase 3 finetuned checkpoints (`best.pt`, `last.pt`) | Hindi | https://drive.google.com/drive/folders/1SliWJWeMXxkpp7ybroLu3asjcFZnEQ7H?usp=sharing |
| Phase 3 finetuned checkpoints (`best.pt`, `last.pt`) | Nepali | https://drive.google.com/drive/folders/1JEJy1XVYIse2uOeSo7_7pW7SiMrnj-vo?usp=sharing |

---

Books collection : https://drive.google.com/drive/folders/1j4u5S7glsMkOuiD8I-74cXO1q8Sy6Ajx?usp=sharing

All Files - https://drive.google.com/drive/folders/1yEQ_m8RPMGJmMhQApk2cNl_0e3aqTwnE?usp=sharing

All dataset, pretrained-checkpoint, and Phase 3 finetuned-checkpoint links are maintained together in the table above. The Phase 3 folders contain both `best.pt` for final evaluation and `last.pt` for resume-capable continuation.

## Reproduction Steps

### Prerequisites
```bash
pip install torch sentencepiece pymupdf pdf2image pytesseract datasets trafilatura sacrebleu rouge-score
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

### Stage 3: Phase 2 Pretraining
Convert the text files into memory-mapped binary arrays (for efficient RAM usage during training):
```bash
python hindi/scripts/prepare_data.py
python nepali/scripts/prepare_data.py
```
Train the models from scratch (resumable if interrupted):
```bash
python hindi/scripts/train.py
python nepali/scripts/train.py
```

### Stage 4: Phase 2 Evaluation
Evaluate PPL, BPB, Generation Metrics (BLEU/chrF/ROUGE-L), and Attention Heatmaps:
```bash
python hindi/scripts/evaluate.py
python nepali/scripts/evaluate.py
```

### Stage 5: Phase 3 Reasoning Finetuning and Analysis

Generate the deterministic, leakage-controlled reasoning splits (this does not retrain either tokenizer):
```bash
python3 phase3/scripts/generate_reasoning_data.py
```

For a clean CPU environment, install the Phase 3 dependencies first:
```bash
python3 -m venv .venv_phase3
.venv_phase3/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv_phase3/bin/python -m pip install sentencepiece matplotlib seaborn
```

Download the Phase 2 checkpoints from the links above to `hindi/checkpoints/final.pt` and `nepali/checkpoints/final.pt`. Then run separate full finetuning jobs:
```bash
python3 phase3/scripts/finetune_reasoning.py --language hindi
python3 phase3/scripts/finetune_reasoning.py --language nepali
```

The best checkpoints are saved separately at `phase3/checkpoints/<language>/best.pt`; resume an interrupted run with `--resume phase3/checkpoints/<language>/last.pt`. Evaluate pretrained versus finetuned checkpoints and generate paired reasoning-prompt attention plots:
```bash
python3 phase3/scripts/evaluate_reasoning.py --language hindi
python3 phase3/scripts/evaluate_reasoning.py --language nepali
python3 phase3/scripts/compare_reasoning_attention.py --language hindi
python3 phase3/scripts/compare_reasoning_attention.py --language nepali
python3 phase3/scripts/plot_finetune_loss.py --language hindi
python3 phase3/scripts/plot_finetune_loss.py --language nepali
```

See `report/phase3_report.md` for the dataset controls, actual-command output locations, final results, and attention figures. The Phase 3 checkpoint folders linked above contain both `best.pt` and `last.pt` for each language.

## Final Submission Documents

- [Phase 1 report](report/phase1_report.md): collection, cleaning, splits, and tokenizer construction.
- [Phase 2 report](report/phase2_report.md): architecture, pretraining, language-modeling evaluation, generation, and attention analysis.
- [Phase 3 final report](report/phase3_report.md): reasoning finetuning, pretrained-versus-finetuned evaluation, post-finetuning attention comparison, cross-phase synthesis, and the final deliverable index.

The complete reproduction order is: **Stage 1 data collection → Stage 2 tokenizer training and split export → Stage 3 Phase 2 pretraining → Stage 4 Phase 2 evaluation → Stage 5 Phase 3 reasoning finetuning and analysis**. Large datasets and checkpoints are linked through Google Drive rather than committed to Git.

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

## Phase 2: Pretraining Results

Below is a snapshot of the final language-modeling and generation metrics computed on the held-out test sets. For a deep dive into the loss curves, architectural justification, attention analysis, and qualitative generation results, please read the full `report/phase2_report.md`.

| Metric | Hindi (Model H) | Nepali (Model L) |
|--------|-----------------|------------------|
| Cross-Entropy Loss | 4.0162 | 3.6299 |
| Perplexity (PPL) | 55.4911 | 37.7073 |
| Bits-Per-Byte (BPB)| 5.7942 | 5.2368 |
| BLEU-4 | 3.51 | 0.00 |
| chrF | 9.19 | 4.43 |
| ROUGE-L | 0.0612 | 0.0680 |

---

## Contact

Name - Raj k jain
Roll No. - 2025201036
Email -  raj.jain@students.iiit.ac.in
