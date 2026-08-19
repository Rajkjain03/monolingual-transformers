# Phase 1 Report: Data Collection & Tokenizer Construction
Name - Raj k jain
Roll No. - 2025201036
Email -  raj.jain@students.iiit.ac.in   

## 1. Language Selection

### Model H — Hindi (Higher-Resource)
Hindi is one of the most widely spoken languages in India with over 600 million speakers. It is written in the Devanagari script and has abundant public text resources available through curated datasets like AI4Bharat's Sangraha corpus. The availability of large-scale verified data makes it an ideal choice for the higher-resource model, allowing us to comfortably reach the ~500M training token target.

### Model L — Nepali (Lower-Resource)
Nepali was selected from the allowed lower-resource language list. While Nepali shares the Devanagari script with Hindi, it has distinct grammar, vocabulary, and morphological patterns. Nepali has comparatively fewer public text resources than Hindi, though the AI4Bharat Sangraha corpus provides a verified Nepali split that enables reaching the token target. The shared script allows for interesting comparisons in tokenizer behavior while maintaining complete linguistic independence between the two models.

---

## 2. Data Sources

### 2.1 Downloaded Data
The primary downloaded data source is the **AI4Bharat Sangraha** corpus (`ai4bharat/sangraha` on Hugging Face).

| Language | Sangraha Subset | Documents | Tokens |
|----------|----------------|-----------|--------|
| Hindi | `verified/hin` | 1,292,199 | 673,660,474 |
| Nepali | `verified/nep` | 1,835,390 | 775,632,844 |

The data was collected via Hugging Face `datasets` library in streaming mode to avoid downloading the entire dataset at once.

### 2.2 Manual Data
Manual data consists of manually downloaded PDF books (processed via PyMuPDF and OCR) and scraped news articles (via Trafilatura).

#### Hindi Manual Sources
| Source Type | Documents | Total Tokens |
|-------------|-----------|--------------|
| Manual PDF Books (Premchand, etc.) | 1,421 | 411,198 |
| Manual Text Books/Archives | 4,725 | 1,173,346 |
| Manual Web Scraping (News) | 24,771 | 5,577,880 |
| **Total Hindi Manual** | **30,917** | **7,162,424** |

#### Nepali Manual Sources
| Source Type | Documents | Total Tokens |
|-------------|-----------|--------------|
| Manual PDF Books (Govt Plans, etc) | 902 | 360,123 |
| **Total Nepali Manual** | **902** | **360,123** |

**Extraction Method:** PDF text extraction was performed using PyMuPDF (fitz) for text-layer PDFs, with pytesseract as fallback for scanned pages. Web scraping was conducted using Trafilatura for precise main-content extraction. Documents were chunked into ~200-word paragraph-level pieces to ensure consistent lengths.

### 2.3 Manual Data Fraction (Train Split)

| Language | Manual Tokens (Train) | Total Tokens (Train) | Manual % | Target | Status |
|----------|-----------------------|----------------------|----------|--------|--------|
| Hindi | 6,777,503 | 613,059,508 | 1.11% | ≥ 20.0% | ❌ Below target |
| Nepali | 322,028 | 698,466,193 | 0.05% | ≥ 20.0% | ❌ Below target |

### 2.4 Known Limitations

**Data Fraction Shortfall:**
The manual data fraction is currently below the 20% target for both languages (Hindi 1.11%, Nepali 0.05%). The Sangraha corpus provided hundreds of millions of tokens, heavily skewing the ratio. Finding freely redistributable books in Hindi and Nepali with clean Devanagari text is challenging, and the Phase 1 deadline limits manual collection time. I can work on this in future phases.

## 3. Data Cleaning Pipeline

The cleaning pipeline applies identical logic to both languages, operating at the line level within each document.

### 3.1 Cleaning Steps (in order)
1. **Unicode Normalization (NFC):** All text is normalized to NFC form.
2. **Whitespace Stripping:** Leading/trailing whitespace removed.
3. **Boilerplate Removal:** For manual books, a two-pass approach is used. Pass 1 builds a line-frequency counter. Pass 2 drops lines appearing with high frequency (headers, footers, page numbers).
4. **Short Line Filtering:** Lines shorter than 3 characters are dropped (`min_line_chars=3`).
5. **Non-Script Filtering:** Lines longer than 15 characters with Devanagari character ratio below 0.15 are dropped. 
6. **Document-Level Deduplication:** SHA-256 hash of the full cleaned document text is used as the document ID. Duplicate documents are silently dropped via `INSERT OR IGNORE`.

---

## 4. Train / Validation / Test Splits

### 4.1 Split Method
Documents are assigned to splits using a **deterministic hash-bucket approach**:
- Compute SHA-256 hash of each document's cleaned text
- Map to a bucket in [0, 99] using `int(hash_hex[:8], 16) % 100`
- Assign: **train** (bucket < 90), **val** (90 ≤ bucket < 95), **test** (bucket ≥ 95)

### 4.2 Split Statistics

| Language | Train Split Tokens | Val + Test Split Tokens | Total Tokens |
|----------|--------------------|-------------------------|--------------|
| Hindi | 613,059,508 | ~68,140,928 | 681,200,436 |
| Nepali | 698,466,193 | ~77,526,774 | 775,992,967 |

The train split exceeds the ~500M training token target perfectly for both languages.

---

## 5. Tokenizer Training

### 5.1 Configuration

| Parameter | Hindi | Nepali |
|-----------|-------|--------|
| Algorithm | SentencePiece (BPE) | SentencePiece (BPE) |
| Vocabulary size | 32,000 | 16,000 |
| Training sample | 120,000 documents | 120,000 documents |
| Character Coverage | 99.98% | 99.98% |

### 5.2 Tokenization Statistics (Held-out Validation Set)

The tokenizers were evaluated on a 10,000-line sample from their respective held-out validation splits (`val.txt`) to measure token fertility and unknown-token rates.

| Metric | Hindi | Nepali |
|--------|-------|--------|
| **Average Chars / Token** | 4.12 | 4.59 |
| **Fertility (Tokens/Word)** | 1.24 | 1.40 |
| **Unknown Tokens (UNK Rate)** | 0.0485% | 0.1585% |

**Vocab Size Justification:** 
Hindi was given a larger vocabulary (32K) due to its larger overall corpus size and to keep its fertility rate low. Nepali was assigned a smaller vocabulary (16K), yet it maintained a very acceptable UNK rate of 0.15% and a healthy fertility of 1.40 tokens/word, confirming that 16K is highly optimal for the available Nepali corpus.

### 5.3 Token Frequency Statistics

**Top 5 most frequent tokens (excluding whitespace/punctuation):**
- **Hindi:** `▁के`, `▁है`, `▁में`, `▁की`, `▁से`
- **Nepali:** `▁छ`, `▁र`, `को`, `▁पनि`, `मा`

*(Note: `▁` represents the SentencePiece space character ` `)*

### 5.4 Tokenization Examples

**Hindi Example:**
> **Original:** इसके अलावा भारी मशीनों के इस्तेमाल से पहाड़ी ढलानों की स्थिरता प्रभावित हो रही है
> 
> **Tokens:** `['▁इसके', '▁अलावा', '▁भारी', '▁मशीनों', '▁के', '▁इस्तेमाल', '▁से', '▁पहाड़ी', '▁ढल', 'ानों', '▁की', '▁स्थिरता', '▁प्रभावित', '▁हो', '▁रही', '▁है']`

**Nepali Example:**
> **Original:** सामाजिक सुरक्षा भत्ताको इतिहास : १०० देखि ४ हजारसम्म, कसको पालामा कति बढ्यो ?
> 
> **Tokens:** `['▁सामाजिक', '▁सुरक्षा', '▁भत्', 'ताको', '▁इतिहास', '▁:', '▁१००', '▁देखि', '▁४', '▁हजार', 'सम्म', ',', '▁कसको', '▁पालामा', '▁कति', '▁बढ्यो', '▁?']`

### 5.5 Storage and Data Management

All data was stored in local SQLite databases (`hindi_state.db` and `nepali_state.db`) during processing. The final train/val/test splits were exported as plain text files (one document per line) to their respective language directories.

| File | Size | Storage |
|------|------|---------|
| `hindi_state.db` | 8.0 GB | Google Drive |
| `nepali_state.db` | 11.0 GB | Google Drive |
| Hindi text splits (train+val+test) | 6.7 GB | Google Drive |
| Nepali text splits (train+val+test) | 8.9 GB | Google Drive |
| `hindi_tokenizer.model/.vocab` | 1.8 MB | Git repository |
| `nepali_tokenizer.model/.vocab` | 1.1 MB | Git repository |
