import hashlib
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from pathlib import Path

WORKING_DIR = Path(".")
# Books dir should be at root
INPUT_BOOKS_DIR = Path("./books")

LANG_SANGRAHA_CODE = {"hindi": "hin", "nepali": "nep"}
TESSERACT_LANG = {"hindi": "hin", "nepali": "nep"}

WORD_TARGET_BUFFER = 1.10
PUBLIC_WORD_TARGET = int(500_000_000 * WORD_TARGET_BUFFER)  
MANUAL_CEILING = 100_000_000      
SANGRAHA_PROGRESS_EVERY = 50_000
CHUNK_TARGET_WORDS = 200 

# -- article scraping config --------------------------------------------
SCRAPE_DOMAINS = {
    "hindi": [
        "https://www.amarujala.com",
        "https://www.jagran.com",
        "https://www.bhaskar.com",
        "https://navbharattimes.indiatimes.com",
    ],
    "nepali": [
        "https://www.onlinekhabar.com",
        "https://www.setopati.com",
        "https://ekantipur.com",
        "https://www.ratopati.com",
    ],
}
SCRAPE_NUM_WORKERS = 20
SCRAPE_MAX_URLS_PER_DOMAIN = 5000

BOOK_METADATA = {
    "201 Prerak Neeti Kathayen (Hindi).pdf": {"title": "201 Prerak Neeti Kathayen", "author": "unknown - verify", "pub_year": "unknown - verify", "license_note": "unknown - verify redistribution rights"},
    "gaban.pdf": {"title": "Gaban", "author": "Munshi Premchand", "pub_year": "1931", "license_note": "public domain (author d. 1936) - verify jurisdiction"},
    "karmbhumi.pdf": {"title": "Karmabhoomi", "author": "Munshi Premchand", "pub_year": "1932", "license_note": "public domain (author d. 1936) - verify jurisdiction"},
    "nirmala.pdf": {"title": "Nirmala", "author": "Munshi Premchand", "pub_year": "1925", "license_note": "public domain (author d. 1936) - verify jurisdiction"},
    "16th_Plan.pdf": {"title": "16th Plan (Nepal)", "author": "National Planning Commission, Nepal", "pub_year": "unknown - verify", "license_note": "govt document - verify redistribution rights"},
    "NPC2076BS_15th_Plan_Approach.pdf": {"title": "15th Plan Approach Paper (NPC 2076 BS)", "author": "National Planning Commission, Nepal", "pub_year": "2019/2020", "license_note": "govt document - verify redistribution rights"},
    "RatoBangalaFoundation_Chandra...pdf": {"title": "unknown - verify exact filename/title", "author": "unknown - verify", "pub_year": "unknown - verify", "license_note": "unknown - verify redistribution rights"},
}


# --------------------------------------------------------------------------
# Database layer
# --------------------------------------------------------------------------
class LanguageDB:
    def __init__(self, language):
        self.language = language
        self.db_path = WORKING_DIR / language / f"{language}_state.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=30)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            source_id TEXT PRIMARY KEY,
            language TEXT, source_name TEXT, source_type TEXT,
            manual_or_downloaded TEXT,
            title TEXT, author TEXT, pub_year TEXT, license_note TEXT,
            location TEXT, collection_method TEXT, ocr_used INTEGER,
            access_date TEXT, docs_processed INTEGER DEFAULT 0,
            status TEXT DEFAULT 'in_progress', last_updated TEXT
        );

        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            source_id TEXT, language TEXT,
            raw_chars INTEGER, cleaned_chars INTEGER,
            content TEXT, split TEXT,
            hash_bucket INTEGER,             -- 0-99, uniform sampling key (Stage 2)
            tokenizer_token_count INTEGER,
            created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);
        CREATE INDEX IF NOT EXISTS idx_documents_split_bucket ON documents(split, hash_bucket);

        CREATE TABLE IF NOT EXISTS ocr_pages (
            source_id TEXT, page_num INTEGER, text TEXT, char_count INTEGER,
            extraction_method TEXT,
            PRIMARY KEY (source_id, page_num)
        );

        CREATE TABLE IF NOT EXISTS cleaning_stats (
            source_id TEXT PRIMARY KEY,
            lines_seen INTEGER DEFAULT 0,
            lines_dropped_short INTEGER DEFAULT 0,
            lines_dropped_nonscript INTEGER DEFAULT 0,
            lines_dropped_boilerplate INTEGER DEFAULT 0,
            docs_dropped_duplicate INTEGER DEFAULT 0
        );
        """)
        self.conn.commit()

    def upsert_source(self, source_id, **fields):
        fields["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cols = ", ".join(fields.keys())
        placeholders = ", ".join("?" for _ in fields)
        updates = ", ".join(f"{k}=excluded.{k}" for k in fields if k != "docs_processed")
        sql = f"""INSERT INTO sources (source_id, {cols}) VALUES (?, {placeholders})
                  ON CONFLICT(source_id) DO UPDATE SET {updates}"""
        self.conn.execute(sql, (source_id, *fields.values()))
        self.conn.commit()

    def get_source_cursor(self, source_id):
        row = self.conn.execute("SELECT docs_processed, status FROM sources WHERE source_id=?",
                                 (source_id,)).fetchone()
        return row if row else (0, None)

    def update_cursor(self, source_id, docs_processed):
        self.conn.execute("UPDATE sources SET docs_processed=? WHERE source_id=?",
                           (docs_processed, source_id))
        self.conn.commit()

    def mark_source_complete(self, source_id):
        self.conn.execute("UPDATE sources SET status='complete' WHERE source_id=?", (source_id,))
        self.conn.commit()

    def insert_document(self, source_id, cleaned_text, raw_chars):
        doc_id = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
        bucket = int(doc_id[:8], 16) % 100
        split = "train" if bucket < 90 else ("val" if bucket < 95 else "test")
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO documents "
            "(doc_id, source_id, language, raw_chars, cleaned_chars, content, split, hash_bucket, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (doc_id, source_id, self.language, raw_chars, len(cleaned_text), cleaned_text,
             split, bucket, time.strftime("%Y-%m-%d %H:%M:%S")),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def provisional_word_total(self):
        row = self.conn.execute(
            "SELECT COALESCE(SUM(LENGTH(content) - LENGTH(REPLACE(content,' ','')) + 1),0) FROM documents"
        ).fetchone()
        return row[0] or 0

    def get_cached_page(self, source_id, page_num):
        row = self.conn.execute(
            "SELECT text FROM ocr_pages WHERE source_id=? AND page_num=?", (source_id, page_num)
        ).fetchone()
        return row[0] if row else None

    def cache_page(self, source_id, page_num, text, method):
        self.conn.execute(
            "INSERT OR REPLACE INTO ocr_pages (source_id, page_num, text, char_count, extraction_method) "
            "VALUES (?,?,?,?,?)", (source_id, page_num, text, len(text), method),
        )
        self.conn.commit()

    def bump_cleaning_stat(self, source_id, field, n=1):
        self.conn.execute(
            f"INSERT INTO cleaning_stats (source_id, {field}) VALUES (?, ?) "
            f"ON CONFLICT(source_id) DO UPDATE SET {field} = {field} + excluded.{field}",
            (source_id, n),
        )
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()


# --------------------------------------------------------------------------
# Cleaning (line-level, shared by Sangraha docs and manual-book chunks)
# --------------------------------------------------------------------------
DEVANAGARI_RANGE = range(0x0900, 0x0980)
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PUNCT_RUN_RE = re.compile(r"([।॥.!?\-_=*~])\1{3,}")
PAGE_NUM_RE = re.compile(r"^\s*[\d०-९]{1,4}\s*$")
WS_RE = re.compile(r"[^\S\n]+")


def devanagari_ratio(line):
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return 1.0
    dev = sum(1 for c in letters if ord(c) in DEVANAGARI_RANGE)
    return dev / len(letters)


def clean_line(raw_line, db, source_id, boilerplate=None,
                min_line_chars=3, min_dev_ratio=0.15, nonscript_len_threshold=15):
    """Relaxed cleaning: short lines and mixed-script short lines survive;
    only clearly non-Devanagari LONG lines and detected boilerplate are cut."""
    line = unicodedata.normalize("NFC", raw_line)
    line = CONTROL_CHARS_RE.sub(" ", line)
    line = PUNCT_RUN_RE.sub(r"\1\1\1", line)
    line = WS_RE.sub(" ", line).strip()

    db.bump_cleaning_stat(source_id, "lines_seen")

    if not line or PAGE_NUM_RE.match(line):
        return None
    if boilerplate and line in boilerplate:
        db.bump_cleaning_stat(source_id, "lines_dropped_boilerplate")
        return None
    if len(line) < min_line_chars:
        db.bump_cleaning_stat(source_id, "lines_dropped_short")
        return None
    if len(line) > nonscript_len_threshold and devanagari_ratio(line) < min_dev_ratio:
        db.bump_cleaning_stat(source_id, "lines_dropped_nonscript")
        return None
    return line


def clean_text(raw_text, db, source_id):
    """Used for Sangraha documents (already paragraph/document-sized units)."""
    kept = [clean_line(l, db, source_id) for l in raw_text.splitlines()]
    return "\n".join(l for l in kept if l).strip()


# --------------------------------------------------------------------------
# Sangraha collection (unchanged in behavior, uses relaxed clean_text)
# --------------------------------------------------------------------------
def collect_sangraha(language, db, target_words=PUBLIC_WORD_TARGET):
    from datasets import load_dataset

    lang_code = LANG_SANGRAHA_CODE[language]
    source_id = f"sangraha_{lang_code}"

    db.upsert_source(
        source_id=source_id, language=language, source_name="ai4bharat/sangraha",
        source_type="sangraha", manual_or_downloaded="downloaded",
        title="Sangraha (verified split)", author="AI4Bharat", pub_year="2024",
        license_note="NOT VERIFIED HERE - check ai4bharat/sangraha dataset card "
                      "and underlying source licenses before submission",
        location=f"huggingface:verified/{lang_code}",
        collection_method="streamed via HF datasets, provisional word-count target",
        ocr_used=0, access_date=time.strftime("%Y-%m-%d"),
    )

    cursor, status = db.get_source_cursor(source_id)
    if status == "complete":
        print(f"[{language}] Sangraha already marked complete ({cursor} docs). Skipping.")
        return

    print(f"[{language}] Streaming Sangraha verified/{lang_code}, resuming after doc #{cursor}...")
    ds = load_dataset("ai4bharat/sangraha", data_dir=f"verified/{lang_code}", streaming=True, split="train")
    if cursor:
        ds = ds.skip(cursor)

    processed_this_run = 0
    docs_processed = cursor
    current_words = db.provisional_word_total()

    for item in ds:
        raw = item.get("text") or item.get("sentence") or item.get("content") or ""
        cleaned = clean_text(raw, db, source_id)
        docs_processed += 1
        processed_this_run += 1

        if cleaned:
            inserted = db.insert_document(source_id, cleaned, raw_chars=len(raw))
            if inserted:
                current_words += len(cleaned.split())
            else:
                db.bump_cleaning_stat(source_id, "docs_dropped_duplicate")

        if processed_this_run % SANGRAHA_PROGRESS_EVERY == 0:
            db.update_cursor(source_id, docs_processed)
            print(f"[{language}] {current_words:,}/{target_words:,} provisional words "
                  f"({docs_processed:,} docs seen)")

        if current_words >= target_words:
            db.update_cursor(source_id, docs_processed)
            db.mark_source_complete(source_id)
            print(f"[{language}] Reached provisional word target: {current_words:,} words.")
            return

    db.update_cursor(source_id, docs_processed)
    db.mark_source_complete(source_id)
    print(f"[{language}] Sangraha exhausted at {current_words:,} words "
          f"(target was {target_words:,}). This is a real shortfall — report it.")


# --------------------------------------------------------------------------
# Manual book collection — page-level OCR (resumable), then TWO STREAMING
# PASSES over the cached pages to produce chunked documents, never holding
# the whole book in memory.
# --------------------------------------------------------------------------
def extract_page(pdf_path, page_num, tess_lang, dpi=260):
    import fitz
    doc = fitz.open(str(pdf_path))
    if page_num > len(doc):
        doc.close()
        return None
    page = doc[page_num - 1]
    text = page.get_text()
    doc.close()

    if len(text.strip()) >= 20:
        return text, "pymupdf"

    from pdf2image import convert_from_path
    import pytesseract
    images = convert_from_path(str(pdf_path), dpi=dpi, first_page=page_num, last_page=page_num)
    if not images:
        return "", "tesseract_ocr"
    ocr_text = pytesseract.image_to_string(images[0], lang=tess_lang)
    del images
    return ocr_text, "tesseract_ocr"


def ocr_all_pages(language, db, pdf_path, source_id):
    """Pass 0: ensure every page is OCR'd and cached (resumable per page).
    Does NOT return page text — caller re-reads from the DB cache, so this
    function's memory footprint is one page at a time."""
    import fitz
    doc = fitz.open(str(pdf_path))
    n_pages = len(doc)
    doc.close()
    tess_lang = TESSERACT_LANG[language]

    cursor, _ = db.get_source_cursor(source_id)
    print(f"[{language}] {pdf_path.name}: {n_pages} pages, resuming OCR after page {cursor}...")
    for p in range(1, n_pages + 1):
        if db.get_cached_page(source_id, p) is not None:
            continue
        try:
            text, method = extract_page(pdf_path, p, tess_lang)
        except Exception as e:
            print(f"    ERROR page {p} of {pdf_path.name}: {e}")
            text, method = "", "error"
        db.cache_page(source_id, p, text or "", method)
        db.update_cursor(source_id, p)
        if p % 25 == 0:
            print(f"    {pdf_path.name}: page {p}/{n_pages} OCR'd ({method})")
    return n_pages


def compute_boilerplate(db, source_id, n_pages):
    """Pass 1: stream cached pages one at a time, keep only a line->count
    Counter (bounded by unique line count, not full text size)."""
    if n_pages < 5:
        return set()
    counts = Counter()
    for p in range(1, n_pages + 1):
        text = db.get_cached_page(source_id, p) or ""
        for line in set(l.strip() for l in text.splitlines() if l.strip()):
            counts[line] += 1
    threshold = max(3, int(n_pages * 0.4))
    return {line for line, c in counts.items() if c >= threshold}


def chunk_and_insert(db, source_id, n_pages, boilerplate, chunk_target_words=CHUNK_TARGET_WORDS):
    """Pass 2: stream cached pages one at a time, clean line-by-line, flush
    ~200-word chunks as independent documents as soon as the buffer fills."""
    buffer_lines, buffer_words = [], 0
    n_chunks_inserted, n_chunks_seen = 0, 0

    def flush():
        nonlocal buffer_lines, buffer_words, n_chunks_inserted, n_chunks_seen
        if not buffer_lines:
            return
        chunk_text = "\n".join(buffer_lines)
        n_chunks_seen += 1
        if db.insert_document(source_id, chunk_text, raw_chars=len(chunk_text)):
            n_chunks_inserted += 1
        else:
            db.bump_cleaning_stat(source_id, "docs_dropped_duplicate")
        buffer_lines, buffer_words = [], 0

    for p in range(1, n_pages + 1):
        raw_page_text = db.get_cached_page(source_id, p) or ""
        for raw_line in raw_page_text.splitlines():
            cleaned = clean_line(raw_line, db, source_id, boilerplate=boilerplate)
            if not cleaned:
                continue
            buffer_lines.append(cleaned)
            buffer_words += len(cleaned.split())
            if buffer_words >= chunk_target_words:
                flush()
    flush()  # final partial chunk

    return n_chunks_inserted, n_chunks_seen


def collect_manual_book(language, db, pdf_path):
    filename = pdf_path.name
    source_id = f"manual_book_{language}_{hashlib.sha1(filename.encode()).hexdigest()[:10]}"
    meta = BOOK_METADATA.get(filename, {
        "title": "unknown - verify", "author": "unknown - verify",
        "pub_year": "unknown - verify", "license_note": "unknown - verify redistribution rights",
    })

    db.upsert_source(
        source_id=source_id, language=language, source_name=filename,
        source_type="manual_book", manual_or_downloaded="manual",
        title=meta["title"], author=meta["author"], pub_year=meta["pub_year"],
        license_note=meta["license_note"], location=str(pdf_path),
        collection_method="PyMuPDF/Tesseract per-page OCR, chunked into "
                           f"~{CHUNK_TARGET_WORDS}-word paragraph documents",
        ocr_used=1, access_date=time.strftime("%Y-%m-%d"),
    )

    _, status = db.get_source_cursor(source_id)
    if status == "complete":
        print(f"[{language}] {filename} already complete. Skipping.")
        return

    n_pages = ocr_all_pages(language, db, pdf_path, source_id)
    boilerplate = compute_boilerplate(db, source_id, n_pages)
    inserted, seen = chunk_and_insert(db, source_id, n_pages, boilerplate)

    db.mark_source_complete(source_id)
    print(f"[{language}] {filename}: done. {inserted}/{seen} chunks inserted "
          f"({seen - inserted} exact duplicates dropped).")


def collect_manual_txt(language, db, txt_path):
    """Direct read + chunk, no OCR needed. Genuinely manual if this is text
    you personally gathered/typed/transcribed -- not a re-download of an
    existing public corpus (that would be 'downloaded', not manual)."""
    filename = txt_path.name
    source_id = f"manual_txt_{language}_{hashlib.sha1(filename.encode()).hexdigest()[:10]}"
    meta = BOOK_METADATA.get(filename, {
        "title": "unknown - verify", "author": "unknown - verify",
        "pub_year": "unknown - verify", "license_note": "unknown - verify redistribution rights",
    })

    db.upsert_source(
        source_id=source_id, language=language, source_name=filename,
        source_type="manual_txt", manual_or_downloaded="manual",
        title=meta["title"], author=meta["author"], pub_year=meta["pub_year"],
        license_note=meta["license_note"], location=str(txt_path),
        collection_method=f"direct text reading, chunked into ~{CHUNK_TARGET_WORDS}-word documents",
        ocr_used=0, access_date=time.strftime("%Y-%m-%d"),
    )

    _, status = db.get_source_cursor(source_id)
    if status == "complete":
        print(f"[{language}] {filename} already complete. Skipping.")
        return

    # Boilerplate detection across pseudo-pages (every 50 lines = one "page")
    # since plain text has no real page boundaries.
    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    pseudo_pages = [lines[i:i + 50] for i in range(0, len(lines), 50)]
    boilerplate = set()
    if len(pseudo_pages) >= 5:
        counts = Counter()
        for page in pseudo_pages:
            for line in set(l.strip() for l in page if l.strip()):
                counts[line] += 1
        threshold = max(3, int(len(pseudo_pages) * 0.4))
        boilerplate = {l for l, c in counts.items() if c >= threshold}

    buffer_lines, buffer_words, n_inserted, n_seen = [], 0, 0, 0

    def flush():
        nonlocal buffer_lines, buffer_words, n_inserted, n_seen
        if not buffer_lines:
            return
        chunk_text = "\n".join(buffer_lines)
        n_seen += 1
        if db.insert_document(source_id, chunk_text, raw_chars=len(chunk_text)):
            n_inserted += 1
        else:
            db.bump_cleaning_stat(source_id, "docs_dropped_duplicate")
        buffer_lines, buffer_words = [], 0

    for raw_line in lines:
        cleaned = clean_line(raw_line, db, source_id, boilerplate=boilerplate)
        if not cleaned:
            continue
        buffer_lines.append(cleaned)
        buffer_words += len(cleaned.split())
        if buffer_words >= CHUNK_TARGET_WORDS:
            flush()
    flush()

    db.mark_source_complete(source_id)
    print(f"[{language}] {filename}: done. {n_inserted}/{n_seen} chunks inserted.")


def collect_manual_books(language, db):
    books_dir = INPUT_BOOKS_DIR / language
    if not books_dir.exists():
        print(f"[{language}] No manual books dir at {books_dir}")
        return
    for path in sorted(books_dir.glob("*")):
        if path.suffix.lower() == ".pdf":
            collect_manual_book(language, db, path)
        elif path.suffix.lower() == ".txt":
            collect_manual_txt(language, db, path)


# --------------------------------------------------------------------------
# Manual article scraping (Phase B) -- reuses the same DB/chunking convention
# as manual books. Uses trafilatura for sitemap discovery + main-content
# extraction, since hand-written CSS selectors per site are too fragile to
# trust unverified. Genuinely manual: you wrote and ran this collection
# pipeline yourself, unlike a bulk public dump.
# --------------------------------------------------------------------------
def discover_urls(domain, max_urls):
    from trafilatura.sitemaps import sitemap_search
    try:
        return sitemap_search(domain, target_lang=None)[:max_urls]
    except Exception as e:
        print(f"    sitemap discovery failed for {domain}: {e}")
        return []


def fetch_and_extract(url):
    import trafilatura
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return url, None
        text = trafilatura.extract(downloaded, favor_precision=True,
                                    include_comments=False, include_tables=False)
        return url, text
    except Exception:
        return url, None


def scrape_articles(language, db, target_words):
    import concurrent.futures as cf

    total_words = 0
    for domain in SCRAPE_DOMAINS.get(language, []):
        if total_words >= target_words:
            break

        source_id = f"manual_scrape_{language}_{domain.split('//')[1].split('/')[0].replace('.', '_')}"
        db.upsert_source(
            source_id=source_id, language=language, source_name=domain,
            source_type="manual_scrape", manual_or_downloaded="manual",
            title=f"Scraped articles: {domain}", author="various (news site)",
            pub_year=time.strftime("%Y"),
            license_note="unknown - verify redistribution rights before submission",
            location=domain, collection_method="trafilatura sitemap discovery + main-content extraction",
            ocr_used=0, access_date=time.strftime("%Y-%m-%d"),
        )

        print(f"[{language}] Discovering URLs from {domain} ...")
        urls = discover_urls(domain, SCRAPE_MAX_URLS_PER_DOMAIN)
        print(f"[{language}] {len(urls)} URLs found at {domain}")

        with cf.ThreadPoolExecutor(max_workers=SCRAPE_NUM_WORKERS) as executor:
            futures = {executor.submit(fetch_and_extract, u): u for u in urls}
            for fut in cf.as_completed(futures):
                try:
                    _, text = fut.result()
                except Exception:
                    text = None
                if not text or len(text.split()) < 50:
                    continue

                buffer_lines, buffer_words = [], 0
                for raw_line in text.splitlines():
                    cleaned = clean_line(raw_line, db, source_id)
                    if not cleaned:
                        continue
                    buffer_lines.append(cleaned)
                    buffer_words += len(cleaned.split())
                    if buffer_words >= CHUNK_TARGET_WORDS:
                        chunk_text = "\n".join(buffer_lines)
                        if db.insert_document(source_id, chunk_text, raw_chars=len(chunk_text)):
                            total_words += buffer_words
                        else:
                            db.bump_cleaning_stat(source_id, "docs_dropped_duplicate")
                        buffer_lines, buffer_words = [], 0
                if buffer_lines:
                    chunk_text = "\n".join(buffer_lines)
                    if db.insert_document(source_id, chunk_text, raw_chars=len(chunk_text)):
                        total_words += buffer_words

                if total_words % 200000 < 5000:
                    print(f"[{language}] Scraping progress: ~{total_words:,}/{target_words:,} words")
                if total_words >= target_words:
                    break

        db.mark_source_complete(source_id)

    print(f"[{language}] Scraping phase done: ~{total_words:,} words added.")
    return total_words


# --------------------------------------------------------------------------
def collect_language(language):
    assert language in ("hindi", "nepali")
    db = LanguageDB(language)
    try:
        # Phase A: manual books (PDF via OCR, TXT via direct read) -- resumable, chunked
        collect_manual_books(language, db)

        # Phase B: manual article scraping, capped at MANUAL_CEILING total
        manual_words_so_far = db.conn.execute("""
            SELECT COALESCE(SUM(LENGTH(d.content) - LENGTH(REPLACE(d.content,' ','')) + 1),0)
            FROM documents d JOIN sources s ON d.source_id = s.source_id
            WHERE s.manual_or_downloaded = 'manual'
        """).fetchone()[0]
        remaining_manual_target = max(0, MANUAL_CEILING - manual_words_so_far)
        if remaining_manual_target > 0:
            print(f"[{language}] Manual so far (books): {manual_words_so_far:,} words. "
                  f"Scraping up to {remaining_manual_target:,} more (won't block Sangraha below)...")
            scrape_articles(language, db, remaining_manual_target)
        else:
            print(f"[{language}] Manual books already reached the {MANUAL_CEILING:,}-word ceiling.")

        # Phase C: Sangraha ALWAYS targets the full 500M -- collecting max
        # public data now costs little time and doesn't foreclose anything.
        # Whether trimming is needed is decided LATER (see trim_corpus.py),
        # once you know the final manual total close to your deadline.
        print(f"[{language}] Collecting Sangraha to the full target regardless of manual progress...")
        collect_sangraha(language, db, target_words=PUBLIC_WORD_TARGET)

        final_manual_words = db.conn.execute("""
            SELECT COALESCE(SUM(LENGTH(d.content) - LENGTH(REPLACE(d.content,' ','')) + 1),0)
            FROM documents d JOIN sources s ON d.source_id = s.source_id
            WHERE s.manual_or_downloaded = 'manual'
        """).fetchone()[0]
        words = db.provisional_word_total()
        print(f"\n[{language}] STAGE 1 COMPLETE. Total: {words:,} words "
              f"(manual: {final_manual_words:,}, "
              f"manual %: {final_manual_words/words*100 if words else 0:.2f}%)")
        print(f"[{language}] If manual % is still below 20%% at export time, "
              f"run trim_corpus.py to bring the TRAIN split into compliance "
              f"without deleting your extra downloaded data from the DB.")
        print(f"[{language}] DB saved at: {db.db_path}")
    finally:
        db.close()


if __name__ == "__main__":
    collect_language("nepali")
