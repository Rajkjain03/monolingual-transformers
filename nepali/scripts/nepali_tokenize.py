"""
STAGE 2 — LOCAL VERSION. Same logic as tokenize_and_report_v2.py, but paths
are configurable via CLI args instead of hardcoded to /kaggle/working, since
this now runs on your own machine against the downloaded .db files.

Usage:
    python tokenize_and_report_local.py --lang hindi  --db /path/to/hindi_state.db  --out_dir ./output --vocab_size 32000
    python tokenize_and_report_local.py --lang nepali --db /path/to/nepali_state.db --out_dir ./output --vocab_size 16000
"""

import argparse
import csv
import sqlite3
from pathlib import Path

REQUIRED_MANUAL_PCT = 20.0

# Module-level so ProcessPoolExecutor workers (separate processes) can import
# and call these directly. Each worker loads its own SentencePieceProcessor
# once (via initializer) and reuses it across all batches it's given.
_WORKER_SP = None


def _init_worker(sp_model_path):
    global _WORKER_SP
    import sentencepiece as spm
    _WORKER_SP = spm.SentencePieceProcessor(model_file=sp_model_path)


def _encode_batch(batch):
    global _WORKER_SP
    return [(len(_WORKER_SP.encode(content, out_type=int)), doc_id) for doc_id, content in batch]


def train_tokenizer(language, db_file, out_dir, vocab_size, max_sample_docs=120_000):
    import sentencepiece as spm

    conn = sqlite3.connect(db_file)
    sample_path = out_dir / "tokenizer" / f"{language}_tokenizer_train_sample.txt"
    sample_path.parent.mkdir(parents=True, exist_ok=True)

    manual_count = 0
    downloaded_count = 0
    for sid, m_or_d in conn.execute("SELECT source_id, manual_or_downloaded FROM sources"):
        count = conn.execute("SELECT COUNT(*) FROM documents WHERE source_id=? AND split='train'", (sid,)).fetchone()[0]
        if m_or_d == 'manual':
            manual_count += count
        else:
            downloaded_count += count

    manual_count = min(manual_count, max_sample_docs)
    remaining_budget = max(0, max_sample_docs - manual_count)
    cutoff = 100 if downloaded_count <= remaining_budget else \
        max(1, int(remaining_budget / downloaded_count * 100))

    print(f"[{language}] Tokenizer sample: {manual_count:,} manual train docs "
          f"+ ~{cutoff}% hash-bucket sample of {downloaded_count:,} downloaded train docs.")

    n_manual, n_downloaded = 0, 0
    with open(sample_path, "w", encoding="utf-8") as f:
        manual_sources = [r[0] for r in conn.execute("SELECT source_id FROM sources WHERE manual_or_downloaded='manual'")]
        for sid in manual_sources:
            if n_manual >= manual_count: break
            for (content,) in conn.execute("SELECT content FROM documents WHERE source_id=? AND split='train'", (sid,)):
                if n_manual >= manual_count: break
                f.write(content.replace("\n", " ") + "\n")
                n_manual += 1

        downloaded_sources = [r[0] for r in conn.execute("SELECT source_id FROM sources WHERE manual_or_downloaded='downloaded'")]
        for sid in downloaded_sources:
            if n_manual + n_downloaded >= max_sample_docs: break
            for (content,) in conn.execute("SELECT content FROM documents WHERE source_id=? AND split='train' AND hash_bucket < ?", (sid, cutoff)):
                f.write(content.replace("\n", " ") + "\n")
                n_downloaded += 1
                if n_manual + n_downloaded >= max_sample_docs: break
    conn.close()

    print(f"[{language}] Sample written: {n_manual:,} manual + {n_downloaded:,} downloaded "
          f"= {n_manual + n_downloaded:,} total docs. Training SentencePiece (vocab={vocab_size})...")

    import os
    num_cpus = os.cpu_count() or 1

    model_prefix = str(out_dir / "tokenizer" / f"{language}_tokenizer")
    spm.SentencePieceTrainer.train(
        input=str(sample_path), model_prefix=model_prefix, vocab_size=vocab_size,
        model_type="bpe", character_coverage=0.9998,
        input_sentence_size=max_sample_docs, shuffle_input_sentence=True,
        num_threads=num_cpus,  # was 0, but spm requires >= 1
        max_sentence_length=4000,  # caps memory spikes from outlier-long documents
    )
    print(f"[{language}] Tokenizer saved: {model_prefix}.model / .vocab")
    return f"{model_prefix}.model"


def compute_true_token_counts(language, db_file, sp_model_path, batch_size=1000, num_workers=None):
    """Parallelized across CPU cores via ProcessPoolExecutor -- encoding is
    embarrassingly parallel (each document independent), so this is the
    single biggest speedup available on a multi-core local machine."""
    import os
    import concurrent.futures as cf

    num_workers = num_workers or max(1, min(4, (os.cpu_count() or 2) - 1))
    print(f"[{language}] Using {num_workers} worker process(es) for encoding "
          f"(lower --num_workers if RAM is tight).")

    conn = sqlite3.connect(db_file)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -100000")  # ~100MB page cache, speeds up streaming reads

    total = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE tokenizer_token_count IS NULL"
    ).fetchone()[0]
    print(f"[{language}] Encoding {total:,} documents for true token counts...")

    done = 0
    with cf.ProcessPoolExecutor(max_workers=num_workers,
                                 initializer=_init_worker, initargs=(sp_model_path,)) as executor:
        while True:
            rows = conn.execute(
                "SELECT doc_id, content FROM documents WHERE tokenizer_token_count IS NULL LIMIT ?",
                (batch_size * num_workers,),
            ).fetchall()
            if not rows:
                break

            chunk_size = max(1, len(rows) // num_workers)
            chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
            futures = [executor.submit(_encode_batch, c) for c in chunks]

            updates = []
            for fut in futures:
                updates.extend(fut.result())
            conn.executemany("UPDATE documents SET tokenizer_token_count=? WHERE doc_id=?", updates)
            conn.commit()

            done += len(rows)
            if done % 50000 < batch_size * num_workers:
                print(f"[{language}] Tokenized {done:,}/{total:,} documents...")

    conn.close()
    print(f"[{language}] True token counts computed for all documents.")


def export_splits_and_report(language, db_file, out_dir):
    conn = sqlite3.connect(db_file)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir = out_dir.parent / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    split_files = {s: open(data_dir / f"{language}_{s}.txt", "w", encoding="utf-8")
                   for s in ("train", "val", "test")}
    for split, f in split_files.items():
        for (content,) in conn.execute("SELECT content FROM documents WHERE split=?", (split,)):
            f.write(content.replace("\n", " ") + "\n")
        f.close()
    print(f"[{language}] Split files written to {data_dir}")

    sources = conn.execute("SELECT source_id, manual_or_downloaded, source_name, title, author, pub_year, license_note, ocr_used FROM sources").fetchall()
    
    train_totals = {"manual": 0, "downloaded": 0}
    overall_total = 0
    source_rows = []
    
    for s_row in sources:
        sid, m_or_d, source_name, title, author, pub_year, license_note, ocr_used = s_row
        
        doc_stats = conn.execute("""
            SELECT COUNT(doc_id), SUM(tokenizer_token_count), SUM(raw_chars), SUM(cleaned_chars)
            FROM documents WHERE source_id = ?
        """, (sid,)).fetchone()
        c_docs, s_tok, s_raw, s_clean = doc_stats
        
        source_rows.append((
            sid, source_name, m_or_d, title, author, pub_year, license_note, ocr_used,
            c_docs or 0, s_tok or 0, s_raw or 0, s_clean or 0
        ))
        overall_total += (s_tok or 0)
        
        train_toks = conn.execute("""
            SELECT SUM(tokenizer_token_count) FROM documents
            WHERE source_id = ? AND split = 'train'
        """, (sid,)).fetchone()[0] or 0
        
        train_totals[m_or_d] += train_toks

    train_manual = train_totals.get("manual", 0)
    train_downloaded = train_totals.get("downloaded", 0)
    train_total = train_manual + train_downloaded
    manual_pct = (train_manual / train_total * 100) if train_total else 0.0
    required_manual = train_total * REQUIRED_MANUAL_PCT / 100
    shortfall = max(0, required_manual - train_manual)
    satisfied = manual_pct >= REQUIRED_MANUAL_PCT

    clean_rows = conn.execute("SELECT * FROM cleaning_stats").fetchall()
    clean_cols = [d[0] for d in conn.execute("SELECT * FROM cleaning_stats").description]

    report_csv = report_dir / f"{language}_source_report.csv"
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "source_name", "manual_or_downloaded", "title", "author",
                    "pub_year", "license_note", "ocr_used", "n_documents",
                    "true_tokenizer_tokens", "raw_chars", "cleaned_chars"])
        w.writerows(source_rows)
    print(f"[{language}] Source-level report written to {report_csv}")

    conn.close()

    report_lines = []
    report_lines.append("\n" + "=" * 55)
    report_lines.append(f"{language.upper()} DATASET — TRAIN-SPLIT ONLY (used for pretraining)")
    report_lines.append("=" * 55)
    report_lines.append(f"Downloaded tokenizer tokens (train) : {train_downloaded:,}")
    report_lines.append(f"Manual tokenizer tokens (train)     : {train_manual:,}")
    report_lines.append(f"Total tokenizer tokens (train)      : {train_total:,}")
    report_lines.append(f"Manual percentage (train)           : {manual_pct:.2f}%")
    report_lines.append(f"Required manual percentage          : {REQUIRED_MANUAL_PCT:.0f}%")
    report_lines.append(f"Requirement satisfied                : {'YES' if satisfied else 'NO'}")
    if not satisfied:
        report_lines.append(f"Additional manual train tokens needed: ~{shortfall:,.0f}")
    if train_total < 500_000_000:
        report_lines.append(f"NOTE: train total ({train_total:,}) is below the 500M target — "
              f"report this shortfall explicitly per the assignment's allowance.")
    report_lines.append("-" * 55)
    report_lines.append(f"(FYI) Overall corpus incl. val+test : {overall_total:,} tokens "
          f"— NOT used for the 20% check above")
    report_lines.append("=" * 55)

    report_lines.append("\nCleaning statistics per source:")
    for row in clean_rows:
        report_lines.append(str(dict(zip(clean_cols, row))))
        
    report_text = "\n".join(report_lines)
    print(report_text)
    
    summary_path = report_dir / f"{language}_summary_report.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\n[{language}] Summary report saved to {summary_path}")


def optimize_database(db_file):
    print("Ensuring database indexes exist (this may take a minute if running for the first time)...")
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_split ON documents(split)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_source_id ON documents(source_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_tok_count ON documents(tokenizer_token_count)")
    conn.commit()
    conn.close()


def run(language, db_file, out_dir, vocab_size, max_sample_docs, num_workers):
    out_dir.mkdir(parents=True, exist_ok=True)
    optimize_database(db_file)
    sp_model = train_tokenizer(language, db_file, out_dir, vocab_size, max_sample_docs=max_sample_docs)
    compute_true_token_counts(language, db_file, sp_model, num_workers=num_workers)
    export_splits_and_report(language, db_file, out_dir)


if __name__ == "__main__":
    run("nepali", Path("nepali/nepali_state.db"), Path("nepali"), 16000, 120_000, None)