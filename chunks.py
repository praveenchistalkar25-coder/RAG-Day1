"""
Newspaper OCR using Azure Computer Vision — Column-Aware + Semantic Chunked
------------------------------------------------------------------
1. Extracts text from image using Azure Computer Vision
2. Sorts text column-by-column (fixes row-mixing problem)
3. Chunks the output semantically at clause boundaries into ~500 token chunks with ~100 token overlap

Requirements:
    pip install azure-ai-vision-imageanalysis tiktoken spacy python-dotenv
    python -m spacy download en_core_web_sm

Setup:
    Set environment variables in .env file (AZURE_KEY, AZURE_ENDPOINT, etc.)
"""

from pathlib import Path
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import tiktoken
import json
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ─────────────────────────────────────────────────────
# CONFIGURATION — now from environment variables
# ─────────────────────────────────────────────────────

IMAGE_PATH  = os.getenv("IMAGE_PATH", r"C:\Users\Swamini\Downloads\samplenewsarticle.png")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", r"C:\Users\Swamini\Downloads\azure_ocr_chunks.txt")
JSON_FILE   = os.getenv("JSON_FILE", r"C:\Users\Swamini\Downloads\azure_ocr_chunks.json")

AZURE_KEY      = os.getenv("AZURE_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

# Column detection — set manually if auto-detect is wrong (e.g. NUM_COLUMNS = 3)
NUM_COLUMNS = int(os.getenv("NUM_COLUMNS", 0)) or None

# Chunking settings
CHUNK_SIZE_TOKENS    = int(os.getenv("CHUNK_SIZE_TOKENS", 500))   # target chunk size (within 400–600 range)
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", 100))   # overlap between chunks (within 80–120 range)
MAX_CHUNKS           = int(os.getenv("MAX_CHUNKS", 10))    # hard limit on the total number of chunks

# Tokenizer model — cl100k_base is used by GPT-4, Claude, most modern LLMs
TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", "cl100k_base")

# ─────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════
#  PART 1 — AZURE OCR + COLUMN SORTING
# ══════════════════════════════════════════════════════

def get_line_bbox(line) -> dict:
    pts = line.bounding_polygon
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return {
        "x":     min(xs),
        "y":     min(ys),
        "x_end": max(xs),
        "y_end": max(ys),
        "text":  line.text
    }


def detect_columns(lines: list, image_width: int, num_columns=None) -> int:
    if num_columns:
        return num_columns
    if not lines:
        return 1
    x_starts = sorted([l["x"] for l in lines])
    gaps = []
    for i in range(1, len(x_starts)):
        gap = x_starts[i] - x_starts[i - 1]
        if gap > image_width * 0.05:
            gaps.append(x_starts[i])
    unique_gaps = sorted(set(round(g / 50) * 50 for g in gaps))
    n_cols = len(unique_gaps) + 1
    print(f"   Auto-detected {n_cols} column(s)")
    return max(1, min(n_cols, 6))


def assign_columns(lines: list, image_width: int, num_columns: int) -> list:
    col_width = image_width / num_columns
    for line in lines:
        col_idx = int(line["x"] // col_width)
        line["column"] = min(col_idx, num_columns - 1)
    return lines


def extract_and_sort(image_path: str, key: str, endpoint: str) -> str:
    """Call Azure, detect columns, return clean column-sorted text."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"\n📂 Loading: {path.name}  ({path.stat().st_size / 1024:.1f} KB)")

    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    with open(image_path, "rb") as f:
        image_data = f.read()

    print("🔍 Sending to Azure Computer Vision...")

    result = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ],
        language="en"
    )

    if not result.read or not result.read.blocks:
        raise ValueError("No text detected in the image.")

    all_lines = []
    for block in result.read.blocks:
        for line in block.lines:
            all_lines.append(get_line_bbox(line))

    print(f"   Found {len(all_lines)} lines of text")

    image_width = max(l["x_end"] for l in all_lines)
    n_cols = detect_columns(all_lines, image_width, NUM_COLUMNS)
    all_lines = assign_columns(all_lines, image_width, n_cols)

    # Sort: column first, then top-to-bottom within each column
    sorted_lines = sorted(all_lines, key=lambda l: (l["column"], l["y"]))

    # Build plain text — each column separated by a blank line
    output = []
    current_col = -1
    for line in sorted_lines:
        if line["column"] != current_col:
            current_col = line["column"]
            if output:
                output.append("")   # blank line between columns
            output.append(f"=== COLUMN {current_col + 1} ===")
        output.append(line["text"])

    full_text = "\n".join(output)
    print(f"   Extracted {len(full_text)} characters total")
    return full_text


# ══════════════════════════════════════════════════════
#  PART 2 — TOKEN CHUNKING
# ══════════════════════════════════════════════════════

def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


def chunk_text(
    text: str,
    chunk_size: int   = CHUNK_SIZE_TOKENS,
    overlap: int      = CHUNK_OVERLAP_TOKENS,
    model: str        = TOKENIZER_MODEL,
    max_chunks: int   = MAX_CHUNKS
) -> list[dict]:
    """
    Split text into semantic chunks limited to a fixed maximum count.

    Strategy:
      - Use spaCy to split text into clauses
      - Group clauses into up to max_chunks
      - Adjust target token size so the entire document fits in max_chunks
      - Preserve clause boundaries and provide rich chunk metadata

    Returns a list of chunk dicts.
    """
    import math
    import spacy
    import re
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy model 'en_core_web_sm' not found. Installing...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    clauses = []
    for sent in doc.sents:
        sent_text = sent.text
        parts = re.split(r'(\s*[ ,;:]+\s*)', sent_text)
        current_clause = ""
        for part in parts:
            current_clause += part
            if re.search(r'[ ,;:]$', part.strip()) and current_clause.strip():
                clauses.append(current_clause.strip())
                current_clause = ""
        if current_clause.strip():
            clauses.append(current_clause.strip())

    enc = tiktoken.get_encoding(model)
    clause_tokens = [enc.encode(c) for c in clauses]
    clause_token_counts = [len(t) for t in clause_tokens]
    total_tokens = sum(clause_token_counts)
    if total_tokens > 0 and max_chunks:
        avg_target = math.ceil(total_tokens / max_chunks)
        chunk_size = min(chunk_size, avg_target)
    else:
        avg_target = chunk_size

    max_overlap = max(0, chunk_size - avg_target)
    effective_overlap = min(overlap, max(0, chunk_size // 2), max_overlap)
    if effective_overlap != overlap:
        print(f"   Adjusted overlap from {overlap} to {effective_overlap} tokens to fit max_chunks")

    print(f"\n📊 Total tokens in extracted text: {total_tokens}")
    print(f"   Using up to {max_chunks} chunks")
    print(f"   Effective chunk target: {chunk_size} tokens | Overlap: {effective_overlap} tokens")
    print(f"   Detected {len(clauses)} clauses")

    chunks = []
    start_clause = 0
    chunk_idx = 1
    while start_clause < len(clauses):
        if max_chunks and chunk_idx == max_chunks:
            end_clause = len(clauses)
        else:
            current_token_count = 0
            end_clause = start_clause
            while end_clause < len(clauses) and current_token_count < chunk_size:
                current_token_count += clause_token_counts[end_clause]
                end_clause += 1
            if end_clause == start_clause:
                end_clause = min(start_clause + 1, len(clauses))

        chunk_clauses = clauses[start_clause:end_clause]
        chunk_text_str = " ".join(chunk_clauses).strip()
        token_start = sum(clause_token_counts[:start_clause])
        chunk_tokens = enc.encode(chunk_text_str)
        token_end = token_start + len(chunk_tokens)
        prefix_text = enc.decode(enc.encode(text)[:token_start]) if token_start else ""
        char_start = len(prefix_text)
        char_end = char_start + len(chunk_text_str)

        chunks.append({
            "chunk_index":  chunk_idx,
            "token_start":  token_start,
            "token_end":    token_end,
            "token_count":  len(chunk_tokens),
            "char_start":   char_start,
            "char_end":     char_end,
            "clause_count": len(chunk_clauses),
            "text":         chunk_text_str
        })

        print(f"   Chunk {chunk_idx:>2}: {len(chunk_clauses)} clauses, {len(chunk_tokens)} tokens")
        chunk_idx += 1

        if chunk_idx > max_chunks or end_clause >= len(clauses):
            break

        if effective_overlap > 0:
            overlap_tokens_needed = effective_overlap
            temp_count = 0
            overlap_clauses = 0
            for j in range(end_clause - 1, start_clause - 1, -1):
                temp_count += clause_token_counts[j]
                overlap_clauses += 1
                if temp_count >= overlap_tokens_needed:
                    break
            start_clause = max(start_clause, end_clause - overlap_clauses)
        else:
            start_clause = end_clause

    print(f"\n   ✅ Created {len(chunks)} chunk(s) from {total_tokens} tokens")
    return chunks


# ══════════════════════════════════════════════════════
#  PART 3 — SAVE OUTPUTS
# ══════════════════════════════════════════════════════

def save_txt(chunks: list[dict], output_path: str) -> None:
    """Save all chunks to a readable .txt file."""
    lines = []
    lines.append("NEWSPAPER OCR — CHUNKED OUTPUT")
    lines.append(f"Total chunks: {len(chunks)}")
    lines.append("=" * 60)

    for chunk in chunks:
        lines.append(f"\n{'─' * 60}")
        lines.append(f"CHUNK {chunk['chunk_index']}  | tokens {chunk['token_start']}–{chunk['token_end']}  | {chunk['token_count']} tokens")
        lines.append(f"CHARS {chunk['char_start']}–{chunk['char_end']}  | clauses {chunk['clause_count']}")
        lines.append(f"{'─' * 60}")
        lines.append(chunk["text"])

    lines.append(f"\n{'=' * 60}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"💾 TXT saved → {output_path}")


def save_json(chunks: list[dict], output_path: str) -> None:
    """Save chunks as JSON — ready for embedding pipelines (LangChain, LlamaIndex, etc.)."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"💾 JSON saved → {output_path}")


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Azure OCR  →  Column Sort  →  Semantic Chunker")
    print("=" * 60)

    if not AZURE_KEY or not AZURE_ENDPOINT:
        logging.error("AZURE_KEY and AZURE_ENDPOINT must be set in .env file.")
        print("\n❌ ERROR: Set AZURE_KEY and AZURE_ENDPOINT in .env file.")
        return

    try:
        # Step 1 — Extract and sort
        logging.info("Starting OCR extraction and sorting.")
        full_text = extract_and_sort(IMAGE_PATH, AZURE_KEY, AZURE_ENDPOINT)

        # Step 2 — Chunk
        logging.info("Starting semantic chunking.")
        chunks = chunk_text(full_text, max_chunks=MAX_CHUNKS)

        # Step 3 — Preview first 2 chunks in terminal
        print("\n" + "=" * 60)
        print("  PREVIEW — First 2 Chunks")
        print("=" * 60)
        for chunk in chunks[:2]:
            print(f"\n── Chunk {chunk['chunk_index']} ({chunk['token_count']} tokens) ──")
            print(chunk["text"])

        # Step 4 — Save
        print()
        save_txt(chunks, OUTPUT_FILE)
        save_json(chunks, JSON_FILE)

        logging.info(f"Successfully processed {len(chunks)} chunks.")
        print("\n✅ All done!")
        print(f"   → Readable output : {OUTPUT_FILE}")
        print(f"   → JSON for RAG    : {JSON_FILE}")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        print(f"\n❌ {e}")
    except ValueError as e:
        logging.error(f"Value error: {e}")
        print(f"\n❌ {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
