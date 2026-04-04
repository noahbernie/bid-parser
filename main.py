from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pdfplumber
import fitz  # PyMuPDF
import anthropic
import io
import uuid
import os
import json
import time
import asyncio
import base64
import urllib.request
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

from highway.routes import router as highway_router
app.include_router(highway_router)

documents = {}

CHUNK_CHAR_LIMIT = 20000  # ~5k tokens input → response fits in 32k output tokens

HEADER_PROMPT = """You are parsing a road construction bid document. Extract project-level fields only (no street list yet).
Return ONLY valid JSON with these fields:
- bid_number, project_name, city, work_type, estimated_cost, bid_due_date
Use null for any field not found."""

_STREETS_PROMPT_BASE = """Map the table columns to these fields:
- main_street: THE STREET BEING WORKED ON — the PRIMARY street being paved, sealed, or repaired. Always in the FIRST column of the table. Copy it exactly as written — do not substitute or infer a different street name.
- from_street: where the work segment BEGINS. Labeled START, FROM, BEGIN, LIMITS FROM, or similar. Copy exactly as written. It is okay if this is the same name as main_street or to_street.
- to_street: where the work segment ENDS. Labeled END, TO, TERMINUS, LIMITS TO, or similar. Copy exactly as written. It is okay if this is the same name as main_street or from_street.
- work_type: the type of work — use the table section header/title if no explicit column (e.g. "SLURRY/CAPE SEAL LIST" → "Slurry/Cape Seal", "CRACK FILL/REPAIR ONLY LIST" → "Crack Fill/Repair")
- location: any location number or zone identifier if present
- source: ALWAYS set to "{SOURCE_TAG}" for every street you extract — do not change this value

CRITICAL RULES:
1. Copy street names EXACTLY as they appear in the table. Do not rename, reorder, or substitute values between columns.
2. The typical column order left-to-right is: Street Name | From | To | Work Type. Even on continuation pages with no header row, use this order.
3. Read the header row carefully to confirm column order before extracting data rows.
4. Extract every single row. Do not skip any. Each data row = one street object."""

STREETS_PROMPT_TEXT = """You are parsing pages from a road construction bid document. Extract ALL street segments from any tables or lists on these pages.
Return ONLY valid JSON: {"streets": [...]}
Each street object: {"main_street": "...", "from_street": "...", "to_street": "...", "work_type": "...", "location": "...", "source": "text"}

""" + _STREETS_PROMPT_BASE.replace("{SOURCE_TAG}", "text")

STREETS_PROMPT_IMAGE = """You are parsing a scanned table image from a road construction bid document. Extract ALL street segments visible in the image.
Return ONLY valid JSON: {"streets": [...]}
Each street object: {"main_street": "...", "from_street": "...", "to_street": "...", "work_type": "...", "location": "...", "source": "image"}

""" + _STREETS_PROMPT_BASE.replace("{SOURCE_TAG}", "image") + """
Read each row carefully left-to-right. Each row is independent — do not carry over values from adjacent rows.
CRITICAL — MAIN STREET NAME: The main_street is ALWAYS read directly from the first column of that specific row. Never infer, guess, or copy it from another row. If the first column cell is hard to read, do your best to transcribe it exactly as it appears — do not substitute a nearby street name."""


STREET_KEYWORDS = [
    "street", "ave", "avenue", "blvd", "boulevard", "rd", "road",
    "dr", "drive", "ln", "lane", "ct", "court", "way", "location",
    "limits", "slurry", "overlay", "resurfacing", "mill", "pavement",
    "seal", "attachment a", "exhibit a", "scope of work", "linear feet"
]

def is_relevant_page(text: str) -> bool:
    t = text.lower()
    # Skip drawing/plan pages — rotated text extracts as individual characters
    words = t.split()
    if words and sum(1 for w in words if len(w) == 1) / len(words) > 0.25:
        return False
    return sum(1 for kw in STREET_KEYWORDS if kw in t) >= 3

def tables_to_markdown(tables: list) -> str:
    """Convert pdfplumber table list to markdown table strings."""
    parts = []
    for table in tables:
        if not table:
            continue
        # Normalize cells: replace None with empty string
        rows = [[str(cell or "").strip() for cell in row] for row in table]
        if not rows:
            continue
        # Determine column widths
        col_count = max(len(r) for r in rows)
        rows = [r + [""] * (col_count - len(r)) for r in rows]
        col_widths = [max(len(r[c]) for r in rows) for c in range(col_count)]
        def fmt_row(r):
            return "| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(r)) + " |"
        lines = [fmt_row(rows[0])]
        lines.append("| " + " | ".join("-" * w for w in col_widths) + " |")
        for row in rows[1:]:
            lines.append(fmt_row(row))
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def extract_text_smart(page, page_index: int = None, pdf_bytes: bytes = None) -> str:
    """
    Extract text from a pdfplumber page.
    - Always tries extract_tables() first and formats as markdown (preserves column structure)
    - Appends remaining non-table text below
    - Large-format engineering drawings (>14"): use PyMuPDF for better text flow
    """
    is_large_format = max(page.width, page.height) > 1008  # > 14 inches at 72dpi

    # Always try table extraction first — even large-format pages can have tables
    parts = []
    try:
        tables = page.extract_tables()
        if tables:
            parts.append(tables_to_markdown(tables))
    except Exception:
        pass

    if parts:
        # Tables found — also grab plain text for any content outside the table cells
        plain = page.extract_text() or ""
        if plain:
            parts.append(plain)
        return "\n\n".join(parts)

    # No tables found — use best available plain text extractor
    if is_large_format and pdf_bytes is not None and page_index is not None:
        # PyMuPDF handles large-format CAD drawings better than pdfplumber
        try:
            fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            fitz_page = fitz_doc[page_index]
            text = fitz_page.get_text("text")
            fitz_doc.close()
            if text:
                return text
        except Exception:
            pass

    return page.extract_text() or ""

def render_page_as_image(pdf_bytes: bytes, page_index: int, dpi: int = 250) -> str:
    """Render a PDF page to a base64 PNG."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc[page_index].get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return base64.standard_b64encode(img_bytes).decode()

def render_page_as_strips(pdf_bytes: bytes, page_index: int, dpi: int = 250) -> list:
    """Render a PDF page and split into top/bottom half strips. Returns list of b64 strings."""
    import io
    from PIL import Image as PILImage
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc[page_index].get_pixmap(matrix=mat)
    doc.close()
    img = PILImage.open(io.BytesIO(pix.tobytes("png")))
    mid = img.height // 2
    strips = [img.crop((0, 0, img.width, mid)), img.crop((0, mid, img.width, img.height))]
    result = []
    for strip in strips:
        buf = io.BytesIO()
        strip.save(buf, format="PNG")
        result.append(base64.standard_b64encode(buf.getvalue()).decode())
    return result

def page_has_tables(page, pdf_bytes: bytes = None, page_index: int = None) -> bool:
    """
    Detect if a page likely has table structure by looking for rows with
    many words spread across the page width (multi-column = table rows).
    Uses PyMuPDF word positions — works even when table borders are rasterized.
    Falls back to pdfplumber rect/edge detection for standard pages.
    """
    if pdf_bytes is not None and page_index is not None:
        try:
            from collections import defaultdict
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            fitz_page = doc[page_index]
            words = fitz_page.get_text("words")
            page_width = fitz_page.rect.width
            doc.close()

            if len(words) >= 20:
                rows = defaultdict(list)
                for w in words:
                    y_bin = round(w[1] / 3) * 3
                    rows[y_bin].append(w[0])  # collect x positions

                # Count rows where >=4 words span >40% of page width (multi-column)
                multi_col_rows = sum(
                    1 for x_list in rows.values()
                    if len(x_list) >= 4 and (max(x_list) - min(x_list)) / page_width > 0.4
                )
                if multi_col_rows >= 5:
                    return True
        except Exception:
            pass

    # Fallback: pdfplumber geometry (works when borders are vector lines)
    try:
        if len(page.rects) >= 6:
            return True
        h_edges = [e for e in page.edges if e.get("orientation") == "h"]
        if len(h_edges) >= 6:
            return True
    except Exception:
        pass
    return False


def call_claude(client, prompt: str, content_blocks: list, max_tokens: int = 4096, model: str = "claude-sonnet-4-6") -> dict:
    content = [{"type": "text", "text": prompt}] + content_blocks
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        msg = stream.get_final_message()
    raw = msg.content[0].text.strip()
    # Log input and response to file for debugging
    with open("/tmp/claude_last_input.txt", "w") as f:
        for block in content_blocks:
            f.write(block.get("text", "") + "\n")
    with open("/tmp/claude_last_response.txt", "w") as f:
        f.write(f"stop_reason: {msg.stop_reason}\n")
        f.write(f"raw_len: {len(raw)}\n")
        f.write("---RAW---\n")
        f.write(raw)
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            if part.startswith("json"):
                raw = part[4:].strip()
                break
            elif part.strip().startswith("{"):
                raw = part.strip()
                break
    return json.loads(raw)

def call_claude_with_retry(client, prompt, content_blocks, max_tokens=4096, max_retries=4, log_fn=None, model="claude-sonnet-4-6"):
    """Call Claude with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return call_claude(client, prompt, content_blocks, max_tokens, model=model)
        except anthropic.RateLimitError:
            wait = 30 * (2 ** attempt)  # 30, 60, 120, 240s
            if log_fn:
                log_fn(f"  ⚠ Rate limit hit — waiting {wait}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
            raise
    raise Exception("Max retries exceeded due to rate limits")


def call_gemini_image(prompt: str, b64_image: str, max_retries: int = 4, log_fn=None) -> dict:
    """Call Gemini 2.5 Pro via REST API for image-based table extraction."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise Exception("GEMINI_API_KEY not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={api_key}"
    payload = json.dumps({
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/png", "data": b64_image}},
        ]}],
        "generationConfig": {"maxOutputTokens": 65536, "temperature": 0},
    }).encode()

    for attempt in range(max_retries):
        try:
            if log_fn:
                log_fn(f"    → [Gemini] Sending request (attempt {attempt+1}, payload {len(payload)//1024}KB)...")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=240) as resp:
                if log_fn:
                    log_fn(f"    ← [Gemini] Response received, reading body...")
                data = json.loads(resp.read())
                if log_fn:
                    log_fn(f"    ← [Gemini] Body parsed OK")
            raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            with open("/tmp/gemini_last_response.txt", "w") as f:
                f.write(raw)
            if "```" in raw:
                for part in raw.split("```"):
                    if part.startswith("json"):
                        raw = part[4:].strip()
                        break
                    elif part.strip().startswith("{"):
                        raw = part.strip()
                        break
            return json.loads(raw)
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if e.code == 429:
                wait = 30 * (2 ** attempt)
                if log_fn:
                    log_fn(f"  ⚠ Gemini rate limit — waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(3)
            else:
                raise Exception(f"Gemini HTTP {e.code}: {body[:200]}")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                raise
    raise Exception("Gemini max retries exceeded")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    contents = await file.read()
    doc_id = str(uuid.uuid4())[:8]
    with pdfplumber.open(io.BytesIO(contents)) as pdf:
        total = len(pdf.pages)
    documents[doc_id] = {
        "filename": file.filename,
        "total_pages": total,
        "bytes": contents,
        "page_cache": {},
        "extracted_schema": None,
        "progress": None,
    }
    return {"doc_id": doc_id, "filename": file.filename, "total_pages": total}


@app.get("/doc/{doc_id}/page/{page_num}")
async def get_page(doc_id: str, page_num: int):
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    doc = documents[doc_id]
    if page_num < 1 or page_num > doc["total_pages"]:
        raise HTTPException(status_code=400, detail="Page out of range")
    if page_num not in doc["page_cache"]:
        with pdfplumber.open(io.BytesIO(doc["bytes"])) as pdf:
            doc["page_cache"][page_num] = extract_text_smart(
                pdf.pages[page_num - 1], page_index=page_num - 1, pdf_bytes=doc["bytes"]
            )
    return {"page": page_num, "total": doc["total_pages"], "text": doc["page_cache"][page_num]}


@app.get("/doc/{doc_id}/all")
async def get_all_text(doc_id: str):
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    doc = documents[doc_id]
    with pdfplumber.open(io.BytesIO(doc["bytes"])) as pdf:
        pages_text = []
        for i, page in enumerate(pdf.pages):
            text = doc["page_cache"].get(i + 1) or page.extract_text() or ""
            doc["page_cache"][i + 1] = text
            pages_text.append(text)
    return {"filename": doc["filename"], "total_pages": doc["total_pages"],
            "text": "\n\n--- PAGE BREAK ---\n\n".join(pages_text)}


@app.get("/doc/{doc_id}/status")
async def get_status(doc_id: str):
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    doc = documents[doc_id]
    return {
        "done": doc["extracted_schema"] is not None,
        "progress": doc.get("progress"),
        "schema": doc["extracted_schema"],
    }


@app.get("/doc/{doc_id}/chunks")
async def get_chunks(doc_id: str):
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = documents[doc_id].get("chunk_debug", [])
    if not chunks:
        raise HTTPException(status_code=404, detail="No chunk data available yet")
    lines = []
    for c in chunks:
        lines.append(f"{'='*60}")
        lines.append(f"CHUNK {c['index']+1}/{c['total']}  [{c['source'].upper()}]  {c['char_count']} chars")
        lines.append(f"{'='*60}")
        lines.append(c['text'] if c['text'] else "(image — no text content)")
        lines.append("")
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines))


def run_extraction(doc_id: str, api_key: str):
    """Run the full extraction pipeline — text only, no images."""
    doc = documents[doc_id]

    def log(msg, streets_so_far=None):
        p = doc.get("progress") or {"logs": [], "streets_so_far": []}
        p["logs"].append(msg)
        if streets_so_far is not None:
            p["streets_so_far"] = streets_so_far
        doc["progress"] = p

    # --- Step 1: scan all pages with smart text extraction ---
    log("Scanning all pages...")
    relevant_indices = []
    table_page_indices = set()  # pages that have detected table grid structures

    pdf_bytes = doc["bytes"]
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = extract_text_smart(page, page_index=i, pdf_bytes=pdf_bytes)
            doc["page_cache"][i + 1] = text
            has_table = page_has_tables(page, pdf_bytes=pdf_bytes, page_index=i)
            relevant = is_relevant_page(text)
            if has_table:
                table_page_indices.add(i)
            if relevant:
                relevant_indices.append(i)
            tags = []
            if relevant:
                tags.append("street content")
            if has_table:
                tags.append("📊 table detected → will send image")
            if tags:
                log(f"  Page {i + 1}: {', '.join(tags)}")

    log(f"────────────────────────────────────")
    log(f"Scan complete: {doc['total_pages']} pages total")
    log(f"  Street-relevant: {len(relevant_indices)} pages ({[i+1 for i in relevant_indices]})")
    log(f"  Table pages (image): {len(table_page_indices)} pages ({sorted([i+1 for i in table_page_indices])})")
    log(f"────────────────────────────────────")

    def is_empty(v):
        return not v or str(v).strip() in ("?", "null", "None", "")

    client = anthropic.Anthropic(api_key=api_key)

    # --- Step 2: extract header info from first 5 pages ---
    log("Extracting project info from cover pages...")
    header_indices = list(range(min(5, doc["total_pages"])))
    header_blocks = []
    header_chars = 0
    for page_idx in header_indices:
        text = doc["page_cache"].get(page_idx + 1, "")
        entry = f"\n--- Page {page_idx + 1} ---\n{text}"
        header_blocks.append({"type": "text", "text": entry})
        header_chars += len(entry)

    try:
        schema = call_claude_with_retry(client, HEADER_PROMPT, header_blocks, max_tokens=1024, log_fn=log)
        log(f"✓ Project: {schema.get('project_name')} | {schema.get('city')} | {schema.get('bid_number')}")
    except Exception as e:
        log(f"✗ Header extraction failed: {e}")
        return

    schema["streets"] = []
    all_streets = []

    # --- Step 3: build chunks from all relevant pages ---
    # Table pages send image-only. Non-table pages are batched as text chunks.
    # chunks is a list of {"blocks": [...], "source": "text"|"image"}
    chunks = []
    current_blocks = []
    current_size = 0

    def flush_text_chunk():
        nonlocal current_blocks, current_size
        if current_blocks:
            chunks.append({"blocks": current_blocks, "source": "text"})
            current_blocks = []
            current_size = 0

    for page_idx in relevant_indices:
        if page_idx in table_page_indices:
            flush_text_chunk()
            try:
                strips = render_page_as_strips(pdf_bytes, page_idx)
                for strip_num, b64 in enumerate(strips):
                    image_block = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}
                    chunks.append({"blocks": [image_block], "source": "image", "page": page_idx + 1, "strip": strip_num + 1})
                log(f"  Page {page_idx + 1}: queued {len(strips)} image strips (table page)")
            except Exception as e:
                log(f"  Page {page_idx + 1}: image render failed ({e}), skipping")

        # non-table pages skipped — image-only mode

    n_img = sum(1 for c in chunks if c["source"] == "image")
    n_txt = sum(1 for c in chunks if c["source"] == "text")
    log(f"Split into {len(chunks)} chunks ({n_txt} text, {n_img} image). Starting street extraction...")

    doc["chunk_debug"] = []
    log_lock = threading.Lock()

    def process_image_chunk(i, chunk):
        """Process a single image chunk: main Gemini call + optional rescue. Returns (i, streets, logs)."""
        chunk_blocks = chunk["blocks"]
        chunk_text = "\n".join(b.get("text", "") for b in chunk_blocks if b["type"] == "text")

        def chunk_log(msg, data=None):
            with log_lock:
                log(msg, data)

        page_label = f"p{chunk.get('page','?')} strip {chunk.get('strip','?')}"
        chunk_log(f"🚀 [IMG {i+1}/{len(chunks)} {page_label}] Starting parallel extraction...")
        t_start = time.time()
        try:
            b64 = chunk_blocks[0]["source"]["data"]
            chunk_log(f"  → [IMG {i+1} {page_label}] Sending to Gemini ({len(b64) * 3 // 4 // 1024}KB)...")
            result = call_gemini_image(STREETS_PROMPT_IMAGE, b64, log_fn=chunk_log)
            elapsed = time.time() - t_start
            new_streets = result.get("streets", [])
            for s in new_streets:
                s.setdefault("source", "image")
            chunk_log(f"  ✓ [IMG {i+1}] Done in {elapsed:.1f}s — {len(new_streets)} streets")
            return i, new_streets, chunk_text

        except Exception as e:
            chunk_log(f"  ✗ [IMG {i+1}] Error: {str(e)[:200]}")
            return i, [], chunk_text

    # Separate image and text chunks, preserving original order index
    image_chunks = [(i, c) for i, c in enumerate(chunks) if c["source"] == "image"]

    # Build chunk debug info
    for i, chunk in enumerate(chunks):
        chunk_text = "\n".join(b.get("text", "") for b in chunk["blocks"] if b["type"] == "text")
        doc["chunk_debug"].append({
            "index": i, "total": len(chunks), "source": chunk["source"],
            "char_count": len(chunk_text), "text": chunk_text,
        })

    # Collect results keyed by original chunk index so we can merge in order
    chunk_results = {}  # index -> list of streets

    # --- Run all image chunks in parallel ---
    if image_chunks:
        log(f"⚡ Launching {len(image_chunks)} image chunk(s) in parallel...")
        t_parallel_start = time.time()
        with ThreadPoolExecutor(max_workers=len(image_chunks)) as executor:
            futures = {executor.submit(process_image_chunk, i, c): i for i, c in image_chunks}
            for future in as_completed(futures):
                i, streets, _ = future.result()
                chunk_results[i] = streets
                # Stream partial results immediately as each chunk finishes
                partial = []
                for idx in sorted(chunk_results.keys()):
                    partial.extend(chunk_results[idx])
                with log_lock:
                    log(f"⚡ [IMG {i+1}] finished — {len(streets)} streets (partial total: {len(partial)})", partial)
        log(f"⚡ All image chunks done in {time.time() - t_parallel_start:.1f}s total")

    # --- Merge results in original chunk order ---
    for i in sorted(chunk_results.keys()):
        streets = chunk_results[i]
        all_streets.extend(streets)
        schema["streets"] = all_streets
        log(f"  📥 [Chunk {i+1}] merged {len(streets)} streets (running total: {len(all_streets)})")

    # --- Deduplication ---
    # 1. Remove exact duplicates (same main+from+to+work_type)
    # 2. If a street has null/? from AND null/? to, and a richer version exists, drop the sparse one
    _SUFFIX_MAP = {
        "STREET": "ST", "AVENUE": "AV", "DRIVE": "DR", "BOULEVARD": "BL",
        "ROAD": "RD", "COURT": "CT", "LANE": "LN", "PLACE": "PL",
        "WAY": "WY", "CIRCLE": "CIR", "TERRACE": "TER", "TRAIL": "TRL",
    }
    def norm_name(v):
        if not v:
            return ""
        parts = v.strip().upper().split()
        if parts and parts[-1] in _SUFFIX_MAP:
            parts[-1] = _SUFFIX_MAP[parts[-1]]
        return " ".join(parts)

    seen = {}
    for s in all_streets:
        key = (
            norm_name(s.get("main_street")),
            norm_name(s.get("from_street")),
            norm_name(s.get("to_street")),
            (s.get("work_type") or "").strip().upper(),
        )
        empty = is_empty(s.get("from_street")) and is_empty(s.get("to_street"))
        src = s.get("source", "text")
        if key not in seen:
            seen[key] = (s, empty)
        else:
            existing, existing_empty = seen[key]
            existing_src = existing.get("source", "text")
            # Priority: image > text; richer (non-empty) > sparse within same source
            if src == "image" and existing_src != "image":
                seen[key] = (s, empty)  # image wins over text
            elif not empty and existing_empty and src == existing_src:
                seen[key] = (s, empty)  # richer wins within same source

    # Also drop sparse entries whose main_street has ANY richer entry
    mains_with_data = {k[0] for k, (_, emp) in seen.items() if not emp}
    dropped_sparse = []
    deduped = []
    for (k, (s, emp)) in seen.items():
        if not emp or k[0] not in mains_with_data:
            deduped.append(s)
        else:
            dropped_sparse.append({"reason": "sparse+richer_exists", "street": s})

    before = len(all_streets)
    all_streets = deduped

    # Drop streets with no from_street AND no to_street
    dropped_empty = [s for s in all_streets if is_empty(s.get("from_street")) and is_empty(s.get("to_street"))]
    all_streets = [s for s in all_streets if not (is_empty(s.get("from_street")) and is_empty(s.get("to_street")))]

    schema["streets"] = all_streets
    log(f"  Deduplication: {before} → {len(all_streets)} streets (removed {before - len(all_streets)} duplicates)")

    # Write drop log for debugging
    with open("/tmp/dedup_dropped.txt", "w") as f:
        f.write(f"=== DROPPED: sparse (richer version exists) — {len(dropped_sparse)} ===\n")
        for d in dropped_sparse:
            s = d["street"]
            f.write(f"  {s.get('main_street')} | {s.get('from_street')} → {s.get('to_street')} | src={s.get('source')}\n")
        f.write(f"\n=== DROPPED: both from+to empty — {len(dropped_empty)} ===\n")
        for s in dropped_empty:
            f.write(f"  {s.get('main_street')} | {s.get('from_street')} → {s.get('to_street')} | src={s.get('source')}\n")
        f.write(f"\n=== KEPT: {len(all_streets)} streets ===\n")
        for s in all_streets:
            f.write(f"  {s.get('main_street')} | {s.get('from_street')} → {s.get('to_street')} | src={s.get('source')}\n")

    schema["_meta"] = {
        "total_pages": doc["total_pages"],
        "chunks_processed": len(chunks),
        "street_pages_found": len(relevant_indices),
        "total_streets": len(all_streets),
    }
    doc["extracted_schema"] = schema
    log(f"✓ Done! {len(all_streets)} streets extracted total.", all_streets)


@app.post("/doc/{doc_id}/extract")
async def extract_schema(doc_id: str):
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    doc = documents[doc_id]
    if doc["extracted_schema"]:
        return doc["extracted_schema"]
    if doc.get("progress") is not None:
        return {"status": "already_running"}
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    # Reset progress
    doc["progress"] = {"logs": [], "streets_so_far": []}

    # Run extraction in a background thread so we can return immediately
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, run_extraction, doc_id, api_key)

    return {"status": "started"}


@app.delete("/doc/{doc_id}/extract")
async def clear_extract(doc_id: str):
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    documents[doc_id]["extracted_schema"] = None
    documents[doc_id]["progress"] = None
    return {"status": "cleared"}


@app.get("/documents")
async def list_docs():
    return [
        {"doc_id": k, "filename": v["filename"], "total_pages": v["total_pages"]}
        for k, v in documents.items()
    ]


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
