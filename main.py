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

STREETS_PROMPT = """You are parsing pages from a road construction bid document. Extract ALL street segments from any tables or lists on these pages.
Return ONLY valid JSON: {"streets": [...]}
Each street object: {"main_street": "...", "from_street": "...", "to_street": "...", "work_type": "...", "location": "..."}
Use null for missing fields. Return {"streets": []} if no street data on these pages.
IMPORTANT: Read every row of every table. Do not skip any streets.
NOTE: Some pages are CAD engineering drawings where text may be partially clipped at cell borders (e.g. "ARLING ST" = "YEARLING ST", "BURY DR" = continuation of previous row). Use context and common street naming patterns to reconstruct partial names. Include every street you can identify even if the name is partially truncated."""


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

def extract_text_smart(page, page_index: int = None, pdf_bytes: bytes = None) -> str:
    """
    Extract text from a pdfplumber page.
    Standard pages: pdfplumber extract_text() works well.
    Large-format engineering drawings (>14"): use PyMuPDF which produces cleaner
    line-by-line output than pdfplumber's word-coordinate approach on these files.
    """
    is_large_format = max(page.width, page.height) > 1008  # > 14 inches at 72dpi
    if not is_large_format:
        return page.extract_text() or ""

    # Use PyMuPDF for large-format pages — better text flow on engineering drawings
    if pdf_bytes is not None and page_index is not None:
        try:
            fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            fitz_page = fitz_doc[page_index]
            text = fitz_page.get_text("text")
            fitz_doc.close()
            return text or ""
        except Exception:
            pass

    return page.extract_text() or ""

def call_claude(client, prompt: str, content_blocks: list, max_tokens: int = 4096) -> dict:
    content = [{"type": "text", "text": prompt}] + content_blocks
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        msg = stream.get_final_message()
    raw = msg.content[0].text.strip()
    # Log raw response to file for debugging
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

def call_claude_with_retry(client, prompt, content_blocks, max_tokens=4096, max_retries=3):
    """Call Claude with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return call_claude(client, prompt, content_blocks, max_tokens)
        except anthropic.RateLimitError as e:
            wait = 60 * (attempt + 1)
            raise RateLimitRetry(wait, str(e))
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            raise

class RateLimitRetry(Exception):
    def __init__(self, wait_seconds, msg):
        self.wait_seconds = wait_seconds
        self.msg = msg


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

    pdf_bytes = doc["bytes"]
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = extract_text_smart(page, page_index=i, pdf_bytes=pdf_bytes)
            doc["page_cache"][i + 1] = text
            if is_relevant_page(text):
                relevant_indices.append(i)

    log(f"Found {len(relevant_indices)} street-relevant pages out of {doc['total_pages']} total")

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
        schema = call_claude_with_retry(client, HEADER_PROMPT, header_blocks, max_tokens=1024)
        log(f"✓ Project: {schema.get('project_name')} | {schema.get('city')} | {schema.get('bid_number')}")
    except Exception as e:
        log(f"✗ Header extraction failed: {e}")
        return

    schema["streets"] = []
    all_streets = []

    # --- Step 3: build text chunks from all relevant pages ---
    chunks = []
    current_blocks = []
    current_size = 0
    for page_idx in relevant_indices:
        text = doc["page_cache"].get(page_idx + 1, "")
        entry = f"\n--- Page {page_idx + 1} ---\n{text}"
        if current_size + len(entry) > CHUNK_CHAR_LIMIT and current_blocks:
            chunks.append(current_blocks)
            current_blocks = []
            current_size = 0
        current_blocks.append({"type": "text", "text": entry})
        current_size += len(entry)
    if current_blocks:
        chunks.append(current_blocks)

    log(f"Split into {len(chunks)} chunks. Starting street extraction...")

    # Rate limit: 30k tokens/min. Each chunk is ~20k tokens (80k chars / 4).
    # Sleep proportionally — always sleep before every chunk including the first
    # since the header call already consumed tokens in this minute window.
    RATE_LIMIT_CHARS_PER_MIN = 120000  # ~30k tokens × 4 chars/token

    def sleep_for_chunk(chars_used: int):
        wait = max(5, int((chars_used / RATE_LIMIT_CHARS_PER_MIN) * 65))
        log(f"  Waiting {wait}s for rate limit...")
        time.sleep(wait)

    for i, chunk_blocks in enumerate(chunks):
        chunk_size = sum(len(b["text"]) for b in chunk_blocks)
        sleep_chars = (header_chars + chunk_size) if i == 0 else chunk_size
        sleep_for_chunk(sleep_chars)
        log(f"Processing chunk {i+1}/{len(chunks)} (~{chunk_size // 1000}k chars)...")
        try:
            result = call_claude_with_retry(client, STREETS_PROMPT, chunk_blocks, max_tokens=32000)
            new_streets = result.get("streets", [])
            all_streets.extend(new_streets)
            schema["streets"] = all_streets
            if new_streets:
                log(f"  ✓ {len(new_streets)} streets found (total: {len(all_streets)})", all_streets)
            else:
                log(f"  · No streets on these pages")
        except RateLimitRetry as e:
            log(f"  ⚠ Rate limit — waiting {e.wait_seconds}s then retrying...")
            time.sleep(e.wait_seconds)
            try:
                result = call_claude_with_retry(client, STREETS_PROMPT, chunk_blocks, max_tokens=32000)
                new_streets = result.get("streets", [])
                all_streets.extend(new_streets)
                schema["streets"] = all_streets
                log(f"  ✓ Retry ok: {len(new_streets)} streets (total: {len(all_streets)})", all_streets)
            except Exception as e2:
                log(f"  ✗ Failed after retry: {str(e2)[:200]}")
        except Exception as e:
            log(f"  ✗ Error: {str(e)[:200]}")

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
