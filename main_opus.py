from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
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
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Persistent cache paths
DOCAI_CACHE_DIR  = os.path.join(BASE_DIR, "docai_cache")
SCREEN_CACHE_DIR = os.path.join(BASE_DIR, "screen_cache")
VISION_CACHE_FILE = os.path.join(BASE_DIR, "vision_cache.json")
os.makedirs(DOCAI_CACHE_DIR, exist_ok=True)
os.makedirs(SCREEN_CACHE_DIR, exist_ok=True)

# Global semaphore: cap concurrent LLM calls to avoid rate limits across parallel eval workers

_header_cache_lock   = threading.Lock()

# Load vision header cache from disk on startup (persists across server restarts)
def _key_to_json(k):
    """Recursively convert a nested tuple key to a JSON-serializable list."""
    if isinstance(k, (tuple, list)):
        return [_key_to_json(x) for x in k]
    return k

def _json_to_key(v):
    """Recursively convert a nested list back to tuples for use as dict keys."""
    if isinstance(v, list):
        return tuple(_json_to_key(x) for x in v)
    return v

def _load_vision_cache() -> dict:
    if os.path.exists(VISION_CACHE_FILE):
        try:
            with open(VISION_CACHE_FILE, "r") as f:
                raw = json.load(f)
            return {_json_to_key(json.loads(k)): v for k, v in raw.items()}
        except Exception:
            pass
    return {}

def _save_vision_cache(cache: dict):
    try:
        serializable = {json.dumps(_key_to_json(k)): v for k, v in cache.items()}
        with open(VISION_CACHE_FILE, "w") as f:
            json.dump(serializable, f)
    except Exception:
        pass

_vision_header_cache = _load_vision_cache()  # header_key → confirmed col mapping from Haiku Vision
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI()

# CORS — restrict to specific origins in prod via ALLOWED_ORIGINS env var
# e.g. ALLOWED_ORIGINS=https://partner.com,https://app.partner.com
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
_origins = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

from highway.routes import router as highway_router
app.include_router(highway_router)

from eval_routes import router as eval_router
app.include_router(eval_router)

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

STREETS_PROMPT_IMAGE = """You are parsing a scanned table image from a road construction bid document.

FIRST — decide if this image could contain street segment data.
- Return exactly {"streets": []} ONLY if the image is clearly a GEOGRAPHIC MAP, ENGINEERING DRAWING, PHOTOGRAPH, or COVER/TITLE PAGE with no tabular data at all.
- If the image shows rows of text — even without visible column headers (it may be a continuation strip of a larger table) — treat it as a data table and extract every row.
- When in doubt, attempt extraction.

Return ONLY valid JSON: {"streets": [...]}
Each street object: {"main_street": "...", "from_street": "...", "to_street": "...", "work_type": "...", "location": "...", "source": "image"}

""" + _STREETS_PROMPT_BASE.replace("{SOURCE_TAG}", "image") + """
Read each row carefully left-to-right. Each row is independent — do not carry over values from adjacent rows.
Important: main_street must be copied exactly from the first column of that specific row. Do not infer or substitute it from another row. If the text is hard to read, transcribe it as closely as possible.
Important: Copy street values exactly — suffixes like AV, ST, DR, BL, RD are distinct and not interchangeable. Placeholders like EOS, EOC, BOS, BOC, EOP are valid cross-street values, not blanks. Numeric-only cells (location numbers, sequence numbers) are not street names — do not put them in from_street or to_street."""


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
    """Render a PDF page and split into 4 equal horizontal strips. Returns list of b64 strings."""
    import io
    from PIL import Image as PILImage
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc[page_index].get_pixmap(matrix=mat)
    doc.close()
    img = PILImage.open(io.BytesIO(pix.tobytes("png")))
    h, w = img.height, img.width
    q = h // 4
    strips = [
        img.crop((0, 0,     w, q)),
        img.crop((0, q,     w, q * 2)),
        img.crop((0, q * 2, w, q * 3)),
        img.crop((0, q * 3, w, h)),
    ]
    result = []
    for strip in strips:
        buf = io.BytesIO()
        strip.save(buf, format="PNG")
        result.append(base64.standard_b64encode(buf.getvalue()).decode())
    return result

# Column header keyword sets for deterministic text extraction
_COL_KEYWORDS = {
    "main_street": ["street name", "street", "roadway", "road name", "location name"],
    "from_street": ["cross street 1", "cross st 1", "from street", "from", "begin", "start", "limits from", "cross street from"],
    "to_street":   ["cross street 2", "cross st 2", "to street", "to", "end", "terminus", "limits to", "cross street to"],
    "work_type":   ["activity", "project description", "work type", "work", "type", "description", "project descripton"],
    "location":    ["location", "district", "council district", "segment id", "segment", "project title", "community planning area", "zone"],
}

def _match_col(header_cell: str) -> str:
    """Return the schema field name that best matches a header cell, or '' if none."""
    h = (header_cell or "").strip().lower()
    for field, keywords in _COL_KEYWORDS.items():
        for kw in keywords:
            if kw in h:
                return field
    return ""

def _is_header_row(row: list) -> bool:
    """Check if a row looks like a column header (has recognizable field keywords)."""
    matched = sum(1 for cell in row if _match_col(str(cell or "")))
    return matched >= 2

def _row_to_street(row: list, col_map: dict, page_num: int) -> dict:
    """Convert a table row to a street dict using col_map. Returns None if no main_street."""
    import re as _re
    s = {"source": "text", "page": page_num}
    for col_idx, field in col_map.items():
        if col_idx < len(row):
            val = str(row[col_idx] or "").strip()
            s[field] = val if val else None
    main = s.get("main_street") or ""
    skip_values = {"street name", "street", "name", "roadway", "location name", ""}
    if main.lower() in skip_values:
        return None
    # Strip leading segment ID (e.g. "SS-026228-PV1 TEASDALE AV" → "TEASDALE AV")
    main = _re.sub(r'^[A-Z]{1,4}-\d{4,8}-[A-Z0-9]+(?:-[A-Z0-9]+)*\s+', '', main).strip()
    if main:
        s["main_street"] = main
    # Clean work_type: strip date bleed and leading planning-area prefix
    wt = s.get("work_type") or ""
    if wt:
        wt = _re.sub(r'\s+\d{2}/\d{4}.*$', '', wt).strip()
        wt = _re.sub(r'(\s+\b\d{1,2}\b)+\s*$', '', wt).strip()
        wt = _re.sub(r'^(?:[A-Z][A-Z /]+?)\s+(?=AC\b|CRACK|SLURRY|CAPE|GRIND|OVERLAY|PATCH|MILL|RESURFACE)', '', wt).strip()
        s["work_type"] = wt or None
    return s


def _find_header_xmap(all_words: list) -> tuple:
    """
    Scan word lines for a header row containing recognizable multi-word column phrases.
    Returns (header_xmap, header_bottom_y) where header_xmap is {field: x0}, or (None, None).

    Uses n-gram phrase matching so "Cross Street 1" (3 separate words) correctly maps
    to from_street even though no single word matches.
    """
    # Group words into lines by y-position
    line_buckets = {}
    for w in all_words:
        y = round(w["top"] / 3) * 3
        line_buckets.setdefault(y, []).append(w)

    sorted_ys = sorted(line_buckets.keys())

    for y in sorted_ys:
        line_words = sorted(line_buckets[y], key=lambda w: w["x0"])
        matches = {}
        used_indices = set()
        # Try n-grams of length 1-5 from left to right
        for i in range(len(line_words)):
            if i in used_indices:
                continue
            for n in range(5, 0, -1):
                if i + n > len(line_words):
                    continue
                phrase = " ".join(line_words[j]["text"] for j in range(i, i + n)).lower()
                for field, keywords in _COL_KEYWORDS.items():
                    if field in matches:
                        continue
                    for kw in keywords:
                        if kw == phrase:
                            matches[field] = line_words[i]["x0"]
                            used_indices.update(range(i, i + n))
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break  # found a match starting at i, move to next unused word

        if len(matches) >= 2 and "main_street" in matches:
            header_bottom_y = max(w["bottom"] for w in line_words) + 2
            return matches, header_bottom_y

    return None, None


def try_extract_tables_text(pdf_bytes: bytes, page_index: int, page_num: int, fallback_xmap: dict = None) -> tuple:
    """
    Deterministic street extraction from text-based PDFs.

    Two-pass approach:
    1. extract_tables() — captures bordered/structured table rows perfectly
    2. x-band parsing — captures borderless plain-text rows using column x-positions
       learned from the header row via n-gram phrase matching.

    fallback_xmap: x-band map inherited from a previous page (for continuation pages
                   that have no header row of their own).

    Returns (streets, xmap) where streets is a list or None, and xmap is the header
    map used (to pass as fallback_xmap to the next page).
    """
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page = pdf.pages[page_index]
        tables = page.extract_tables()
        all_words = page.extract_words()

    all_streets = []
    found_valid_table = False
    pass1_rows = set()  # track (main, from, to) tuples added in pass 1 to avoid dupes

    # --- Pass 1: bordered table rows via extract_tables() ---
    for table in (tables or []):
        if not table or len(table) < 2:
            continue

        header_row = None
        data_start = 0
        for row_idx, row in enumerate(table):
            if _is_header_row(row):
                header_row = row
                data_start = row_idx + 1
                break
        if header_row is None:
            continue

        col_map = {}
        for col_idx, cell in enumerate(header_row):
            field = _match_col(str(cell or ""))
            if field and field not in col_map.values():
                col_map[col_idx] = field

        fields_found = set(col_map.values())
        if "main_street" not in fields_found:
            continue
        if "from_street" not in fields_found and "to_street" not in fields_found:
            continue

        found_valid_table = True

        for row in table[data_start:]:
            if not row or not any(c and str(c).strip() for c in row):
                continue
            s = _row_to_street(row, col_map, page_num)
            if s:
                all_streets.append(s)
                key = (s.get("main_street", ""), s.get("from_street", ""), s.get("to_street", ""))
                pass1_rows.add(key)

    # --- Pass 2: x-band parsing for borderless / text-only rows ---
    # Use n-gram phrase matching to find header on this page; fall back to inherited header
    header_xmap, header_bottom_y = _find_header_xmap(all_words)
    if header_xmap is None and fallback_xmap is not None:
        # Continuation page — no header row, inherit column layout from previous page
        header_xmap = fallback_xmap
        header_bottom_y = 0  # all words are data

    used_xmap = header_xmap  # return to caller for next page

    if header_xmap and ("from_street" in header_xmap or "to_street" in header_xmap):
        found_valid_table = True

        sorted_fields = sorted(header_xmap.items(), key=lambda kv: kv[1])
        xband_list = []
        for i, (field, x0) in enumerate(sorted_fields):
            # Use midpoint between adjacent headers as boundary so that data words
            # starting slightly left of their column header still land in the right band.
            # e.g. if "Cross Street 1" header is at x=250 but "BROOKBURN" data is at
            # x=220, the midpoint boundary (e.g. 150) correctly puts it in from_street.
            if i + 1 < len(sorted_fields):
                next_x0 = sorted_fields[i + 1][1]
                x_end = (x0 + next_x0) / 2
            else:
                x_end = 9999
            if i == 0:
                # First column starts at x=0 to catch multi-word names that begin left
                # of the header word (e.g. "CAM DE LA COSTA")
                x_actual_start = 0
            else:
                prev_x0 = sorted_fields[i - 1][1]
                x_actual_start = (prev_x0 + x0) / 2
            xband_list.append((x_actual_start, x_end, field))

        # Group words below header into lines
        line_buckets = {}
        for w in all_words:
            if w["top"] <= header_bottom_y:
                continue
            y = round(w["top"] / 3) * 3
            line_buckets.setdefault(y, []).append(w)

        for y in sorted(line_buckets.keys()):
            line_words = sorted(line_buckets[y], key=lambda w: w["x0"])
            cells = {}
            for w in line_words:
                # Use word center for column assignment so words that slightly
                # straddle a boundary land in whichever column holds most of the word.
                w_center = (w["x0"] + w["x1"]) / 2
                for x_start, x_end, field in xband_list:
                    if w_center >= x_start and w_center < x_end:
                        cells[field] = (cells.get(field, "") + " " + w["text"]).strip()
                        break

            main = cells.get("main_street", "").strip()
            if not main:
                continue
            if _match_col(main) or main.lower() in ("street name", "street", "name"):
                continue

            # Filter garbage: street names are ALL CAPS; mixed-case = sentence text
            alpha_chars = [c for c in main if c.isalpha()]
            if alpha_chars and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) < 0.8:
                continue

            # Filter form/legal text: checkbox patterns like "( ) DECLARED", "(X) YES"
            if any("(" in (cells.get(f) or "") for f in ("main_street", "from_street", "to_street")):
                continue

            # Require at least one recognized street-type word anywhere in main_street.
            # This catches garbage like "EMERGENCY", "PROJECT", "EXEMPTION" while allowing
            # streets with suffixes (BAMBURGH PL), Spanish names (CAM PLAYA AZUL, VIA DEL
            # COSIRA), and other patterns common across US city docs.
            _STREET_TYPE_WORDS = {
                # English suffixes
                "ST", "AV", "AVE", "DR", "RD", "BL", "BLVD", "CT", "LN", "PL",
                "WY", "WAY", "CIR", "CR", "TER", "TRL", "TRAIL", "HWY", "FWY",
                "PKWY", "PY", "LOOP", "EXPY", "ALY", "ALLEY", "XING",
                # Spanish/California prefixes and types
                "VIA", "CAM", "CAMINO", "CALLE", "PASEO", "CTE", "CORTE",
                "AVNDA", "AVENIDA", "RANCHO",
            }
            main_words = set(main.upper().split())
            if not main_words & _STREET_TYPE_WORDS:
                continue

            # Skip rows where from/to fields are unreasonably long — indicates a different
            # table format (e.g. work order pages with widths, costs, map refs) where
            # everything to the right of the cross street bleeds into the to_street band.
            # Real street names are short; 60 chars is generous.
            if any(len(cells.get(f) or "") > 60 for f in ("from_street", "to_street")):
                continue

            # Truncate cross street fields at the first standalone integer word.
            # In docs with more columns than expected, extra columns (district number,
            # map reference, planning area, road type) bleed into the to_street band.
            # e.g. "STEADMAN ST 6 1208-G5 MIRA MESA Residential" → "STEADMAN ST"
            # Starts at index 1 so a leading digit like "1ST" is not stripped.
            for f in ("from_street", "to_street"):
                v = cells.get(f, "")
                if not v:
                    continue
                words = v.split()
                for idx, word in enumerate(words):
                    if idx > 0 and word.isdigit():
                        cells[f] = " ".join(words[:idx]).strip() or None
                        break
                else:
                    if len(words) == 1 and words[0].isdigit():
                        cells[f] = None  # purely numeric — clear it

            # Strip leading segment ID from main_street (e.g. "SS-026228-PV1 TEASDALE AV" → "TEASDALE AV")
            import re as _re
            ms = cells.get("main_street") or ""
            ms = _re.sub(r'^(?:[A-Z]\d{4}\s+)?[A-Z]{1,4}-\d{4,8}-[A-Z0-9]+(?:-[A-Z0-9]+)*\s+', '', ms).strip()
            if ms:
                cells["main_street"] = ms

            # Clean work_type: strip leading ALL-CAPS planning-area prefix and trailing
            # date/numeric bleed (e.g. "UNIVERSITY AC - Slurry Seal 03/2026 07/2026 5 8 Residential"
            # → "AC - Slurry Seal").
            wt = cells.get("work_type") or ""
            if wt:
                # Remove trailing: date patterns (MM/YYYY), standalone short integers, and
                # Residential/Major/Local classification words that bleed in from right columns.
                wt = _re.sub(r'\s+\d{2}/\d{4}.*$', '', wt).strip()  # truncate at first date
                wt = _re.sub(r'(\s+\b\d{1,2}\b)+\s*$', '', wt).strip()  # trailing short ints
                # Strip leading ALL-CAPS single-word prefix that is not part of the activity
                # (planning area names like "UNIVERSITY", "MIRAMAR RANCH NORTH", etc.)
                # Heuristic: if work_type starts with all-caps word(s) before "AC " or a
                # known activity keyword, strip them.
                wt = _re.sub(r'^(?:[A-Z][A-Z /]+?)\s+(?=AC\b|CRACK|SLURRY|CAPE|GRIND|OVERLAY|PATCH|MILL|RESURFACE)', '', wt).strip()
                cells["work_type"] = wt or None

            populated = sum(1 for f in ["main_street", "from_street", "to_street"] if cells.get(f))
            if populated < 2:
                continue

            key = (cells.get("main_street"), cells.get("from_street"), cells.get("to_street"))
            if key in pass1_rows:
                continue  # already captured by bordered table pass

            all_streets.append({
                "source": "text",
                "page": page_num,
                "main_street": cells.get("main_street") or None,
                "from_street": cells.get("from_street") or None,
                "to_street":   cells.get("to_street") or None,
                "work_type":   cells.get("work_type") or None,
                "location":    cells.get("location") or None,
            })

    if not found_valid_table:
        return None, used_xmap

    return (all_streets if all_streets else None), used_xmap


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

def call_claude_with_retry(client, prompt, content_blocks, max_tokens=4096, max_retries=6, log_fn=None, model="claude-haiku-4-5-20251001"):
    """Call Claude with Gemini fallback on rate limit errors — alternates between models."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}" if gemini_key else None

    for attempt in range(max_retries):
        use_gemini = (attempt % 2 == 1) and gemini_url  # alternate: Claude on even, Gemini on odd
        try:
            if not use_gemini:
                return call_claude(client, prompt, content_blocks, max_tokens, model=model)
            else:
                # Reassemble text content for Gemini
                text_parts = []
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif hasattr(block, "type") and block.type == "text":
                        text_parts.append(block.text)
                combined = prompt + "\n\n" + "\n".join(text_parts)
                payload = json.dumps({
                    "contents": [{"parts": [{"text": combined}]}],
                    "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0},
                }).encode()
                req = urllib.request.Request(gemini_url, data=payload, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                return _parse_llm_json(raw)
        except anthropic.RateLimitError:
            if log_fn:
                log_fn(f"  ⚠ Claude rate limit (attempt {attempt+1}) — switching to Gemini...")
            time.sleep(5)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                if log_fn:
                    log_fn(f"  ⚠ Gemini rate limit (attempt {attempt+1}) — switching to Claude...")
                time.sleep(5)
            elif attempt < max_retries - 1:
                time.sleep(3)
            else:
                raise
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                raise
    raise Exception("Max retries exceeded (both Claude and Gemini rate limited)")


def call_gemini_image(prompt: str, b64_image: str, max_retries: int = 4, log_fn=None) -> dict:
    """Call Gemini 2.5 Pro via REST API for image-based table extraction."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise Exception("GEMINI_API_KEY not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
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
            with urllib.request.urlopen(req, timeout=120) as resp:
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


def call_vision_with_retry(prompt: str, b64_image: str, max_tokens: int = 512, max_retries: int = 6, log_fn=None) -> dict:
    """Call Haiku Vision with Gemini Flash fallback on rate limits — alternates between models."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}" if gemini_key else None

    for attempt in range(max_retries):
        use_gemini = (attempt % 2 == 1) and gemini_url
        try:
            if not use_gemini:
                client = anthropic.Anthropic(api_key=anthropic_key)
                msg = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=max_tokens,
                    temperature=0,
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_image}},
                        {"type": "text", "text": prompt},
                    ]}],
                )
                return _parse_llm_json(msg.content[0].text.strip())
            else:
                payload = json.dumps({
                    "contents": [{"parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": b64_image}},
                    ]}],
                    "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0},
                }).encode()
                req = urllib.request.Request(gemini_url, data=payload, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read())
                raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                return _parse_llm_json(raw)
        except anthropic.RateLimitError:
            if log_fn:
                log_fn(f"  ⚠ Haiku Vision rate limit (attempt {attempt+1}) — switching to Gemini...")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                if log_fn:
                    log_fn(f"  ⚠ Gemini Vision rate limit (attempt {attempt+1}) — switching to Haiku...")
            elif attempt < max_retries - 1:
                time.sleep(3)
            else:
                raise
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                raise
    raise Exception("Vision max retries exceeded (both Haiku and Gemini rate limited)")


# ─── Document AI Form Parser ─────────────────────────────────────────────────
DOCAI_PROJECT      = "284828153354"
DOCAI_LOCATION     = "us"
DOCAI_PROCESSOR_ID = "8e7372377435d1ba"
DOCAI_CRED_FILE    = os.path.join(BASE_DIR, "bid-parser-492923-ea3bbe06380d.json")

STREETS_PROMPT_DOCAI = """You are parsing road construction bid document tables extracted by a layout parser.
Each table is shown with pipe-separated cells. The first row is usually the column header.

Map columns to these fields and return ONLY valid JSON: {"streets": [...]}
Each entry: {"main_street": "...", "from_street": "...", "to_street": "...", "work_type": "...", "location": "..."}

Column mapping:
- main_street: THE STREET BEING WORKED ON — first data column. Copy exactly.
- from_street: where work BEGINS — FROM, START, BEGIN, LIMITS FROM, CROSS STREET 1, or similar.
- to_street: where work ENDS — TO, END, TERMINUS, LIMITS TO, CROSS STREET 2, or similar.
- work_type: type of work — use section header if no explicit column.
- location: location/zone/district number if present, otherwise null.

Important: Copy street values exactly — suffixes like AV, ST, DR, BL, RD are distinct and not interchangeable.
Placeholders like EOS, EOC, BOS, BOC, EOP are valid cross-street values, not blanks.
Numeric-only cells are not street names — do not put them in main_street, from_street, or to_street.
Skip header rows and tables with no from/to column. Extract every data row."""


def _get_docai_credentials():
    """Load Document AI service account creds from env var or local key file."""
    from google.oauth2 import service_account
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    creds_json_str = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json_str:
        info = json.loads(creds_json_str)
        return service_account.Credentials.from_service_account_info(info, scopes=scopes)
    if os.path.exists(DOCAI_CRED_FILE):
        return service_account.Credentials.from_service_account_file(DOCAI_CRED_FILE, scopes=scopes)
    return None


def docai_extract_all_tables(pdf_bytes: bytes, log_fn=None, save_raw_path=None) -> dict:
    """
    Send PDF to Document AI Form Parser in 10-page chunks.
    Returns {page_num (1-indexed): [(header_rows, body_rows), ...]}.

    Form Parser returns tables via document.pages[i].tables.
    Cell text is extracted via text anchors into document.text.
    Columns are preserved separately — unlike Layout Parser which merges adjacent columns.
    """
    # Check persistent DocAI cache first
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    cache_path = os.path.join(DOCAI_CACHE_DIR, f"{pdf_hash}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                raw = json.load(f)
            # Deserialize: keys are string page numbers, values are list of {header_rows, body_rows}
            result = {}
            for page_str, tables in raw.items():
                result[int(page_str)] = [(t["header_rows"], t["body_rows"]) for t in tables]
            if log_fn:
                log_fn(f"  💾 DocAI cache hit ({pdf_hash[:8]}…) — {sum(len(v) for v in result.values())} tables across {len(result)} pages")
            # Write raw data to save_raw_path even on cache hit so the DocAI Raw tab works
            if save_raw_path:
                try:
                    with open(save_raw_path, "w") as f:
                        json.dump(raw, f)
                except Exception:
                    pass
            return result
        except Exception as e:
            if log_fn:
                log_fn(f"  ⚠️ DocAI cache load failed: {e} — re-running DocAI")

    from google.cloud import documentai

    credentials = _get_docai_credentials()
    if not credentials:
        raise Exception("No Document AI credentials — set GOOGLE_APPLICATION_CREDENTIALS_JSON or provide key file")

    client = documentai.DocumentProcessorServiceClient(
        credentials=credentials,
        client_options={"api_endpoint": f"{DOCAI_LOCATION}-documentai.googleapis.com"},
    )
    processor_name = f"projects/{DOCAI_PROJECT}/locations/{DOCAI_LOCATION}/processors/{DOCAI_PROCESSOR_ID}"

    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(src)
    src.close()

    CHUNK_SIZE = 10
    all_tables: dict = {}

    def _cell_text(cell_layout, full_text: str) -> str:
        """Extract text from a Form Parser table cell using text anchors into document.text."""
        try:
            segs = cell_layout.text_anchor.text_segments
        except AttributeError:
            return ""
        parts = []
        for seg in segs:
            try:
                start = int(seg.start_index) if seg.start_index is not None else 0
                end   = int(seg.end_index)   if seg.end_index   is not None else 0
                if end > start:
                    parts.append(full_text[start:end])
            except Exception:
                pass
        return "".join(parts).strip()

    def _parse_form_table(table, full_text: str) -> tuple:
        """
        Parse a Form Parser table into (header_rows, body_rows).
        Form Parser natively splits header vs body rows.
        Returns (header_rows, body_rows) each as list of list of strings.
        """
        header_rows = []
        for row in (table.header_rows or []):
            header_rows.append([_cell_text(cell.layout, full_text) for cell in (row.cells or [])])

        body_rows = []
        for row in (table.body_rows or []):
            body_rows.append([_cell_text(cell.layout, full_text) for cell in (row.cells or [])])

        return header_rows, body_rows

    def _process_chunk(chunk_start: int, chunk_end: int):
        """Send one chunk of pages to DocAI Form Parser and add results to all_tables."""
        src_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(src_doc, from_page=chunk_start, to_page=chunk_end - 1)
        chunk_bytes = chunk_doc.tobytes()
        src_doc.close()
        chunk_doc.close()

        req = documentai.ProcessRequest(
            name=processor_name,
            raw_document=documentai.RawDocument(content=chunk_bytes, mime_type="application/pdf"),
        )
        resp = client.process_document(request=req)
        doc_obj = resp.document
        full_text = doc_obj.text or ""  # Form Parser stores all text here; cells reference via anchors

        if log_fn:
            log_fn(f"    Form Parser: {len(doc_obj.pages)} pages, text={len(full_text)} chars")

        # Form Parser: tables live in document.pages[i].tables
        for page in doc_obj.pages:
            # page.page_number is 1-indexed within this chunk
            local_page  = getattr(page, "page_number", None)
            if local_page is None:
                continue
            global_page = chunk_start + local_page
            tables_on_page = getattr(page, "tables", []) or []
            for table in tables_on_page:
                header_rows, body_rows = _parse_form_table(table, full_text)
                if body_rows:
                    all_tables.setdefault(global_page, []).append((header_rows, body_rows))

    for chunk_start in range(0, total_pages, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_pages)
        if log_fn:
            log_fn(f"  🔷 Form Parser: pages {chunk_start+1}–{chunk_end} of {total_pages}...")

        try:
            _process_chunk(chunk_start, chunk_end)
        except Exception as chunk_err:
            # Chunk may be too large (DocAI has a ~40 MB limit per request).
            # Retry by splitting into sub-chunks of 5 pages each.
            sub_size = max(1, (chunk_end - chunk_start) // 3)
            if log_fn:
                log_fn(f"  ⚠️ DocAI chunk pages {chunk_start+1}–{chunk_end} failed — retrying in {sub_size}-page sub-chunks: {str(chunk_err)[:120]}")
            for sub_start in range(chunk_start, chunk_end, sub_size):
                sub_end = min(sub_start + sub_size, chunk_end)
                try:
                    _process_chunk(sub_start, sub_end)
                except Exception as sub_err:
                    if log_fn:
                        log_fn(f"    ⚠️ Sub-chunk pages {sub_start+1}–{sub_end} also failed (skipping): {str(sub_err)[:120]}")

        # Log summary for this chunk
        chunk_pages = [p for p in all_tables if chunk_start < p <= chunk_end]
        if log_fn and chunk_pages:
            for p in sorted(chunk_pages):
                total_rows = sum(len(t[1]) for t in all_tables[p])
                log_fn(f"    Page {p}: {len(all_tables[p])} table(s), {total_rows} rows")

    # Save DocAI result to persistent cache
    try:
        serializable_cache = {}
        for page_num, tables in all_tables.items():
            serializable_cache[str(page_num)] = [
                {"header_rows": hdr, "body_rows": body}
                for hdr, body in tables
            ]
        with open(cache_path, "w") as f:
            json.dump(serializable_cache, f)
        if log_fn:
            log_fn(f"  💾 DocAI result cached ({pdf_hash[:8]}…)")
    except Exception as e:
        if log_fn:
            log_fn(f"  ⚠️ Could not save DocAI cache: {e}")

    # Optionally save raw parsed table data for debugging
    if save_raw_path:
        try:
            serializable = {}
            for page_num, tables in all_tables.items():
                serializable[str(page_num)] = [
                    {"header_rows": hdr, "body_rows": body}
                    for hdr, body in tables
                ]
            with open(save_raw_path, "w") as f:
                json.dump(serializable, f, indent=2)
            if log_fn:
                log_fn(f"  💾 Raw DocAI table data saved to {save_raw_path}")
        except Exception as e:
            if log_fn:
                log_fn(f"  ⚠️ Could not save raw DocAI data: {e}")

    return all_tables


def _parse_llm_json(raw: str) -> dict:
    """Extract JSON from an LLM response — handles markdown fences, trailing text, and extra commentary."""
    import re
    raw = raw.strip()
    # Strip markdown fences first
    if "```" in raw:
        for part in raw.split("```"):
            if part.startswith("json"):
                raw = part[4:].strip(); break
            elif part.strip().startswith("{"):
                raw = part.strip(); break
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Extract the first top-level {...} block — handles trailing text/comments after JSON
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Last resort: find the outermost balanced braces manually
    start = raw.find('{')
    if start != -1:
        depth = 0
        for i, ch in enumerate(raw[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:i+1])
                    except json.JSONDecodeError:
                        break
    raise json.JSONDecodeError("Could not extract valid JSON from LLM response", raw, 0)


def call_gemini_text(prompt: str, text: str, max_retries: int = 6, log_fn=None) -> dict:
    """Call Gemini Flash, alternating to Claude Haiku when either hits rate limits."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
    gemini_payload = json.dumps({
        "contents": [{"parts": [{"text": prompt + "\n\n" + text}]}],
        "generationConfig": {"maxOutputTokens": 65536, "temperature": 0},
    }).encode()

    for attempt in range(max_retries):
        use_claude = (attempt % 2 == 1)  # alternate: Gemini on even, Claude on odd attempts
        try:
            if not use_claude:
                if not gemini_key:
                    use_claude = True
                else:
                    req = urllib.request.Request(gemini_url, data=gemini_payload, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        data = json.loads(resp.read())
                    raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    return _parse_llm_json(raw)

            if use_claude:
                client = anthropic.Anthropic(api_key=anthropic_key)
                msg = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=4096,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt + "\n\n" + text}],
                )
                return _parse_llm_json(msg.content[0].text.strip())

        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if e.code == 429:
                if log_fn:
                    log_fn(f"  ⚠ Gemini rate limit (attempt {attempt+1}) — switching to Claude...")
                time.sleep(5)
            elif attempt < max_retries - 1:
                time.sleep(3)
            else:
                raise Exception(f"Gemini HTTP {e.code}: {body[:200]}")
        except anthropic.RateLimitError:
            if log_fn:
                log_fn(f"  ⚠ Claude rate limit (attempt {attempt+1}) — switching to Gemini...")
            time.sleep(5)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                raise
    raise Exception("LLM max retries exceeded (both Gemini and Claude rate limited)")


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


_GEMINI_PAGE_SCREEN_PROMPT = (
    "You are screening a page from a road construction bid document. "
    "Does this page contain a TABULAR WORK SCHEDULE listing STREET LOCATIONS where paving, slurry seal, resurfacing, or utility adjustment work will be performed? "
    "Answer YES only if the page has a structured multi-column table whose rows each represent a street segment or address where physical work will be done. "
    "This includes: "
    "(A) Street work schedule tables with columns like Street Name, From, To, Begin, End, Limits, Cross Street, Activity, Treatment. "
    "(B) Utility adjustment location lists with address/street rows and a work type column (e.g. Address | Utility Type | Frame Size). "
    "(C) Continuation pages of such tables even if the column headers are not visible — the page has rows of street data in column-aligned tabular format. "
    "Answer NO if ANY of these are true: "
    "(1) The table is a TRUCK ROUTE list, permit route, or hauling route ordinance. "
    "(2) The table is a quantities/cost/bid schedule with no street names or addresses. "
    "(3) The page is a CAD engineering drawing, plan sheet, striping plan, slurry seal plan, or intersection diagram — recognizable by road geometry/line work showing curbs, lanes, and centerlines, with a title block in the corner. Answer NO even if the page contains a small embedded table labeled 'STREETS THIS SHEET', 'QUANTITIES THIS SHEET', 'NO DETAILS THIS SHEET', or similar plan-sheet summary tables. "
    "(4) The table has a STATUS column listing Restricted/Unrestricted — this is a project zones table. "
    "(5) The page is a LANE/SHOULDER CLOSURE REQUEST FORM or contractor form with blank fields to fill in. "
    "(6) The table columns are about traffic control, permits, or administrative data only. "
    "(7) The page is a section cover page, appendix title page, table of contents, or blank divider — even if the title mentions 'locations' or 'list'. "
    "(8) The page contains specification requirements or contractor instructions that MENTION streets by name in prose paragraphs or as a numbered/bulleted list, but those streets are NOT organized as rows in a multi-column table. "
    "(9) The page is an extremely dense pavement management database export — identifiable by having MORE THAN 12 narrow columns packed tightly across the page AND a Latitude/Longitude column. These PMS asset inventory tables typically have columns like C, M, SP, PCI, Supervisorial District, Council District, and hundreds of rows of data. Answer NO for these even if they contain street names. "
    'Reply ONLY with valid JSON: {"is_street_schedule": true} or {"is_street_schedule": false}'
)


def run_extraction(doc_id: str, api_key: str):
    """New pipeline: Gemini Flash page screen → DocAI (filtered pages) → Gemini Pro extraction."""
    doc = documents[doc_id]
    pdf_bytes = doc["bytes"]

    def log(msg, streets_so_far=None):
        p = doc.get("progress") or {"logs": [], "streets_so_far": []}
        p["logs"].append(msg)
        if streets_so_far is not None:
            p["streets_so_far"] = streets_so_far
        doc["progress"] = p

    client = anthropic.Anthropic(api_key=api_key)

    # --- Step 1: Extract project header from first 5 pages (include city & state) ---
    log("📋 Step 1: Extracting project info from cover pages...")
    _HEADER_PROMPT_V2 = """You are parsing a road construction bid document. Extract project-level fields only.
Return ONLY valid JSON with these fields:
- bid_number, project_name, city, state, work_type, estimated_cost, bid_due_date
Use null for any field not found. city and state are the city/state where the work will be performed."""

    header_blocks = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i in range(min(5, len(pdf.pages))):
            text = pdf.pages[i].extract_text() or ""
            header_blocks.append({"type": "text", "text": f"\n--- Page {i+1} ---\n{text}"})
    try:
        schema = call_claude_with_retry(client, _HEADER_PROMPT_V2, header_blocks, max_tokens=1024, log_fn=log)
        log(f"✓ Project: {schema.get('project_name')} | {schema.get('city')}, {schema.get('state')} | bid={schema.get('bid_number')}")
    except Exception as e:
        log(f"✗ Header extraction failed: {e}")
        schema = {}

    city  = schema.get("city") or ""
    state = schema.get("state") or ""
    schema["streets"] = []
    all_streets = []

    # --- Step 2: Gemini Flash page screen — classify every page in parallel ---
    gemini_key = os.environ.get("GEMINI_API_KEY")
    gemini_flash_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
        if gemini_key else None
    )

    total_pages = doc["total_pages"]
    log(f"🔍 Step 2: Gemini Flash page screen — {total_pages} pages in parallel...")

    pdf_screen_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]

    def _screen_page(page_idx: int) -> tuple:
        """Returns (page_num_1indexed, is_street_schedule: bool)."""
        page_num = page_idx + 1
        # Disk cache — survives server restarts
        cache_file = os.path.join(SCREEN_CACHE_DIR, f"{pdf_screen_hash}_p{page_num}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    return page_num, json.load(f)["result"]
            except Exception:
                pass  # corrupt cache entry — re-screen

        b64 = render_page_as_image(pdf_bytes, page_idx, dpi=150)
        payload = json.dumps({
            "contents": [{"parts": [
                {"text": _GEMINI_PAGE_SCREEN_PROMPT},
                {"inline_data": {"mime_type": "image/png", "data": b64}},
            ]}],
            "generationConfig": {"maxOutputTokens": 512, "temperature": 0},
        }).encode()

        # Retry up to 2 times on transient errors (timeouts, broken pipe, etc.)
        last_err = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    gemini_flash_url, data=payload,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                # Gemini 2.5 Flash may include thinking tokens — always use the LAST text part
                parts = data["candidates"][0]["content"].get("parts", [])
                text_parts = [p["text"].strip() for p in parts if isinstance(p, dict) and p.get("text")]
                if not text_parts:
                    log(f"  ⚠ Page {page_num}: no text in response — assuming no")
                    result = False
                else:
                    answer = text_parts[-1]  # last part is always the final answer
                    log(f"  🔎 Page {page_num} answer: {repr(answer[:120])}")
                    answer_lower = answer.lower().replace(" ", "").replace("\n", "")
                    if '"is_street_schedule":true' in answer_lower:
                        result = True
                    elif '"is_street_schedule":false' in answer_lower:
                        result = False
                    elif "true" in answer_lower:
                        result = True
                    else:
                        result = False
                # Cache to disk on success
                try:
                    with open(cache_file, "w") as f:
                        json.dump({"result": result}, f)
                except Exception:
                    pass
                return page_num, result
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(2 ** attempt)  # 1s, 2s backoff
        log(f"  ⚠ Page screen failed p.{page_num} (3 attempts): {str(last_err)[:80]} — assuming no")
        return page_num, False

    selected_pages = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(_screen_page, i): i for i in range(total_pages)}
        results = {}
        for future in as_completed(futures):
            page_num, is_schedule = future.result()
            results[page_num] = is_schedule

    # Expand selections to include continuation pages:
    # If page N is selected, also include up to 4 following pages that weren't explicitly
    # rejected (i.e., they either timed out or are continuations with no visible header).
    # Stop expanding if a page was screened and explicitly returned false.
    raw_selected = {p for p, v in results.items() if v}
    expanded = set(raw_selected)
    for p in sorted(raw_selected):
        for offset in range(1, 5):
            candidate = p + offset
            if candidate > total_pages:
                break
            if candidate in raw_selected:
                break  # already selected, stop
            if results.get(candidate) is False:
                break  # explicitly rejected — stop expanding
            # Not screened (timeout) or adjacent to a selected page — include it
            expanded.add(candidate)

    for page_num in sorted(results):
        verdict = page_num in expanded
        icon = "✅" if verdict else "⏭"
        log(f"  {icon} Page {page_num}: {'STREET SCHEDULE — will send to DocAI' if verdict else 'skip'}")
        if verdict:
            selected_pages.append(page_num)

    log(f"")
    log(f"📊 Page screen complete — {len(selected_pages)}/{total_pages} pages selected for DocAI:")
    log(f"   Pages: {selected_pages}")
    log(f"")

    if not selected_pages:
        log("⚠ No pages selected — nothing to extract.")
        doc["extracted_schema"] = schema
        return

    # --- Step 3: DocAI Form Parser — send each selected page individually in parallel ---
    log(f"🔷 Step 3: Document AI Form Parser — {len(selected_pages)} pages in parallel...")

    from google.cloud import documentai as _docai

    credentials = _get_docai_credentials()
    if not credentials:
        log("✗ No Document AI credentials — set GOOGLE_APPLICATION_CREDENTIALS_JSON")
        doc["extracted_schema"] = schema
        return

    docai_client = _docai.DocumentProcessorServiceClient(
        credentials=credentials,
        client_options={"api_endpoint": f"{DOCAI_LOCATION}-documentai.googleapis.com"},
    )
    processor_name = f"projects/{DOCAI_PROJECT}/locations/{DOCAI_LOCATION}/processors/{DOCAI_PROCESSOR_ID}"

    def _cell_text(cell_layout, full_text: str) -> str:
        try:
            segs = cell_layout.text_anchor.text_segments
        except AttributeError:
            return ""
        parts = []
        for seg in segs:
            try:
                start = int(seg.start_index) if seg.start_index is not None else 0
                end   = int(seg.end_index)   if seg.end_index   is not None else 0
                if end > start:
                    parts.append(full_text[start:end])
            except Exception:
                pass
        return "".join(parts).strip()

    def _parse_form_table(table, full_text: str) -> tuple:
        header_rows = []
        for row in (table.header_rows or []):
            header_rows.append([_cell_text(cell.layout, full_text) for cell in (row.cells or [])])
        body_rows = []
        for row in (table.body_rows or []):
            body_rows.append([_cell_text(cell.layout, full_text) for cell in (row.cells or [])])
        return header_rows, body_rows

    def _docai_single_page(page_num_1indexed: int) -> tuple:
        """Extract tables from a single page via DocAI. Returns (page_num, [(hdr,body),...])."""
        page_idx = page_num_1indexed - 1
        # Check per-page cache using (pdf_hash, page_num) key
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
        page_cache_path = os.path.join(DOCAI_CACHE_DIR, f"{pdf_hash}_p{page_num_1indexed}.json")
        if os.path.exists(page_cache_path):
            try:
                with open(page_cache_path) as f:
                    raw = json.load(f)
                tables = [(t["header_rows"], t["body_rows"]) for t in raw]
                log(f"  💾 DocAI cache hit p.{page_num_1indexed} ({len(tables)} tables)")
                return page_num_1indexed, tables
            except Exception:
                pass
        try:
            src_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            single_doc = fitz.open()
            single_doc.insert_pdf(src_doc, from_page=page_idx, to_page=page_idx)
            page_bytes = single_doc.tobytes()
            src_doc.close()
            single_doc.close()

            req = _docai.ProcessRequest(
                name=processor_name,
                raw_document=_docai.RawDocument(content=page_bytes, mime_type="application/pdf"),
            )
            resp = docai_client.process_document(request=req)
            doc_obj = resp.document
            full_text = doc_obj.text or ""

            tables = []
            for page in doc_obj.pages:
                for table in (getattr(page, "tables", []) or []):
                    hdr, body = _parse_form_table(table, full_text)
                    if body:
                        tables.append((hdr, body))

            log(f"  🔷 DocAI p.{page_num_1indexed}: {len(tables)} table(s)")

            # Cache result
            try:
                serializable = [{"header_rows": h, "body_rows": b} for h, b in tables]
                with open(page_cache_path, "w") as f:
                    json.dump(serializable, f)
            except Exception:
                pass

            return page_num_1indexed, tables
        except Exception as e:
            log(f"  ⚠ DocAI failed p.{page_num_1indexed}: {str(e)[:100]} — skipping")
            return page_num_1indexed, []

    all_page_tables: dict = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_docai_single_page, p): p for p in selected_pages}
        for fut in as_completed(futs):
            pn, tables = fut.result()
            if tables:
                all_page_tables[pn] = tables

    total_tables = sum(len(v) for v in all_page_tables.values())
    total_rows   = sum(len(t[1]) for v in all_page_tables.values() for t in v)
    log(f"✓ DocAI complete — {len(all_page_tables)} pages with tables, {total_tables} tables, {total_rows} body rows")

    if not all_page_tables:
        log("⚠ DocAI found no tables on selected pages.")
        schema["streets"] = []
        schema["_meta"] = {"total_pages": total_pages, "selected_pages": selected_pages,
                           "city": city, "state": state, "total_streets": 0}
        doc["extracted_schema"] = schema
        return

    # --- Step 4: Extract streets (Haiku Vision col-confirm → Gemini 2.5 Pro extraction) ---
    log("🤖 Step 4: Extracting streets — Haiku col-confirm → Gemini 2.5 Pro...")

    STREET_SUFFIXES = {"RD", "AV", "AVE", "DR", "LN", "CT", "PL", "ST", "BL", "BLVD", "WY", "WAY",
                       "TR", "TRL", "CIR", "TER", "ML", "HWY", "PKWY", "FWY"}
    _MAX_UNSCRAMBLE_SUFFIXES = 10

    _STREET_TABLE_HEADER_KW = {
        "STREET", "ROAD", "ROADWAY", "FROM", "TO", "BEGIN", "END",
        "LIMITS", "CROSS", "INTERSECTION", "LOCATION", "WORK", "TREATMENT",
        "SCOPE", "ACTIVITY", "OVERLAY", "SLURRY", "SEAL", "RESURFAC",
    }
    _STREET_KEYWORDS = {
        "FROM", "TO", "START", "END", "BEGIN", "BEGINNING", "LIMITS", "LIMIT",
        "PORTION", "SEGMENT", "STREET", "ROAD", "AVENUE", "CROSS", "NAME",
    }

    def _text_header_filter(header_rows, body_rows):
        for row in header_rows:
            for cell in row:
                if any(w in _STREET_KEYWORDS for w in str(cell or "").upper().split()):
                    return True
        if not header_rows and body_rows:
            for cell in body_rows[0]:
                if any(w in _STREET_KEYWORDS for w in str(cell or "").upper().split()):
                    return True
        return False

    # ── Gemini Vision extraction (used when DocAI dropped the STREET column) ───
    _GEMINI_VISION_TABLE_PROMPT = (
        "You are extracting street work locations from a road construction bid document page.\n\n"
        "PRIMARY SOURCE — use this form parser output as your main data source:\n"
        "{docai_text}\n\n"
        "The page is a structured work schedule table. The form parser captured the LIMITS column "
        "(cross-street pairs like 'CROSS ST A to CROSS ST B') correctly, but is MISSING the leftmost "
        "STREET NAME column because the PDF uses merged/spanning cells that the parser could not read.\n\n"
        "Use the page IMAGE ONLY to identify which street name corresponds to each LIMITS row — "
        "look at the leftmost column in the table where one street name visually spans multiple rows.\n\n"
        "Instructions:\n"
        "- The LIMITS data from the form parser is the source of truth — use it exactly as-is\n"
        "- For each LIMITS row, read the corresponding street name from the image's leftmost column\n"
        "- Split each LIMITS entry on ' to ' to get from_street and to_street\n\n"
        "Rules:\n"
        "- Copy street names exactly as shown in the image\n"
        "- Skip header rows and blank rows\n"
        "- If a limits value is just 'to END', from_street is empty\n\n"
        'Return ONLY valid JSON: {"streets": [{"main_street": "...", "from_street": "...", "to_street": "..."}]}'
    )

    def _extract_page_with_gemini_vision(page_num, header_rows, body_rows):
        """Use Gemini Vision to extract streets directly from the page image."""
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            log(f"  ⚠ p.{page_num}: GEMINI_API_KEY not set")
            return []
        try:
            b64_img = render_page_as_image(pdf_bytes, page_num - 1, dpi=150)
        except Exception as e:
            log(f"  ⚠ p.{page_num}: could not render page image: {e}")
            return []
        log(f"  🖼 p.{page_num}: STREET col missing from DocAI — using Gemini Vision + text...")
        docai_text = json.dumps({"header_rows": header_rows, "body_rows": body_rows}, ensure_ascii=False, indent=2)
        prompt = _GEMINI_VISION_TABLE_PROMPT.replace("{docai_text}", docai_text)
        try:
            result = call_gemini_image(prompt, b64_img, log_fn=log)
        except Exception as e:
            log(f"  ✗ p.{page_num}: Gemini Vision failed: {e}")
            return []
        streets = []
        for s in result.get("streets", []):
            main = (s.get("main_street") or "").strip()
            if not main:
                continue
            streets.append({
                "main_street": main,
                "from_street": s.get("from_street") or None,
                "to_street":   s.get("to_street")   or None,
                "work_type":   None,
                "source": "gemini-vision",
                "page": page_num,
            })
        log(f"  ✓ p.{page_num}: {len(streets)} streets extracted via vision")
        return streets

    # ── Gemini Pro extraction from DocAI table data (no Haiku pre-confirm) ─────
    _OPUS_CHUNK_PROMPT = """You are extracting street work segments from a road construction bid document table.

Table data (header rows + body rows) extracted by a form parser:
{table_data}

Instructions:
1. Identify which column is the PRIMARY STREET being worked on (main_street).
   - Typical headers: STREET NAME, STREET, ROAD, ROADWAY, LOCATION
   - The values should be street names (e.g. "MAIN ST", "JAMBOREE RD")
2. Identify which column is the FROM/BEGIN cross-street (from_street).
   - Typical headers: FROM, BEGIN, START, CROSS STREET 1
3. Identify which column is the TO/END cross-street (to_street).
   - Typical headers: TO, END, CROSS STREET 2
4. Extract ALL data rows.

Rules:
- Copy street names EXACTLY as they appear — do not rename or substitute
- Strip asset IDs and work order numbers (SS-001459-PV1, S2624, etc.) — not street names
- Skip header rows, totals rows, subtotals, and blank rows
- Use "" for missing from_street or to_street
- HEADER PREFIX: Some cells have the column header merged into the value (e.g. "STREET BARRANCA PKWY", "LIMITS JAMBOREE RD TO MAIN ST"). Strip the leading keyword before extracting.
- INVERTED SUFFIX: DocAI sometimes reads multi-line cells with the street type first (e.g. "RD JAMBOREE"). Reorder if needed.
- STACKED ROWS: When a cell contains multiple street names or limits separated by newlines, extract EACH as a separate entry. Row positions match across columns.
- LIMITS column: If a single column contains "FROM ST to TO ST" combined, split on " to " or " TO " to get from_street and to_street.

Return ONLY valid JSON, no markdown:
{{"streets": [{{"main_street": "...", "from_street": "...", "to_street": "..."}}]}}"""

    def _extract_with_gemini_pro(header_rows, body_rows, page_num):
        gemini_key = os.environ.get("GEMINI_API_KEY")
        gemini_pro_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={gemini_key}"
            if gemini_key else None
        )
        table_data = {"header_rows": header_rows, "body_rows": body_rows}
        full_prompt = _OPUS_CHUNK_PROMPT.format(
            table_data=json.dumps(table_data, ensure_ascii=False, indent=2)
        )

        result = None
        for attempt in range(6):
            try:
                payload = json.dumps({
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {"maxOutputTokens": 65536, "temperature": 0},
                }).encode()
                req = urllib.request.Request(gemini_pro_url, data=payload, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data = json.loads(resp.read())
                raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                result = _parse_llm_json(raw)
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 503):
                    log(f"  ⚠ Gemini Pro HTTP {e.code} p.{page_num} attempt {attempt+1} — retrying...")
                    time.sleep(3 * (attempt + 1))
                else:
                    log(f"  ⚠ Gemini Pro HTTP {e.code} p.{page_num} — {e}")
                    break
            except Exception as e:
                log(f"  ⚠ Gemini Pro error p.{page_num} attempt {attempt+1}: {str(e)[:80]}")
                time.sleep(3 * (attempt + 1))

        if result is None:
            log(f"  ⚠ Gemini Pro failed after retries p.{page_num} — falling back to Opus...")
            try:
                content_blocks = [{"type": "text", "text": json.dumps(table_data, ensure_ascii=False, indent=2)}]
                result = call_claude_with_retry(
                    client, full_prompt, content_blocks, max_tokens=8192,
                    model="claude-opus-4-6", log_fn=log,
                )
            except Exception as e:
                log(f"  ✗ Opus fallback also failed p.{page_num}: {e}")
                return []

        streets = []
        for s in result.get("streets", []):
            main = (s.get("main_street") or "").strip()
            if not main:
                continue
            streets.append({
                "main_street": main,
                "from_street": s.get("from_street") or None,
                "to_street":   s.get("to_street")   or None,
                "work_type":   None,
                "source": "gemini-pro",
                "page": page_num,
            })
        log(f"  ✓ p.{page_num}: {len(streets)} streets extracted")
        return streets

    skipped_text  = 0
    sent_to_gemini = 0
    sent_to_vision = 0

    for page_num in sorted(all_page_tables.keys()):
        for table_tuple in all_page_tables[page_num]:
            header_rows, body_rows = table_tuple
            if not body_rows:
                continue

            # Pre-filter: skip DocAI artifact tables (stacked header cells) when a clean sibling exists
            if header_rows and len(body_rows) <= 2:
                first_cell = str(header_rows[0][0]) if header_rows[0] else ""
                _ARTIFACT_KWS = ("STREET\n", "LIMITS\n", "ZONE\n", "STREET NAME\n", "BEGIN\n", "END\n")
                if any(first_cell.startswith(kw) for kw in _ARTIFACT_KWS):
                    page_tables_list = all_page_tables[page_num]
                    has_clean_sibling = any(
                        other_h and other_h[0] and "\n" not in str(other_h[0][0])
                        for (other_h, other_b) in page_tables_list
                        if (other_h, other_b) is not table_tuple and other_b and len(other_b) > 2
                    )
                    if has_clean_sibling:
                        log(f"  ⏩ p.{page_num}: skipping DocAI artifact (clean sibling exists)")
                        skipped_text += 1
                        continue

            # Stacked-header expansion
            # Triggers when the first header cell has 3+ newlines (data rows crammed in)
            # OR starts with a known column keyword.
            # After expansion, if col-0 of the first expanded row is "LIMITS" it means
            # DocAI dropped the STREET column entirely → use Vision fallback.
            header_was_garbled = False
            if header_rows:
                first_cell = str(header_rows[0][0] if header_rows[0] else "")
                _STACK_KWS = ("STREET\n", "LIMITS\n", "ZONE\n", "STREET NAME\n", "BEGIN\n", "END\n",
                              "CROSS STREET\n", "FROM\n", "TO\n")
                first_cell_lines = first_cell.split("\n")
                second_line = first_cell_lines[1].strip() + "\n" if len(first_cell_lines) > 1 else ""
                # Trigger: keyword at start OR keyword on second line
                should_expand = (
                    any(first_cell.startswith(kw) for kw in _STACK_KWS)
                    or any(second_line == kw for kw in _STACK_KWS)
                )
                if should_expand:
                    num_cols = max(len(r) for r in header_rows)
                    split_cols = []
                    for col_i in range(num_cols):
                        cell = str(header_rows[0][col_i]) if col_i < len(header_rows[0]) else ""
                        split_cols.append([p for p in cell.split("\n") if p.strip()])
                    max_data_lines = max((len(p) - 1 for p in split_cols), default=0)
                    if max_data_lines > 0:
                        clean_header = [[p[0] if p else "" for p in split_cols]]
                        expanded_body = []
                        for row_i in range(max_data_lines):
                            expanded_body.append([p[row_i + 1] if row_i + 1 < len(p) else "" for p in split_cols])
                        # Page 218 pattern: after expansion col-0 row 0 = "LIMITS" →
                        # STREET column was dropped entirely → Vision fallback
                        if expanded_body and str(expanded_body[0][0] if expanded_body[0] else "").strip().upper() == "LIMITS":
                            header_was_garbled = True
                        header_rows = clean_header
                        body_rows   = expanded_body + list(body_rows)
                        log(f"  📋 p.{page_num}: expanded stacked header — {max_data_lines} rows from header cells")

            # Text keyword filter
            if not _text_header_filter(header_rows, body_rows):
                skipped_text += 1
                log(f"  ⏩ p.{page_num}: text filter skip")
                continue

            # Garbled header → STREET column was dropped by DocAI → use Gemini Vision
            if header_was_garbled:
                sent_to_vision += 1
                extracted = _extract_page_with_gemini_vision(page_num, header_rows, body_rows)
                all_streets.extend(extracted)
                continue

            # Gemini 2.5 Pro: extract streets from DocAI table data
            sent_to_gemini += 1
            extracted = _extract_with_gemini_pro(header_rows, body_rows, page_num)
            all_streets.extend(extracted)

    log(f"📊 Extraction summary — {skipped_text} text-filtered, {sent_to_vision} vision pages, {sent_to_gemini} tables sent to Gemini")

    # --- Step 5: Deduplicate ---
    log("🔀 Step 5: Deduplicating...")
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

    before = len(all_streets)
    seen = {}
    for s in all_streets:
        key = (
            norm_name(s.get("main_street")),
            norm_name(s.get("from_street")),
            norm_name(s.get("to_street")),
            (s.get("work_type") or "").strip().upper(),
        )
        if key not in seen:
            seen[key] = s
    all_streets = list(seen.values())
    log(f"  Dedup: {before} → {len(all_streets)} streets")

    schema["streets"] = all_streets
    schema["_meta"] = {
        "total_pages": total_pages,
        "selected_pages": selected_pages,
        "city": city,
        "state": state,
        "total_streets": len(all_streets),
    }
    doc["extracted_schema"] = schema
    log(f"✓ Done! {len(all_streets)} streets extracted.", all_streets)


def run_extraction_DISABLED(doc_id: str, api_key: str):
    """OLD pipeline — kept for reference, not called."""
    doc = documents[doc_id]
    pdf_bytes = doc["bytes"]

    def log(msg, streets_so_far=None):
        p = doc.get("progress") or {"logs": [], "streets_so_far": []}
        p["logs"].append(msg)
        if streets_so_far is not None:
            p["streets_so_far"] = streets_so_far
        doc["progress"] = p

    client = anthropic.Anthropic(api_key=api_key)
    schema = {}
    all_streets = []

    # --- Step 2: Extract all tables via Document AI Form Parser ---
    log("📄 Sending to Document AI Form Parser (processor: 8e7372377435d1ba)...")
    try:
        raw_save_path = os.path.join(BASE_DIR, f"docai_raw_{doc_id}.json")
        all_page_tables = docai_extract_all_tables(pdf_bytes, log_fn=log, save_raw_path=raw_save_path)
        total_tables = sum(len(v) for v in all_page_tables.values())
        total_rows   = sum(len(t[1]) for v in all_page_tables.values() for t in v)
        log(f"✓ Form Parser complete — {len(all_page_tables)} pages, {total_tables} tables, {total_rows} body rows")
        for pg in sorted(all_page_tables):
            for i, (hdr, body) in enumerate(all_page_tables[pg]):
                log(f"  📋 Page {pg} table {i+1}: {len(hdr)} header row(s), {len(body)} body rows | headers={[r[:4] for r in hdr[:1]]}")
    except Exception as e:
        log(f"✗ Document AI failed: {e}")
        doc["extracted_schema"] = schema
        return

    if not all_page_tables:
        log("⚠ No tables found in document")
        doc["extracted_schema"] = schema
        return

    # --- Step 3: Map column headers with Gemini, then extract rows in Python ---
    # Collect all unique header rows across all tables
    HEADER_PROMPT_DOCAI = """You are mapping column headers from a road construction bid table to these fields.
Return ONLY valid JSON: {"main_street": <col_index>, "from_street": <col_index>, "to_street": <col_index>, "work_type": <col_index_or_null>, "location": <col_index_or_null>}
Use 0-based column index. Use null if no matching column exists.

Column roles:
- main_street: THE STREET BEING WORKED ON. Typical headers: STREET NAME, STREET, MAIN STREET, ROADWAY, ROAD NAME, ROAD, PRIMARY STREET, STREET/ROAD
- from_street: where work BEGINS or the first cross street. Typical headers: FROM, START, BEGIN, LIMITS FROM, CROSS STREET 1, CROSS ST 1, CROSS STREET FROM, INTERSECTING STREET 1, AT (if only one cross street column exists), BEGIN LOCATION, START LOCATION
- to_street: where work ENDS or the second cross street. Typical headers: TO, END, TERMINUS, LIMITS TO, CROSS STREET 2, CROSS ST 2, CROSS STREET TO, INTERSECTING STREET 2, END LOCATION
- SPECIAL CASE: A column labeled "TO FROM", "FROM TO", or "LIMITS" that contains BOTH endpoints in a single cell should be mapped to from_street (the data will be split later). Do NOT map it to main_street or location.
- work_type: type of work if present. Typical headers: WORK TYPE, WORK, TREATMENT, SCOPE, ACTIVITY, DESCRIPTION
- location: location/zone/district/sequence number if present. Typical headers: LOCATION, LOC, ZONE, DISTRICT, NO, #, SEQ

IMPORTANT: "CROSS STREET 1" always maps to from_street. "CROSS STREET 2" always maps to to_street.
If there is only one cross-street column (e.g. "CROSS STREET" with no number), map it to from_street.
If the table has a column that is clearly a street being intersected, even if labeled differently, map it to from_street or to_street.

Headers:"""

    # Build a set of unique header signatures → ask Gemini once per unique header layout
    header_cache: dict = {}  # header tuple → col_map dict

    def _clean_header_cell(cell: str) -> str:
        """Strip data values from a header cell, keeping only the column label.
        Stops at the first token that looks like data: a digit, a street name,
        a Segment ID (SS-...), or a work order number."""
        first_line = cell.split("\n")[0].strip()
        words = first_line.split()
        label_words = []
        for i, w in enumerate(words):
            wu = w.upper().strip("().,:")
            # Allow a trailing ordinal digit (1 or 2) that disambiguates a column label
            # e.g. "Cross Street 1" → keep "1" so Gemini can distinguish from "Cross Street 2"
            if w in ("1", "2") and label_words:
                label_words.append(w)
                break  # stop after the number — anything after is data
            # Stop if we hit a digit-containing token (measurements, IDs, dates)
            if any(c.isdigit() for c in w):
                break
            # Stop if we hit an all-caps word that looks like a street name value
            if wu in STREET_SUFFIXES:
                if label_words:
                    break
            label_words.append(w)
        return " ".join(label_words).strip() if label_words else first_line

    def get_col_map(header_row: list) -> dict:
        # Clean each header cell to strip out data values mixed in with labels
        flat = [_clean_header_cell(h) for h in header_row]
        key = tuple(h.upper() for h in flat)
        if key in header_cache:
            return header_cache[key]
        header_text = " | ".join(flat)
        log(f"  🔍 Mapping headers: {header_text}")
        try:
            result = call_gemini_text(HEADER_PROMPT_DOCAI, header_text, log_fn=log)
            with _header_cache_lock:
                header_cache[key] = result
            log(f"  📐 Column map: {result}")
            return result
        except Exception as e:
            log(f"  ✗ Header mapping failed: {e}")
            return {}

    STREET_SUFFIXES = {"RD", "AV", "AVE", "DR", "LN", "CT", "PL", "ST", "BL", "BLVD", "WY", "WAY",
                        "TR", "TRL", "CIR", "TER", "ML", "HWY", "PKWY", "FWY"}

    # Max suffixes before we switch from positional unscramble to flexible LLM splitter.
    # Cells with >10 suffixes are too large for "split into exactly N rows" prompting.
    _MAX_UNSCRAMBLE_SUFFIXES = 10

    def _looks_merged(cell: str, ms_idx: int, fr_idx, to_idx) -> bool:
        """Return True if a cell looks like multiple street names jammed together."""
        words = cell.strip().split()
        suffix_count = sum(1 for w in words if w.upper() in STREET_SUFFIXES)
        return suffix_count > 1

    def _unscramble_row_with_llm(row: list, col_map: dict, page_num: int) -> list:
        """Send a merged row to Claude Haiku to split it into individual street records."""
        ms_idx = col_map.get("main_street")
        fr_idx = col_map.get("from_street")
        to_idx = col_map.get("to_street")
        wt_idx = col_map.get("work_type")

        # Build a readable representation of the relevant columns
        col_labels = {ms_idx: "Street Name", fr_idx: "Cross Street 1", to_idx: "Cross Street 2"}
        if wt_idx is not None:
            col_labels[wt_idx] = "Work Type"

        cell_lines = []
        for idx, label in sorted((i, l) for i, l in col_labels.items() if i is not None):
            val = row[idx] if idx < len(row) else ""
            cell_lines.append(f"{label}: {val}")

        # Count N by looking at suffix AND prefix markers in the main street cell.
        # Streets like "PLAZA DOLORES" have no suffix — count PLAZA/VIA/PASEO etc. too.
        STREET_PREFIXES = {"PLAZA", "VIA", "CALLE", "CAMINO", "PASEO", "AVENIDA", "AVNDA", "CAM"}
        main_cell_val = row[ms_idx] if ms_idx < len(row) else ""
        words_upper = [w.upper() for w in main_cell_val.split()]
        n = sum(1 for w in words_upper if w in STREET_SUFFIXES or w in STREET_PREFIXES)
        n = max(n, 2)  # at least 2 if we got here

        prompt = f"""This table row from a road construction bid document has {n} street segments stacked vertically inside each cell.
The values in each cell are listed top-to-bottom and correspond positionally — the 1st value in Street Name goes with the 1st value in Cross Street 1 and Cross Street 2, etc.
Split them into exactly {n} individual records.
Return ONLY valid JSON: {{"rows": [{{"main_street": "...", "from_street": "...", "to_street": "...", "work_type": "..."}}]}}
Use null for work_type if not present. Do not invent values — only use what is given. Do not merge or skip any.

Cells:
"""
        try:
            log(f"  🔀 Unscrambling {n} stacked records on page {page_num}...")
            content_blocks = [{"type": "text", "text": "\n".join(cell_lines)}]
            result = call_claude_with_retry(
                anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")),
                prompt, content_blocks, max_tokens=2048,
                model="claude-haiku-4-5-20251001", log_fn=log
            )
            rows_out = result.get("rows", [])
            streets = []
            for r in rows_out:
                main = (r.get("main_street") or "").strip()
                if not main:
                    continue
                streets.append({
                    "main_street": main,
                    "from_street": r.get("from_street") or None,
                    "to_street":   r.get("to_street") or None,
                    "work_type":   r.get("work_type") or None,
                    "source": "docai+llm",
                    "page": page_num,
                })
            return streets
        except Exception as e:
            log(f"  ✗ Unscramble failed for page {page_num}: {e}")
            return []

    def _split_triple_merged_cell(cell_val: str, page_num: int, col_order: str = "main_from_to") -> list:
        """Split a single merged cell that contains Street Name + Cross Street 1 + Cross Street 2.
        col_order: 'main_from_to' (default) or 'from_main_to' (Cross Street 1 comes before Street Name).
        Also strips work order / asset ID noise (e.g. 'S2624', 'SS-XXXXXX-PV1' tokens).
        Returns a list of street dicts."""
        if col_order == "from_main_to":
            order_hint = """- IMPORTANT: In this document, values appear in this fixed order within the cell: Cross Street 1 (from_street), then Street Name (main_street), then Cross Street 2 (to_street). So the FIRST street name is from_street, the MIDDLE street name is main_street, and the LAST is to_street for each segment."""
        else:
            order_hint = """- The first street name in each segment is main_street, followed by from_street, then to_street."""

        prompt = f"""This cell from a road construction bid document has one or more road segments merged together.
Each segment has: main_street (primary road being worked on), from_street (start/first cross street), to_street (end/second cross street).

Rules:
{order_hint}
- The word "TO" (uppercase, surrounded by spaces) is a separator between from_street and to_street within a limits description. e.g. "CAMPUS DR TO HARVARD AVE" means from_street=CAMPUS DR, to_street=HARVARD AVE.
- When a cell has a pattern like "MAIN_STREET FROM_STREET TO TO_STREET", e.g. "TELLER AVE CAMPUS DR TO DUPONT DR", parse it as main_street=TELLER AVE, from_street=CAMPUS DR, to_street=DUPONT DR.
- Street prefixes like AVNDA, CTE, CAM, VIA, CALLE, CAMINO, PASEO mark the start of a street name.
- Street suffixes like BL, ST, AV, DR, CT, RD, LN, PL, WY, BLVD, PKWY, FWY, HWY mark the end of a street name.
- Ignore non-street tokens: work order numbers (like "S2624", "52624", "2624"), asset IDs (like "SS-XXXXXX-PV1"), and measurement data.
- Use "" for missing from_street or to_street.
- If multiple segments are present, return all of them.

Return ONLY valid JSON: {{"records": [{{"main_street": "...", "from_street": "...", "to_street": "..."}}]}}
Cell: """
        try:
            log(f"  🔀 Splitting triple-merged cell on page {page_num}: {cell_val[:60]}...")
            content_blocks = [{"type": "text", "text": cell_val}]
            result = call_claude_with_retry(
                anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")),
                prompt, content_blocks, max_tokens=1024,
                model="claude-haiku-4-5-20251001", log_fn=log
            )
            streets = []
            for r in result.get("records", []):
                main = (r.get("main_street") or "").strip()
                if not main:
                    continue
                streets.append({
                    "main_street": main,
                    "from_street": r.get("from_street") or None,
                    "to_street":   r.get("to_street") or None,
                    "work_type":   None,
                    "source": "docai+llm",
                    "page": page_num,
                })
            return streets
        except Exception as e:
            log(f"  ✗ Triple-merge split failed for page {page_num}: {e}")
            return []

    def _split_begin_street_name(merged: str, end_val: str, page_num: int) -> tuple:
        """Split a 'BEGIN STREET NAME' merged cell into (main_street, from_street).
        Uses Claude Haiku with the END column as context to disambiguate ordering.
        Returns (main, from_street) — falls back to suffix-split if LLM fails."""
        prompt = """A road construction document has a column called "BEGIN STREET NAME" where two fields were merged into one cell:
- STREET NAME: the primary street being paved/sealed (the main road being worked on)
- BEGIN: the cross-street where the work segment starts

The cell contains two street names concatenated. The END column shows where the work segment ends.

Given the merged cell and the END value, identify which part is STREET NAME (main) and which is BEGIN (from).
The STREET NAME is the road that runs continuously between BEGIN and END cross-streets.

Return ONLY valid JSON: {"main_street": "...", "from_street": "..."}
Do not invent names — only use the exact text from the merged cell."""
        try:
            content_blocks = [{"type": "text", "text": f"Merged cell: {merged}\nEND value: {end_val or '(unknown)'}"}]
            result = call_claude_with_retry(
                anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")),
                prompt, content_blocks, max_tokens=128,
                model="claude-haiku-4-5-20251001", log_fn=log
            )
            main = (result.get("main_street") or "").strip()
            from_s = (result.get("from_street") or "").strip()
            if main and from_s:
                return main, from_s
        except Exception as e:
            log(f"  ✗ BEGIN/STREET NAME split failed p.{page_num}: {e}")
        # Fallback: suffix split, second part = main (BEGIN precedes STREET NAME)
        words = merged.split()
        split_at = next((wi for wi, w in enumerate(words) if w.upper() in STREET_SUFFIXES), None)
        if split_at is not None and split_at < len(words) - 1:
            return " ".join(words[split_at + 1:]), " ".join(words[:split_at + 1])
        return merged, ""

    def apply_col_map(rows: list, col_map: dict, page_num: int) -> list:
        """Apply a column index map to data rows, returning street dicts.
        Detects merged multi-value cells and unscrambles them via LLM."""
        streets = []
        ms_idx = col_map.get("main_street")
        fr_idx = col_map.get("from_street")
        to_idx = col_map.get("to_street")
        wt_idx = col_map.get("work_type")
        lo_idx = col_map.get("location")
        split_from_main = (fr_idx == "_SPLIT_FROM_MAIN")
        if split_from_main:
            fr_idx = None  # will be derived by splitting the main cell
        if ms_idx is None:
            return streets  # can't extract without main street column

        # Detect if from_street column contains merged FROM+TO values (e.g. "Perris Bl Lasselle St").
        # This happens with "To From" / "LIMITS" headers where DocAI collapses two columns into one.
        split_from_to = False
        if fr_idx is not None and to_idx is None:
            sample = [row[fr_idx].strip() for row in rows[:5] if fr_idx < len(row) and row[fr_idx].strip()]
            to_keyword_count = sum(1 for cell in sample if " TO " in cell.upper())
            suffix_count = sum(
                1 for cell in sample
                if sum(1 for w in cell.split() if w.upper() in STREET_SUFFIXES) >= 2
            )
            if to_keyword_count >= 2 or suffix_count >= 2:
                split_from_to = True
                log(f"  🔧 Detected merged from+to column at col {fr_idx} — will split on ' TO ' or first suffix")

        for row in rows:
            # Triple-merged column: entire cell contains "Street Name Cross Street1 Cross Street2"
            # Send straight to LLM to split — no further column-map processing needed.
            if col_map.get("triple_merged") and ms_idx is not None:
                cell_val = row[ms_idx].strip() if ms_idx < len(row) else ""
                if cell_val:
                    streets.extend(_split_triple_merged_cell(cell_val, page_num, col_order=col_map.get("triple_merged_order", "main_from_to")))
                continue

            def get(idx):
                if idx is None or not isinstance(idx, int) or idx >= len(row):
                    return None
                v = row[idx].strip()
                return None if v.replace(",", "").replace(".", "").isdigit() else (v or None)
            main_cell = row[ms_idx] if ms_idx < len(row) else ""
            main = get(ms_idx)
            if not main:
                continue
            # Strip leading sequence number from main_street: "1 COLUMBIA ST" → "COLUMBIA ST"
            # Handles plan-sheet tables where location# is concatenated with street name.
            _main_words = main.split()
            if len(_main_words) >= 2 and _main_words[0].rstrip('.').isdigit():
                main = " ".join(_main_words[1:])
            elif len(_main_words) >= 2 and len(_main_words[0]) <= 3 and _main_words[0][0].isalpha() and _main_words[0][1:].isdigit():
                # handles "A1 VIA MIRALESTE" → "VIA MIRALESTE"
                main = " ".join(_main_words[1:])
            # Strip asset ID prefixes like "SS-001459-PV1 63RD ST" → "63RD ST"
            import re as _re
            main = _re.sub(r'^[A-Za-z]{1,4}-\d{4,8}-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\s+', '', main).strip()
            if not main:
                continue
            # Reject junk rows: numbered/lettered list items ("1.", "2.", "a)", "b)"),
            # totals/subtotals, and repeated header text.
            _main_stripped = main.strip()
            if (
                # Numbered list: "1." "2." "1)" etc.
                (_main_stripped[:2].rstrip(').').isdigit()) or
                # Lettered list: "a)" "b)" "a." etc.
                (len(_main_stripped) >= 2 and _main_stripped[0].isalpha() and _main_stripped[1] in ').' and _main_stripped[0].islower()) or
                # Totals/subtotals
                _main_stripped.upper() in {"TOTAL", "SUBTOTAL", "SUB-TOTAL", "GRAND TOTAL", "TOTAL:", "AVERAGE"} or
                # Repeated header keywords
                _main_stripped.upper() in {"STREET NAME", "STREET", "ROAD", "ROADWAY", "MAIN STREET"}
            ):
                continue
            from_val = get(fr_idx)
            to_val = None

            # ── Row-merge detection (all column modes except triple_merged) ──────
            # DocAI sometimes stacks multiple table rows into one body_row.
            # Detect via multiple street suffixes in the main_street cell.
            # split_from_main rows normally have 2 suffixes (main + from), so require 3+.
            if not col_map.get("triple_merged"):
                _merge_thresh = 3 if split_from_main else 2
                main_suffix_ct = sum(1 for w in main_cell.split() if w.upper() in STREET_SUFFIXES)
                # Also treat main cell as merged when it contains " TO " and multiple suffixes
                # (split_from_to rows where all data is in the main cell, limits column empty)
                _main_has_to = " TO " in main_cell.upper() and main_suffix_ct >= 2
                if main_suffix_ct >= _merge_thresh or (split_from_to and _main_has_to and not from_val):
                    # For split_from_to with TO in main cell and no separate from column — use flexible splitter
                    if split_from_to and _main_has_to and not from_val:
                        streets.extend(_split_triple_merged_cell(main_cell, page_num))
                    elif main_suffix_ct <= _MAX_UNSCRAMBLE_SUFFIXES and (fr_idx is not None or to_idx is not None):
                        streets.extend(_unscramble_row_with_llm(row, col_map, page_num))
                    else:
                        # Too many stacked rows for positional unscramble — use flexible splitter
                        cell_parts = [main_cell]
                        if fr_idx is not None and fr_idx < len(row) and row[fr_idx].strip():
                            cell_parts.append(row[fr_idx])
                        if to_idx is not None and to_idx < len(row) and row[to_idx].strip():
                            cell_parts.append(row[to_idx])
                        streets.extend(_split_triple_merged_cell(" | ".join(p for p in cell_parts if p), page_num))
                    continue

            if split_from_main:
                to_val = get(to_idx)
                main, from_val = _split_begin_street_name(main_cell, to_val, page_num)
            elif split_from_to and from_val:
                # Split "JAMBOREE RD TO CONSTRUCTION S" or "Perris Bl Lasselle St"
                # Prefer explicit " TO " separator; fall back to suffix split
                _fv = from_val
                if " TO " in _fv.upper():
                    _si = _fv.upper().index(" TO ")
                    from_val = _fv[:_si].strip()
                    to_val = _fv[_si + 4:].strip()
                else:
                    words = _fv.split()
                    split_at = next((wi for wi, w in enumerate(words) if w.upper() in STREET_SUFFIXES), None)
                    if split_at is not None and split_at < len(words) - 1:
                        to_val = " ".join(words[split_at + 1:])
                        from_val = " ".join(words[:split_at + 1])
            else:
                to_val = get(to_idx)
            streets.append({
                "main_street": main,
                "from_street": from_val,
                "to_street":   to_val,
                "work_type":   row[wt_idx].strip() if wt_idx is not None and wt_idx < len(row) else None,
                "location":    row[lo_idx].strip() if lo_idx is not None and lo_idx < len(row) else None,
                "source": "docai",
                "page": page_num,
            })
        return streets

    # ── Pre-filter helpers ────────────────────────────────────────────────────

    _STREET_TABLE_HEADER_KW = {
        "STREET", "ROAD", "ROADWAY", "FROM", "TO", "BEGIN", "END",
        "LIMITS", "CROSS", "INTERSECTION", "LOCATION", "WORK", "TREATMENT",
        "SCOPE", "ACTIVITY", "OVERLAY", "SLURRY", "SEAL", "RESURFAC",
    }
    _BODY_SUFFIXES = {
        "ST", "AV", "AVE", "BL", "BLVD", "RD", "DR", "LN", "CT", "PL",
        "WY", "WAY", "CIR", "TER", "HWY", "PKWY", "FWY",
    }

    def _heuristic_is_street_table(header_rows, body_rows):
        """Free instant check — returns True if table looks street-related.
        Checks headers for keywords AND scans body cells for street suffixes.
        Very permissive — only rejects tables with zero street signals."""
        # Check all header cells for street-related keywords
        header_text = " ".join(c.upper() for row in header_rows for c in row if c)
        # Also include first body row (DocAI sometimes puts headers there)
        if body_rows:
            header_text += " " + " ".join(c.upper() for c in body_rows[0] if c)
        if any(kw in header_text for kw in _STREET_TABLE_HEADER_KW):
            return True
        # Scan ALL body rows for any street suffix — even one hit is enough
        for row in body_rows:
            for cell in row:
                words = str(cell or "").upper().split()
                if any(w in _BODY_SUFFIXES for w in words):
                    return True
        return False

    # ── Stage 0: Text keyword filter (free, no API call) ─────────────────────

    _STREET_KEYWORDS = {
        "FROM", "TO", "START", "END", "BEGIN", "BEGINNING",
        "LIMITS", "LIMIT", "PORTION", "SEGMENT",
        "STREET", "ROAD", "AVENUE", "CROSS", "NAME",
    }

    def _text_header_filter(header_rows, body_rows):
        """
        Check header cells for street-table vocabulary.
        Returns False immediately (skip table) if no keywords found.
        """
        for row in header_rows:
            for cell in row:
                words = str(cell or "").upper().split()
                if any(w in _STREET_KEYWORDS for w in words):
                    return True
        # No headers? Fall back to checking first body row
        if not header_rows and body_rows:
            for cell in body_rows[0]:
                words = str(cell or "").upper().split()
                if any(w in _STREET_KEYWORDS for w in words):
                    return True
        return False

    # ── Stage 1: Haiku Vision — is this a street table? ──────────────────────

    _HAIKU_IS_STREET_TABLE_PROMPT = (
        "Look at this page image and table data from a road construction bid document. "
        "Does this table contain a LIST of streets where pavement WORK WILL BE PERFORMED (repaved, slurry sealed, rehabilitated, etc.)? "
        "Answer YES only if ALL of these are true: "
        "(A) The content is a proper GRID TABLE with visible cell borders or clear column/row structure — NOT a formatted list, bulleted list, lettered list (a, b, c...), or two-column prose layout that merely looks like a table. A real table has distinct cells; a formatted list has bullets or letters like 'a) Street Name    Portion...' "
        "(B) The table is a street WORK SCHEDULE listing streets with from/to cross-streets or limits for construction work. "
        "Answer NO if any of these are true: "
        "(1) The page heading, section title, or any bold text near the table says DESIGNATED TRUCK ROUTES, TRUCK ROUTE ORDINANCE, PERMIT ROUTES, HAULING ROUTES, or similar — these are regulatory lists, not work schedules. "
        "(2) The table column header is 'Portion Designated' or 'Name of Street / Portion Designated' — this phrasing is specific to truck route ordinance tables, NOT pavement work schedules. "
        "(3) The rows are labeled with letters like a), b), c)... rather than numbered work items — this is a regulatory list format. "
        "(4) The page is a map, schematic, or striping plan with a small reference/detail callout (SNS details, panel numbers, sign locations, quantities). "
        "(5) The table columns are PANEL NO, SIGN NO, QUANTITY, ITEM, UNIT, DESCRIPTION — not FROM/TO/BEGIN/END. "
        "(6) The table has fewer than 4 rows and is clearly a legend, key, or detail table. "
        "(7) The table contains ZERO street names and is purely numerical/specification data. "
        "(8) Street names only appear in running text/paragraphs, not in a structured table grid with labeled columns. "
        "Default to YES only when the table is clearly a structured pavement work schedule grid. "
        'Reply with ONLY valid JSON: {"is_street_table": true} or {"is_street_table": false}'
    )

    def _haiku_confirms_street_table(header_rows, body_rows, page_num):
        """Haiku Vision call — is this a street table? Fails open. Cached per header signature."""
        # Include first body row in cache key — tables with empty headers need body content to distinguish
        body_sample = tuple(tuple(r) for r in body_rows[:2])
        header_key = ("street_check",) + tuple(tuple(r) for r in header_rows) + body_sample
        with _header_cache_lock:
            if header_key in _vision_header_cache:
                return _vision_header_cache[header_key]

        try:
            b64_img = render_page_as_image(pdf_bytes, max(0, page_num - 1), dpi=120)
        except Exception as e:
            log(f"  ⚠ Could not render page {page_num} for street-table check: {e} — assuming yes")
            return True

        sample = {"header_rows": header_rows, "body_rows": body_rows[:6]}
        text = json.dumps(sample, ensure_ascii=False)
        try:
            headers_preview = str(header_rows[:1])[:120] if header_rows else str(body_rows[:1])[:120]
            log(f"  🔎 Page {page_num}: Haiku (claude-haiku-4-5-20251001) — is this a street table? headers={headers_preview}")
            result = call_vision_with_retry(
                _HAIKU_IS_STREET_TABLE_PROMPT + "\n\n" + text,
                b64_img, max_tokens=64, log_fn=log,
            )
            verdict = bool(result.get("is_street_table", True))
            log(f"  {'✅' if verdict else '⏭'} Page {page_num}: Haiku street-table={verdict} — headers={headers_preview}")
            with _header_cache_lock:
                _vision_header_cache[header_key] = verdict
                _save_vision_cache(_vision_header_cache)
            return verdict
        except Exception as e:
            log(f"  ⚠ Street-table check failed p.{page_num}: {e} — assuming yes")
            return True  # fail open

    # ── Call 1: Haiku Vision — confirm column mapping from image + sample rows ─

    _HEADER_CONFIRM_PROMPT = """You are analyzing a table from a road construction bid document.

Raw table data (headers + first 3 data rows):
{table_sample}

The page image above shows how this table looks visually.

Identify (using 0-based column index):
1. main_col: column containing the PRIMARY STREET being worked on
2. from_col: column for where work BEGINS / first cross-street (FROM, BEGIN, CROSS STREET 1, etc.)
3. to_col: column for where work ENDS / second cross-street (TO, END, CROSS STREET 2, etc.)

IMPORTANT: Look at the actual DATA ROWS, not just the header text. Sometimes the data is in a different order than the header label implies (e.g. "BEGIN STREET NAME" may have the main street first, then the cross street). Trust what you see in the data.

Return ONLY valid JSON:
{"main_col": <int or null>, "from_col": <int or null>, "to_col": <int or null>, "notes": "..."}"""

    def _confirm_headers_with_vision(header_rows, body_rows, page_num):
        """Haiku Vision: confirm column mapping. Cached per unique header+body signature."""
        body_sample = tuple(tuple(r) for r in body_rows[:2])
        header_key = tuple(tuple(r) for r in header_rows) + body_sample
        with _header_cache_lock:
            if header_key in _vision_header_cache:
                cached = _vision_header_cache[header_key]
                log(f"  👁 Page {page_num}: using cached vision header map: {cached}")
                return cached

        try:
            b64_img = render_page_as_image(pdf_bytes, max(0, page_num - 1), dpi=120)
        except Exception as e:
            log(f"  ⚠ Could not render page {page_num} for vision: {e}")
            return {}

        sample = {"header_rows": header_rows, "sample_body_rows": body_rows[:6]}
        prompt = _HEADER_CONFIRM_PROMPT.replace("{table_sample}", json.dumps(sample, ensure_ascii=False, indent=2))

        try:
            log(f"  👁 Page {page_num}: confirming headers with Vision...")
            result = call_vision_with_retry(prompt, b64_img, max_tokens=256, log_fn=log)
            log(f"  ✓ Vision: main={result.get('main_col')} from={result.get('from_col')} to={result.get('to_col')} — {result.get('notes','')}")
            with _header_cache_lock:
                _vision_header_cache[header_key] = result
                _save_vision_cache(_vision_header_cache)
            return result
        except Exception as e:
            log(f"  ✗ Vision header confirmation failed p.{page_num}: {e}")
            return {}

    # ── Call 2: Gemini 2.5 Pro — extract streets from rows + confirmed col map ──

    _OPUS_CHUNK_PROMPT = """You are extracting street work segments from a road construction bid document table.

Column mapping (0-based indices):
- main_street: column {main_col} — the PRIMARY street being worked on
- from_street: column {from_col} — where work BEGINS / first cross-street
- to_street:   column {to_col}   — where work ENDS / second cross-street

Rules:
- Copy street names EXACTLY as they appear — do not rename or substitute
- "TO" (uppercase, surrounded by spaces) between street names separates from_street and to_street
- Strip asset IDs and work order numbers (SS-001459-PV1, S2624, etc.) — not street names
- Skip header rows, totals, subtotals, and non-street data rows
- Use "" for missing from_street or to_street
- HEADER PREFIX: Some cells have the column header word merged into the value (e.g. "STREET BARRANCA PKWY", "LIMITS JAMBOREE RD TO MAIN ST"). Strip the leading column-type keyword (STREET, LIMITS, ROAD, LOCATION, FROM, TO, ZONE) before extracting the actual value.
- INVERTED SUFFIX: DocAI sometimes reads multi-line cells with the street type first (e.g. "RD JAMBOREE", "AVE KELVIN", "BLVD ARTHUR"). If a cell starts with a street suffix (RD, AVE, ST, BLVD, DR, LN, CT, WY, PKWY, HWY, CIR) followed by a name, reorder it (e.g. "RD JAMBOREE" → "JAMBOREE RD").

CRITICAL — STACKED ROWS: When a cell contains multiple street names merged together, extract EACH as a separate segment. Positions match across columns:
  Street cell: "ADAGIO ADELANTE"
  Limits cell: "SAN MARINO to MONTELEGRO LACONIA to PRIMAVERA"
  → TWO segments: {{ADAGIO, SAN MARINO, MONTELEGRO}} and {{ADELANTE, LACONIA, PRIMAVERA}}

CRITICAL — CONCATENATED CELLS: DocAI sometimes dumps multiple rows into a single cell, often separated by asset IDs like SS-001459-PV1. Pattern: "SS-XXXX MAIN_STREET CROSS1 CROSS2 SS-XXXX MAIN_STREET2 CROSS1 CROSS2". Strip the asset IDs and extract each street+cross-street pair as its own segment. The main_street is always the PRIMARY street being worked on — do NOT promote cross-streets to main_street. If you are unsure which is the main street, use the column index confirmed by the vision mapping.

- If from_col and to_col are the same index, that column contains a merged LIMITS/PORTION/SEGMENT cell like "MAIN ST to BROADWAY" or "JAMBOREE RD TO ALTON PKWY" — split on " to " or " TO " to get from_street and to_street. Column headers like LIMITS, PORTION, PORTION DESIGNATED, SEGMENT, LOCATION LIMITS all mean the same thing.

Return ONLY valid JSON, no markdown:
{{"streets": [{{"main_street": "...", "from_street": "...", "to_street": "..."}}]}}"""


    def _extract_chunks_with_opus(header_rows, body_rows, confirmed_cols, page_num):
        """Send all body rows to Gemini 2.5 Pro in a single call with confirmed col mapping."""
        gemini_key = os.environ.get("GEMINI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        gemini_pro_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={gemini_key}" if gemini_key else None

        main_col = confirmed_cols.get("main_col")
        from_col = confirmed_cols.get("from_col")
        to_col   = confirmed_cols.get("to_col")
        vision_notes = confirmed_cols.get("notes", "")

        if main_col is None:
            log(f"  ⚠ Page {page_num}: no main_col confirmed — skipping")
            return []

        # Slice to only the relevant columns so Gemini doesn't hallucinate from
        # quantity/striping columns (LIMIT LINE, STOP EA, TYPE I ARROW, etc.)
        relevant_cols = sorted({c for c in [main_col, from_col, to_col] if c is not None})
        def _slice_rows(rows):
            return [[row[c] if c < len(row) else "" for c in relevant_cols] for row in rows]
        # Remap confirmed col indices to their new positions in the sliced table
        col_remap = {orig: new for new, orig in enumerate(relevant_cols)}
        sliced_header = _slice_rows(header_rows)
        sliced_body   = _slice_rows(body_rows)
        sliced_main = col_remap.get(main_col)
        sliced_from = col_remap.get(from_col) if from_col is not None else None
        sliced_to   = col_remap.get(to_col)   if to_col   is not None else None
        prompt = _OPUS_CHUNK_PROMPT.format(
            main_col=sliced_main,
            from_col=sliced_from if sliced_from is not None else "N/A",
            to_col=sliced_to     if sliced_to   is not None else "N/A",
        )
        if vision_notes:
            prompt += f"\n\nAdditional context from visual inspection: {vision_notes}"
        table_data = {"header_rows": sliced_header, "body_rows": sliced_body}
        table_json = json.dumps(table_data, ensure_ascii=False, indent=2)
        full_prompt = prompt + "\n\n" + table_json

        def _call_gemini_pro():
            payload = json.dumps({
                "contents": [{"parts": [{"text": full_prompt}]}],
                "generationConfig": {"maxOutputTokens": 65536, "temperature": 0},
            }).encode()
            req = urllib.request.Request(gemini_pro_url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
            raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return _parse_llm_json(raw)

        try:
            log(f"  🤖 Page {page_num} ({len(body_rows)} rows) → Gemini 2.5 Pro...")
            result = None
            for attempt in range(6):
                try:
                    result = _call_gemini_pro()
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        log(f"  ⚠ Gemini 2.5 Pro rate limit (attempt {attempt+1}) — retrying...")
                    elif e.code >= 500:
                        log(f"  ⚠ Gemini 2.5 Pro server error {e.code} (attempt {attempt+1}) — retrying...")
                    else:
                        log(f"  ⚠ Gemini 2.5 Pro HTTP error {e.code} (attempt {attempt+1}) — retrying...")
                    time.sleep(3 * (attempt + 1))
                except Exception as e:
                    log(f"  ⚠ Gemini 2.5 Pro error (attempt {attempt+1}): {str(e)[:80]} — retrying...")
                    time.sleep(3 * (attempt + 1))
            if result is None:
                log(f"  ⚠ Gemini 2.5 Pro failed after 6 attempts — falling back to Opus...")
                content_blocks = [{"type": "text", "text": table_json}]
                result = call_claude_with_retry(
                    anthropic.Anthropic(api_key=anthropic_key),
                    prompt, content_blocks, max_tokens=8192,
                    model="claude-opus-4-6", log_fn=log,
                )

            streets = []
            for s in result.get("streets", []):
                main = (s.get("main_street") or "").strip()
                if not main:
                    continue
                streets.append({
                    "main_street": main,
                    "from_street": s.get("from_street") or None,
                    "to_street":   s.get("to_street") or None,
                    "work_type":   None,
                    "source": "gemini-pro",
                    "page": page_num,
                })
            log(f"  ✓ Page {page_num}: {len(streets)} streets")
            return streets
        except Exception as e:
            log(f"  ✗ Extraction failed p.{page_num}: {e}")
            return []

    # ── Main extraction loop ──────────────────────────────────────────────────

    total_doc_pages  = doc["total_pages"]
    total_tables     = sum(len(v) for v in all_page_tables.values())
    skipped_text  = 0
    skipped_haiku = 0
    sent_to_opus  = 0
    opus_pages    = set()

    log("🤖 Extracting streets with new pipeline (text filter → Haiku ×2 → Gemini 2.5 Pro)...")
    for page_num in sorted(all_page_tables.keys()):
        for table_tuple in all_page_tables[page_num]:
            header_rows, body_rows = table_tuple
            if not body_rows:
                continue

            # Pre-filter: skip DocAI artifact tables where the header cell contains
            # a column keyword (STREET/LIMITS/ZONE) followed by stacked data rows,
            # BUT only when a clean sibling table also exists on the same page.
            # If no clean sibling exists, the stacked table IS the real data — don't skip it.
            if header_rows and len(body_rows) <= 2:
                first_cell = str(header_rows[0][0]) if header_rows[0] else ""
                _ARTIFACT_KWS = ("STREET\n", "LIMITS\n", "ZONE\n", "STREET NAME\n", "BEGIN\n", "END\n")
                if any(first_cell.startswith(kw) for kw in _ARTIFACT_KWS):
                    page_tables_list = all_page_tables[page_num]
                    has_clean_sibling = any(
                        other_h and other_h[0] and "\n" not in str(other_h[0][0])
                        for (other_h, other_b) in page_tables_list
                        if (other_h, other_b) is not table_tuple and other_b and len(other_b) > 2
                    )
                    if has_clean_sibling:
                        log(f"  ⏩ p.{page_num}: skipping DocAI artifact table (clean sibling exists on same page)")
                        skipped_text += 1
                        continue

            # Stacked-header expansion: DocAI sometimes packs data rows into header cells
            # e.g. header[0][0] = "STREET\nECCELSTONE CIR\nKENNEDY DR\n..."
            # Expand those embedded lines into additional body rows so Gemini sees them.
            if header_rows:
                first_cell = str(header_rows[0][0] if header_rows[0] else "")
                _STACK_KWS = ("STREET\n", "LIMITS\n", "ZONE\n", "STREET NAME\n", "BEGIN\n", "END\n",
                              "CROSS STREET\n", "FROM\n", "TO\n")
                if any(first_cell.startswith(kw) for kw in _STACK_KWS):
                    num_cols = max(len(r) for r in header_rows)
                    # Split every header cell by newline; first line = column label
                    split_cols = []
                    for col_i in range(num_cols):
                        cell = str(header_rows[0][col_i]) if col_i < len(header_rows[0]) else ""
                        parts = [p for p in cell.split("\n") if p.strip()]
                        split_cols.append(parts)
                    max_data_lines = max((len(p) - 1 for p in split_cols), default=0)
                    if max_data_lines > 0:
                        # Rebuild clean single-line header
                        clean_header = [[p[0] if p else "" for p in split_cols]]
                        # Build expanded body rows from the stacked lines
                        expanded_body = []
                        for row_i in range(max_data_lines):
                            row = []
                            for p in split_cols:
                                row.append(p[row_i + 1] if row_i + 1 < len(p) else "")
                            expanded_body.append(row)
                        header_rows = clean_header
                        body_rows   = expanded_body + list(body_rows)
                        log(f"  📋 p.{page_num}: expanded stacked header — {max_data_lines} rows extracted from header cells")

            # Stage 0: Text keyword filter — free, instant
            if not _text_header_filter(header_rows, body_rows):
                skipped_text += 1
                log(f"  ⏩ p.{page_num}: text filter skip — {str([c for r in header_rows[:1] for c in r])[:80]}")
                continue

            # Stage 1: Haiku Vision — is this a street table?
            if not _haiku_confirms_street_table(header_rows, body_rows, page_num):
                skipped_haiku += 1
                continue

            # Stage 2: Haiku Vision — confirm column mapping
            confirmed_cols = _confirm_headers_with_vision(header_rows, body_rows, page_num)
            if not confirmed_cols or confirmed_cols.get("main_col") is None:
                log(f"  ⏭ p.{page_num}: Haiku could not confirm col mapping — skipping table")
                skipped_haiku += 1
                continue

            # Stage 4: Opus extracts streets in chunks
            sent_to_opus += 1
            opus_pages.add(page_num)
            extracted = _extract_chunks_with_opus(header_rows, body_rows, confirmed_cols, page_num)
            log(f"  ✓ Page {page_num}: {len(extracted)} total streets extracted")
            all_streets.extend(extracted)

    log(
        f"📊 Filter summary — doc pages: {total_doc_pages} | "
        f"tables found by DocAI: {total_tables} | "
        f"skipped (text): {skipped_text} | "
        f"skipped (Haiku): {skipped_haiku} | "
        f"sent to Gemini: {sent_to_opus} tables on {len(opus_pages)} pages"
    )

    # --- Step 4: Deduplication ---
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

    def is_empty(v):
        return not v or str(v).strip() in ("?", "null", "None", "")

    before = len(all_streets)

    seen = {}
    for s in all_streets:
        key = (
            norm_name(s.get("main_street")),
            norm_name(s.get("from_street")),
            norm_name(s.get("to_street")),
            (s.get("work_type") or "").strip().upper(),
        )
        if key not in seen:
            seen[key] = s
    all_streets = list(seen.values())
    log(f"  Dedup: {before} → {len(all_streets)} streets")

    schema["streets"] = all_streets
    schema["_meta"] = {
        "total_pages": doc["total_pages"],
        "table_pages_found": len(all_page_tables),
        "total_streets": len(all_streets),
    }
    doc["extracted_schema"] = schema
    log(f"✓ Done! {len(all_streets)} streets extracted.", all_streets)



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


@app.get("/doc/{doc_id}/docai_raw")
async def get_docai_raw(doc_id: str):
    """Return the raw Document AI parsed table data saved during extraction."""
    path = os.path.join(BASE_DIR, f"docai_raw_{doc_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Raw DocAI data not found — run extraction first")
    with open(path) as f:
        return json.load(f)


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


@app.post("/parse")
async def parse_pdf(
    file: UploadFile = File(...),
    x_api_key: str = Header(default=None),
):
    """
    Single-call endpoint for external integrations.
    Send a PDF as multipart/form-data (field name: 'file').
    Optionally pass X-Api-Key header matching env var PARSER_API_KEY.
    Returns the full extraction result synchronously.
    """
    parser_api_key = os.environ.get("PARSER_API_KEY")
    if parser_api_key and x_api_key != parser_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Api-Key header")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

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
        "progress": {"logs": [], "streets_so_far": []},
    }

    # Run extraction synchronously (blocks until complete)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_extraction, doc_id, anthropic_key)

    result = documents[doc_id].get("extracted_schema")
    if result is None:
        raise HTTPException(status_code=500, detail="Extraction failed — check server logs")

    # Clean up in-memory doc to avoid memory leak
    del documents[doc_id]

    return result


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
