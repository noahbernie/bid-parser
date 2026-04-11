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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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


# ─── Document AI Layout Parser ───────────────────────────────────────────────
DOCAI_PROJECT      = "bid-parser-492923"
DOCAI_LOCATION     = "us"
DOCAI_PROCESSOR_ID = "7bb46b34cc5383cf"
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
    Send PDF to Document AI Layout Parser in 15-page chunks.
    Returns {page_num (1-indexed): [list of table row lists]}.
    Each table is a list of rows; each row is a list of cell strings.

    Layout Parser returns data in document_layout.blocks (not document.pages).
    Each block has table_block / text_block / list_block and a page_span.
    """
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

    def _cell_text(cell) -> str:
        """Extract text from a Layout Parser table cell, preserving newlines between blocks."""
        parts = []
        for block in cell.blocks:
            if block.text_block.text:
                parts.append(block.text_block.text.strip())
        return "\n".join(parts).strip()

    def _parse_table_block(table_block, page_offset: int) -> tuple:
        """
        Parse a Layout Parser table_block into (header_rows, body_rows).
        Uses Document AI's native header/body split — no guessing needed.
        Returns (header_rows, body_rows) each as list of list of strings.
        """
        header_rows = [[_cell_text(cell) for cell in row.cells] for row in table_block.header_rows]
        body_rows   = [[_cell_text(cell) for cell in row.cells] for row in table_block.body_rows]
        return header_rows, body_rows

    def _process_chunk(chunk_start: int, chunk_end: int):
        """Send one chunk of pages to DocAI and add results to all_tables. Returns True on success."""
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
        layout = doc_obj.document_layout

        def _recurse_blocks(blocks):
            for block in blocks:
                has_table = block.table_block.body_rows or block.table_block.header_rows
                if has_table:
                    local_page = block.page_span.page_start
                    global_page = chunk_start + local_page
                    header_rows, body_rows = _parse_table_block(block.table_block, chunk_start)
                    if body_rows:
                        yield global_page, (header_rows, body_rows)
                if block.text_block.text is not None:
                    children = list(block.text_block.blocks)
                    if children:
                        yield from _recurse_blocks(children)

        for global_page, table_tuple in _recurse_blocks(layout.blocks):
            all_tables.setdefault(global_page, []).append(table_tuple)

    for chunk_start in range(0, total_pages, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_pages)
        if log_fn:
            log_fn(f"  DocAI: pages {chunk_start+1}–{chunk_end} of {total_pages}...")

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


def call_gemini_text(prompt: str, text: str, max_retries: int = 4, log_fn=None) -> dict:
    """Call Gemini Flash with a text-only prompt."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise Exception("GEMINI_API_KEY not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt + "\n\n" + text}]}],
        "generationConfig": {"maxOutputTokens": 65536, "temperature": 0},
    }).encode()
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            if "```" in raw:
                for part in raw.split("```"):
                    if part.startswith("json"):
                        raw = part[4:].strip(); break
                    elif part.strip().startswith("{"):
                        raw = part.strip(); break
            return json.loads(raw)
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if e.code == 429:
                wait = 30 * (2 ** attempt)
                if log_fn:
                    log_fn(f"  ⚠ Gemini rate limit — waiting {wait}s...")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(3)
            else:
                raise Exception(f"Gemini HTTP {e.code}: {body[:200]}")
        except Exception:
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
    """Extract streets using Document AI for table extraction + Gemini for column mapping."""
    doc = documents[doc_id]
    pdf_bytes = doc["bytes"]

    def log(msg, streets_so_far=None):
        p = doc.get("progress") or {"logs": [], "streets_so_far": []}
        p["logs"].append(msg)
        if streets_so_far is not None:
            p["streets_so_far"] = streets_so_far
        doc["progress"] = p

    client = anthropic.Anthropic(api_key=api_key)

    # --- Step 1: Extract project header from first 5 pages ---
    log("Extracting project info from cover pages...")
    header_blocks = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i in range(min(5, len(pdf.pages))):
            text = pdf.pages[i].extract_text() or ""
            header_blocks.append({"type": "text", "text": f"\n--- Page {i+1} ---\n{text}"})
    try:
        schema = call_claude_with_retry(client, HEADER_PROMPT, header_blocks, max_tokens=1024, log_fn=log)
        log(f"✓ Project: {schema.get('project_name')} | {schema.get('city')} | {schema.get('bid_number')}")
    except Exception as e:
        log(f"✗ Header extraction failed: {e}")
        return

    schema["streets"] = []
    all_streets = []

    # --- Step 2: Extract all tables via Document AI Layout Parser ---
    log("📄 Sending to Document AI Layout Parser...")
    try:
        raw_save_path = os.path.join(BASE_DIR, f"docai_raw_{doc_id}.json")
        all_page_tables = docai_extract_all_tables(pdf_bytes, log_fn=log, save_raw_path=raw_save_path)
        log(f"✓ Document AI complete — {len(all_page_tables)} pages with tables found")
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
        for w in words:
            wu = w.upper().strip("().,:")
            # Stop if we hit a digit-containing token (measurements, IDs, dates)
            if any(c.isdigit() for c in w):
                break
            # Stop if we hit an all-caps word that looks like a street name value
            # (i.e. is a street suffix or is 2+ caps letters that aren't a known label word)
            if wu in STREET_SUFFIXES:
                # If we already have label words, this suffix is part of a street name value
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

    def _split_triple_merged_cell(cell_val: str, page_num: int) -> list:
        """Split a single merged cell that contains Street Name + Cross Street 1 + Cross Street 2.
        Also strips work order / asset ID noise (e.g. 'S2624', 'SS-XXXXXX-PV1' tokens).
        Returns a list of street dicts."""
        prompt = """This cell from a road construction bid document has one or more road segments merged together.
Each segment has: main_street (primary road being worked on), from_street (start/first cross street), to_street (end/second cross street).

Rules:
- Street prefixes like AVNDA, CTE, CAM, VIA, CALLE, CAMINO, PASEO mark the start of a street name.
- Street suffixes like BL, ST, AV, DR, CT, RD, LN, PL, WY mark the end of a street name.
- Ignore non-street tokens: work order numbers (like "S2624", "52624", "2624"), and asset IDs (like "SS-XXXXXX-PV1", hyphenated codes starting with SS-).
- Use "" for missing from_street or to_street.
- If multiple segments are present, return all of them.

Return ONLY valid JSON: {"records": [{"main_street": "...", "from_street": "...", "to_street": "..."}]}
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
            merged_count = sum(
                1 for cell in sample
                if sum(1 for w in cell.split() if w.upper() in STREET_SUFFIXES) >= 2
            )
            if merged_count >= 2:
                split_from_to = True
                log(f"  🔧 Detected merged from+to column at col {fr_idx} — will split on first suffix")

        for row in rows:
            # Triple-merged column: entire cell contains "Street Name Cross Street1 Cross Street2"
            # Send straight to LLM to split — no further column-map processing needed.
            if col_map.get("triple_merged") and ms_idx is not None:
                cell_val = row[ms_idx].strip() if ms_idx < len(row) else ""
                if cell_val:
                    streets.extend(_split_triple_merged_cell(cell_val, page_num))
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
            main = re.sub(r'^[A-Za-z]{1,4}-\d{4,8}-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\s+', '', main).strip()
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
                if main_suffix_ct >= _merge_thresh:
                    if main_suffix_ct <= _MAX_UNSCRAMBLE_SUFFIXES and (fr_idx is not None or to_idx is not None):
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
                # Split "Perris Bl Lasselle St" → from="Perris Bl", to="Lasselle St"
                words = from_val.split()
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

    log("🧠 Mapping column headers with Gemini, then extracting rows in Python...")
    for page_num in sorted(all_page_tables.keys()):
        for table_tuple in all_page_tables[page_num]:
            header_rows, body_rows = table_tuple
            if not body_rows:
                continue
            # Use Document AI's native header rows — combine all header rows into one flat header
            # (some tables have 2-row headers; join them with space)
            if header_rows:
                header_row = [
                    " ".join(filter(None, [header_rows[i][col] if i < len(header_rows) and col < len(header_rows[i]) else "" for i in range(len(header_rows))]))
                    for col in range(max(len(r) for r in header_rows))
                ]
            else:
                # No explicit header row from DocAI.
                # Skip leading title/section rows and find the first row that looks like a real header.
                # A real header row contains at least one cell matching a known column keyword.
                _HEADER_KEYWORDS = {
                    "STREET", "NAME", "ROAD", "ROADWAY", "FROM", "TO", "BEGIN", "END",
                    "START", "TERMINUS", "LIMITS", "LOCATION", "CROSS", "WORK", "TREATMENT",
                    "SCOPE", "ACTIVITY", "TYPE", "DESCRIPTION", "ZONE", "DISTRICT", "WARD",
                    "SUBZONE", "AT", "SECTION",
                }
                def _looks_like_header(row):
                    if sum(1 for c in row if c.strip()) <= 1:
                        return False
                    # A header row has at least one SHORT cell (≤2 words) that IS a keyword,
                    # OR a cell that exactly matches a known multi-word header phrase.
                    # This rejects title rows like "SEAL STREET DATA" where "STREET" appears
                    # inside a descriptive phrase — those are section titles, not column labels.
                    _EXACT_HEADERS = {
                        "STREET NAME", "STREET", "ROAD", "ROADWAY", "MAIN STREET",
                        "FROM", "TO", "BEGIN", "END", "START", "LIMITS", "AT",
                        "LOCATION", "LOC", "ZONE", "DISTRICT", "WARD", "SUBZONE",
                        "WORK TYPE", "WORK", "TREATMENT", "SCOPE", "TYPE",
                        "CROSS STREET", "CROSS STREET 1", "CROSS STREET 2",
                        "BEGIN LOCATION", "END LOCATION", "NO", "NO.", "#", "SEQ",
                    }
                    _HEADER_KW_EXTENDED = _HEADER_KEYWORDS | {
                        "NAME", "NUMBER", "STREET", "LOC", "ADDR", "ADDRESS",
                    }
                    for cell in row:
                        cu = cell.strip().upper()
                        # Exact match to known header phrase
                        if cu in _EXACT_HEADERS:
                            return True
                        words = cu.split()
                        # Single-word keyword
                        if len(words) == 1 and words[0].strip("().,:#") in _HEADER_KEYWORDS:
                            return True
                        # Short cell (≤4 words) where ≥2 are header keywords
                        # catches "Name Location Street", "Street Name Begin", etc.
                        if len(words) <= 4:
                            kw_count = sum(1 for w in words if w.strip("().,:#") in _HEADER_KW_EXTENDED)
                            if kw_count >= 2:
                                return True
                        # Long merged cell (>4 words) with ≥4 keyword matches
                        # catches "Street Name Cross Street 2 Cross Street 1" (DocAI-merged header)
                        elif len(words) > 4:
                            kw_count = sum(1 for w in words if w.strip("().,:#") in _HEADER_KW_EXTENDED)
                            if kw_count >= 4:
                                return True
                    return False

                # Skip up to 10 non-header rows (plan-sheet tables can have many garbage header rows)
                for _ in range(10):
                    if not body_rows:
                        break
                    if _looks_like_header(body_rows[0]):
                        break
                    body_rows = body_rows[1:]
                if not body_rows:
                    continue
                raw_first = body_rows[0]
                header_row = [cell.split("\n")[0] for cell in raw_first]
                body_rows = body_rows[1:]

            col_map = get_col_map(header_row)
            if not col_map:
                log(f"  ⚠️ Page {page_num}: no column map returned, skipping table")
                continue
            if col_map.get("main_street") is None:
                # Check if from_street col has cells with 2 streets merged (e.g. "COLUMBIA ST FORDHAM AV")
                # This happens when a "BEGIN STREET NAME" column contains both main + from street.
                fr_col = col_map.get("from_street")
                if fr_col is not None:
                    sample = [row[fr_col].strip() for row in body_rows[:5] if fr_col < len(row) and row[fr_col].strip()]
                    merged_cols = sum(
                        1 for cell in sample
                        if sum(1 for w in cell.split() if w.upper() in STREET_SUFFIXES) >= 2
                    )
                    if merged_cols >= 2:
                        # from_street col actually contains "MAIN_STREET FROM_STREET" — split on first suffix
                        col_map["main_street"] = fr_col
                        col_map["from_street"] = "_SPLIT_FROM_MAIN"
                        log(f"  🔧 Detected merged main+from column at col {fr_col} — will split on suffix")
                    else:
                        log(f"  ⚠️ Page {page_num}: main_street column not found in map {col_map}, skipping table")
                        continue
                else:
                    log(f"  ⚠️ Page {page_num}: main_street column not found in map {col_map}, skipping table")
                    continue
            # Detect "triple-merged" column: DocAI merged Street Name + Cross Street 1 + Cross Street 2
            # into a single column. Pattern: main_street is set, from/to are null, and the main_street
            # header cell mentions "Cross Street". Route each data cell to the LLM to split.
            if (col_map.get("main_street") is not None and
                    col_map.get("from_street") is None and
                    col_map.get("to_street") is None):
                ms_col_idx = col_map["main_street"]
                if ms_col_idx < len(header_row):
                    ms_hdr_upper = header_row[ms_col_idx].upper()
                    if "CROSS STREET" in ms_hdr_upper or "CROSS ST " in ms_hdr_upper:
                        col_map["triple_merged"] = True
                        log(f"  🔀 Triple-merged column detected at col {ms_col_idx}: '{header_row[ms_col_idx]}'")

            # If we found main_street but no cross streets (and it's not triple-merged),
            # check if header text mentions "Cross Street" — try to infer separate column positions.
            if not col_map.get("triple_merged") and col_map.get("from_street") is None and col_map.get("to_street") is None:
                joined = " | ".join(h.upper() for h in header_row)
                if "CROSS STREET" in joined or "STREET NAME" in joined:
                    # Find indices of cells containing "Street Name", "Cross Street 1/2"
                    for ci, h in enumerate(header_row):
                        hu = h.upper()
                        if "STREET NAME" in hu and col_map.get("main_street") is None:
                            col_map["main_street"] = ci
                        elif "CROSS STREET 1" in hu or "CROSS ST 1" in hu:
                            col_map["from_street"] = ci
                        elif "CROSS STREET 2" in hu or "CROSS ST 2" in hu:
                            col_map["to_street"] = ci
                        elif "CROSS STREET" in hu and "1" not in hu and "2" not in hu:
                            if col_map.get("from_street") is None:
                                col_map["from_street"] = ci
                    log(f"  🔧 Inferred cross-street columns: {col_map}")
            streets = apply_col_map(body_rows, col_map, page_num)
            all_streets.extend(streets)
            log(f"  ✓ Page {page_num} table: {len(body_rows)} rows → {len(streets)} streets extracted (map={col_map})")

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
