"""
Eval API routes — /eval/*
Powers the eval dashboard at /eval
"""
import asyncio
import io
import json
import os
import re
import sys
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pdfplumber
import shutil
from fastapi import APIRouter
from fastapi.responses import FileResponse

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR   = os.path.join(BASE_DIR, "static")
TESTS_DIR    = os.path.join(BASE_DIR, "tests")
GT_DIR       = os.path.join(TESTS_DIR, "ground_truth")
PDF_DIR      = os.path.join(TESTS_DIR, "pdfs")
CACHE_DIR    = os.environ.get("EVAL_CACHE_DIR", os.path.join(TESTS_DIR, "cache"))
COL_MAP      = os.path.join(TESTS_DIR, "column_map.json")
HISTORY_FILE = os.path.join(CACHE_DIR, "_history.json")
LEDGER_FILE  = os.path.join(CACHE_DIR, "_ledger.json")

sys.path.insert(0, TESTS_DIR)
from eval import load_ground_truth, match_streets  # noqa: E402

router = APIRouter(prefix="/eval")

# ── In-memory state ───────────────────────────────────────────────────────────

_jobs    = {}   # job_id  -> job dict
_results = {}   # doc_key -> serialized result dict

# Dedicated thread pool — one thread per parallel eval job
_eval_executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix="eval")

def _load_cached_results():
    os.makedirs(CACHE_DIR, exist_ok=True)
    for fn in os.listdir(CACHE_DIR):
        if fn.endswith("_eval.json"):
            key = fn[:-len("_eval.json")]
            path = os.path.join(CACHE_DIR, fn)
            try:
                with open(path) as f:
                    _results[key] = json.load(f)
            except Exception:
                pass

_load_cached_results()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _doc_keys():
    return [
        os.path.splitext(fn)[0]
        for fn in sorted(os.listdir(GT_DIR))
        if fn.endswith(".xlsx")
    ]

def _parser_cache_path(doc_key):
    return os.path.join(CACHE_DIR, doc_key + ".json")

def _eval_cache_path(doc_key):
    return os.path.join(CACHE_DIR, doc_key + "_eval.json")

def _cache_mtime(path):
    if os.path.exists(path):
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
    return None

def _load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []

def _append_history(doc_key, result):
    history = _load_history()
    history.append({
        "timestamp": result["timestamp"],
        "doc_key":   doc_key,
        "f1":        result["f1"],
        "precision": result["precision"],
        "recall":    result["recall"],
        "matched":   result["matched_count"],
        "missed":    result["missed_count"],
        "extra":     result["extra_count"],
    })
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-200:], f)

def _parse_progress(logs):
    """Extract structured progress info from log lines."""
    p = {"phase": "loading", "total_pages": 0, "chunks_done": 0,
         "total_chunks": 0, "streets_found": 0}
    for line in logs:
        if "Scanning all pages" in line:
            p["phase"] = "scanning"
        if "Scan complete:" in line:
            m = re.search(r"(\d+) pages total", line)
            if m: p["total_pages"] = int(m.group(1))
        if "Launching" in line and "image chunk" in line:
            p["phase"] = "parsing"
            m = re.search(r"(\d+) image chunk", line)
            if m: p["total_chunks"] = int(m.group(1))
        if "IMG" in line and "/4 " in line or "IMG" in line and "/3 " in line or "IMG" in line and "/2 " in line:
            m = re.search(r"\[IMG \d+/(\d+)", line)
            if m: p["total_chunks"] = max(p["total_chunks"], int(m.group(1)))
        if "finished —" in line and "partial total" in line:
            m = re.search(r"partial total: (\d+)", line)
            if m: p["streets_found"] = int(m.group(1))
            m2 = re.search(r"\[IMG (\d+)\]", line)
            if m2: p["chunks_done"] = max(p["chunks_done"], int(m2.group(1)))
        if "Matching streets" in line:
            p["phase"] = "matching"
        if "Done —" in line:
            p["phase"] = "done"
    return p

def _ts():
    return datetime.now().strftime("%H:%M:%S")

# ── Background eval runner ────────────────────────────────────────────────────

def _run_eval(job_id: str, doc_key: str, force_reparse: bool):
    job = _jobs[job_id]

    def log(msg, level="info"):
        entry = {"ts": _ts(), "msg": msg, "level": level}
        job["log"].append(entry)
        job["progress"] = _parse_progress([e["msg"] for e in job["log"]])
        print(f"[eval/{doc_key[:28]}] {msg}")

    try:
        log("Loading ground truth...")
        with open(COL_MAP) as f:
            col_map = json.load(f)
        truth = load_ground_truth(doc_key, col_map)
        if not truth:
            job["status"] = "error"
            job["error"] = "No ground truth rows — check column_map.json"
            return
        log(f"✓ {len(truth)} ground truth streets loaded", "success")

        parser_cache = _parser_cache_path(doc_key)
        if not force_reparse and os.path.exists(parser_cache):
            log("Loading parser cache (skipping re-parse)...")
            with open(parser_cache) as f:
                cached = json.load(f)
            # Support old format (plain list) and new format (dict with streets key)
            if isinstance(cached, list):
                parsed = cached
            else:
                parsed = cached.get("streets", [])
            log(f"✓ {len(parsed)} streets from cache", "success")
            job["progress"]["phase"] = "matching"

            # Populate docai_raw in eval cache from persistent docai_cache if not already there
            raw_dst = os.path.join(CACHE_DIR, doc_key + "_docai_raw.json")
            if not os.path.exists(raw_dst):
                pdf_path = os.path.join(PDF_DIR, doc_key + ".pdf")
                if os.path.exists(pdf_path):
                    try:
                        import hashlib as _hl
                        with open(pdf_path, "rb") as _f:
                            _pdf_hash = _hl.sha256(_f.read()).hexdigest()
                        _main_module = os.environ.get("MAIN_MODULE", "main_opus")
                        import importlib as _il
                        _m = _il.import_module(_main_module)
                        _docai_src = os.path.join(_m.DOCAI_CACHE_DIR, f"{_pdf_hash}.json")
                        if os.path.exists(_docai_src):
                            shutil.copy(_docai_src, raw_dst)
                    except Exception:
                        pass
        else:
            pdf_path = os.path.join(PDF_DIR, doc_key + ".pdf")
            if not os.path.exists(pdf_path):
                job["status"] = "error"
                job["error"] = f"PDF not found: tests/pdfs/{doc_key}.pdf"
                return

            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                job["status"] = "error"
                job["error"] = "ANTHROPIC_API_KEY not set"
                return

            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            _main_module = os.environ.get("MAIN_MODULE", "main_opus")
            import importlib as _il
            _m = _il.import_module(_main_module)
            run_extraction = _m.run_extraction
            docs_store = _m.documents
            import time as _time

            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                total = len(pdf.pages)

            log(f"Parsing {doc_key}.pdf  ({total} pages)...")
            job["progress"]["total_pages"] = total

            doc_id = "eval_" + str(uuid.uuid4())[:8]
            docs_store[doc_id] = {
                "filename": doc_key + ".pdf",
                "total_pages": total,
                "bytes": pdf_bytes,
                "page_cache": {},
                "extracted_schema": None,
                "progress": {"logs": [], "streets_so_far": []},
            }

            def _tail_parser_logs():
                seen = 0
                while docs_store.get(doc_id, {}).get("extracted_schema") is None:
                    plogs = docs_store.get(doc_id, {}).get("progress", {}).get("logs", [])
                    for line in plogs[seen:]:
                        log(f"[parser] {line}", "parser")
                    seen = len(plogs)
                    _time.sleep(0.8)

            tail = threading.Thread(target=_tail_parser_logs, daemon=True)
            tail.start()
            run_extraction(doc_id, api_key)
            tail.join(timeout=3)

            # flush remaining
            plogs = docs_store.get(doc_id, {}).get("progress", {}).get("logs", [])
            seen_msgs = {e["msg"] for e in job["log"]}
            for line in plogs:
                msg = f"[parser] {line}"
                if msg not in seen_msgs:
                    log(msg, "parser")

            schema = docs_store.get(doc_id, {}).get("extracted_schema") or {}
            parsed = schema.get("streets", [])
            if doc_id in docs_store:
                del docs_store[doc_id]

            # Persist raw DocAI output keyed by doc_key for the diagnose endpoint
            raw_src = os.path.join(BASE_DIR, f"docai_raw_{doc_id}.json")
            raw_dst = os.path.join(CACHE_DIR, doc_key + "_docai_raw.json")
            if os.path.exists(raw_src):
                shutil.move(raw_src, raw_dst)

            _meta = schema.get("_meta", {})
            with open(parser_cache, "w") as f:
                json.dump({"streets": parsed, "_meta": _meta}, f, indent=2)
            log(f"✓ {len(parsed)} streets extracted", "success")

        log("Matching streets against ground truth...")
        r = match_streets(truth, parsed)

        serialized = {
            "doc_key":       doc_key,
            "timestamp":     datetime.now().isoformat(),
            "logs":          list(job["log"]),
            "total_truth":   r["total_truth"],
            "total_parsed":  r["total_parsed"],
            "precision":     round(r["precision"], 4),
            "recall":        round(r["recall"], 4),
            "f1":            round(r["f1"], 4),
            "matched_count": len(r["matched"]),
            "missed_count":  len(r["missed"]),
            "extra_count":   len(r["extra"]),
            "matched": [{"truth": m["truth"], "parsed": m["parsed"], "score": m["score"]}
                        for m in r["matched"]],
            "missed":  r["missed"],
            "extra":   r["extra"],
        }

        with open(_eval_cache_path(doc_key), "w") as f:
            json.dump(serialized, f, indent=2)
        _results[doc_key] = serialized
        _append_history(doc_key, serialized)

        log(f"✓ Done  —  P {r['precision']:.0%}  R {r['recall']:.0%}  F1 {r['f1']:.0%}", "success")
        job["status"] = "done"
        job["progress"]["phase"] = "done"
        job["result"] = {k: v for k, v in serialized.items()
                         if k not in ("matched", "missed", "extra")}

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["log"].append({"ts": _ts(), "msg": f"ERROR: {e}", "level": "error"})
        traceback.print_exc()

# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/")
async def eval_dashboard():
    return FileResponse(
        os.path.join(STATIC_DIR, "eval.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@router.get("/docs")
async def list_eval_docs():
    docs = []
    for key in _doc_keys():
        res = _results.get(key)
        docs.append({
            "key":               key,
            "has_parser_cache":  os.path.exists(_parser_cache_path(key)),
            "has_eval_cache":    res is not None,
            "parser_cache_time": _cache_mtime(_parser_cache_path(key)),
            "eval_cache_time":   _cache_mtime(_eval_cache_path(key)),
            "total_truth":       res["total_truth"]   if res else None,
            "total_parsed":      res["total_parsed"]  if res else None,
            "precision":         res["precision"]     if res else None,
            "recall":            res["recall"]        if res else None,
            "f1":                res["f1"]            if res else None,
            "matched_count":     res["matched_count"] if res else None,
            "missed_count":      res["missed_count"]  if res else None,
            "extra_count":       res["extra_count"]   if res else None,
        })
    return docs


@router.post("/run")
async def start_eval(doc_key: str, force_reparse: bool = False):
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status": "running", "log": [], "doc_key": doc_key,
        "result": None, "error": None,
        "progress": {"phase": "loading", "total_pages": 0,
                     "chunks_done": 0, "total_chunks": 0, "streets_found": 0},
    }
    asyncio.get_running_loop().run_in_executor(
        _eval_executor, _run_eval, job_id, doc_key, force_reparse
    )
    return {"job_id": job_id}


@router.get("/job/{job_id}")
async def get_job(job_id: str):
    if job_id not in _jobs:
        return {"status": "not_found"}
    j = _jobs[job_id]
    return {
        "status":   j["status"],
        "log":      j["log"],
        "progress": j.get("progress", {}),
        "doc_key":  j["doc_key"],
        "result":   j.get("result"),
        "error":    j.get("error"),
    }


@router.get("/results/{doc_key:path}")
async def get_results(doc_key: str):
    if doc_key not in _results:
        return {"error": "No results — run eval first"}
    return _results[doc_key]


@router.delete("/cache-all")
async def clear_all_cache():
    """Delete all cached eval results, parser outputs, and raw DocAI files."""
    cleared = 0
    for fn in os.listdir(CACHE_DIR):
        if fn.startswith("_"):
            continue  # keep _history.json etc
        path = os.path.join(CACHE_DIR, fn)
        try:
            os.remove(path)
            cleared += 1
        except Exception:
            pass
    _results.clear()
    return {"cleared": cleared}


@router.delete("/cache/{doc_key:path}")
async def clear_cache(doc_key: str):
    cleared = []
    raw_path = os.path.join(CACHE_DIR, doc_key + "_docai_raw.json")
    for path in [_parser_cache_path(doc_key), _eval_cache_path(doc_key), raw_path]:
        if os.path.exists(path):
            os.remove(path)
            cleared.append(os.path.basename(path))
    if doc_key in _results:
        del _results[doc_key]
    return {"cleared": cleared}


def _docai_raw_path(doc_key):
    return os.path.join(CACHE_DIR, doc_key + "_docai_raw.json")


def _street_tokens(text: str) -> set:
    """Normalize and tokenize a street name for fuzzy search."""
    import re
    s = re.sub(r"[^\w\s]", " ", str(text or "").upper())
    return set(s.split()) - {"TO", "FROM", "THE", "AND", "OR", "AT", "OF"}


def _cell_matches(cell: str, tokens: set) -> float:
    """Return overlap ratio between cell tokens and query tokens."""
    if not tokens or not cell:
        return 0.0
    cell_tokens = _street_tokens(cell)
    if not cell_tokens:
        return 0.0
    overlap = len(tokens & cell_tokens)
    return overlap / len(tokens)


@router.get("/diagnose/{doc_key:path}")
async def diagnose(doc_key: str):
    """For each missed street, find the raw DocAI block that should contain it."""
    raw_path = _docai_raw_path(doc_key)
    eval_path = _eval_cache_path(doc_key)

    if not os.path.exists(eval_path):
        return {"error": "No eval results — run eval first"}
    if not os.path.exists(raw_path):
        return {"error": "No raw DocAI data — re-run eval with force_reparse to generate it"}

    with open(eval_path) as f:
        eval_data = json.load(f)
    with open(raw_path) as f:
        raw = json.load(f)  # { "104": [{"header_rows": [...], "body_rows": [...]}] }

    missed = eval_data.get("missed", [])
    extra = eval_data.get("extra", [])
    parsed = eval_data.get("matched", [])

    # Build a lookup: parsed main_street tokens → parsed record
    extra_index = []
    for s in extra:
        ms = s.get("main_street") or ""
        extra_index.append((_street_tokens(ms), s))

    diagnoses = []
    for missed_street in missed:
        ms = missed_street.get("main_street", "")
        fr = missed_street.get("from_street", "")
        to = missed_street.get("to_street", "")
        query_tokens = _street_tokens(ms)

        # Search every raw DocAI cell for a match
        found_in_raw = []
        for page_str, tables in raw.items():
            page_num = int(page_str)
            for tbl_idx, tbl in enumerate(tables):
                headers = tbl.get("header_rows", [])
                header_flat = [" | ".join(h for h in row if h) for row in headers]
                for row_idx, row in enumerate(tbl.get("body_rows", [])):
                    for col_idx, cell in enumerate(row):
                        score = _cell_matches(cell, query_tokens)
                        if score >= 0.6:
                            # Check if this row also contains from/to clues
                            row_text = " ".join(row)
                            from_score = _cell_matches(row_text, _street_tokens(fr)) if fr else 0
                            to_score   = _cell_matches(row_text, _street_tokens(to)) if to else 0
                            found_in_raw.append({
                                "page": page_num,
                                "table_idx": tbl_idx,
                                "row_idx": row_idx,
                                "col_idx": col_idx,
                                "headers": header_flat,
                                "row": row,
                                "main_score": round(score, 2),
                                "from_score": round(from_score, 2),
                                "to_score":   round(to_score, 2),
                            })

        # Sort by combined relevance
        found_in_raw.sort(key=lambda x: -(x["main_score"] + x["from_score"] + x["to_score"]))
        found_in_raw = found_in_raw[:5]  # top 5 hits

        # Determine miss reason
        if not found_in_raw:
            reason = "not_found_in_docai"
        else:
            best = found_in_raw[0]
            best_row = " ".join(best["row"])
            row_suffix_ct = sum(
                1 for w in best_row.upper().split()
                if w in {"ST", "AV", "AVE", "DR", "CT", "RD", "LN", "PL", "BL", "WY", "WAY", "CIR", "TER", "ML", "BLVD"}
            )
            # Check if it appeared as an extra (wrong field mapping)
            in_extra = any(
                len(et & query_tokens) / max(len(query_tokens), 1) >= 0.6
                for et, _ in extra_index
            )
            if in_extra:
                reason = "extracted_wrong_field"
            elif row_suffix_ct >= 4:
                reason = "row_merged"
            elif best["main_score"] >= 0.8 and not found_in_raw[0]["from_score"]:
                reason = "from_to_missing"
            else:
                reason = "col_map_or_parse_error"

        diagnoses.append({
            "missed": missed_street,
            "reason": reason,
            "raw_hits": found_in_raw,
        })

    return {
        "doc_key": doc_key,
        "missed_count": len(missed),
        "diagnoses": diagnoses,
    }


@router.get("/docai-raw/{doc_key:path}")
async def get_docai_raw(doc_key: str):
    """Return the raw DocAI table extraction output for a document."""
    raw_path = _docai_raw_path(doc_key)
    if not os.path.exists(raw_path):
        return {"error": "No raw DocAI data — run eval first"}
    with open(raw_path) as f:
        raw = json.load(f)
    # Summarize: for each page, show page num, row counts, and first header/body row
    pages = []
    for page_num, tables in sorted(raw.items(), key=lambda x: int(x[0])):
        table_list = tables if isinstance(tables, list) else [tables]
        for ti, t in enumerate(table_list):
            pages.append({
                "page": int(page_num),
                "table": ti,
                "header_rows": t.get("header_rows", []),
                "body_rows": t.get("body_rows", []),
            })
    return {"doc_key": doc_key, "page_count": len(pages), "pages": pages}


@router.get("/truth/{doc_key:path}")
async def get_truth(doc_key: str):
    try:
        with open(COL_MAP) as f:
            col_map = json.load(f)
        rows = load_ground_truth(doc_key, col_map)
        return {"doc_key": doc_key, "count": len(rows), "streets": rows}
    except Exception as e:
        return {"error": str(e)}


@router.get("/pdf/{doc_key:path}")
async def serve_pdf(doc_key: str):
    path = os.path.join(PDF_DIR, doc_key + ".pdf")
    if not os.path.exists(path):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "PDF not found"}, status_code=404)
    return FileResponse(path, media_type="application/pdf")


@router.get("/history")
async def get_history():
    return _load_history()


def _load_ledger():
    if os.path.exists(LEDGER_FILE):
        try:
            with open(LEDGER_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []


@router.get("/ledger")
async def get_ledger():
    return _load_ledger()


@router.post("/ledger/snapshot")
async def save_ledger_snapshot(notes: str = ""):
    """Save a snapshot of all current results to the ledger with optional notes."""
    docs_run = {k: v for k, v in _results.items()}
    if not docs_run:
        return {"error": "No results to snapshot — run some evals first"}

    all_keys = _doc_keys()
    doc_entries = {}
    for key in all_keys:
        r = docs_run.get(key)
        if r:
            doc_entries[key] = {
                "f1":        round(r["f1"], 4),
                "precision": round(r["precision"], 4),
                "recall":    round(r["recall"], 4),
                "matched":   r["matched_count"],
                "missed":    r["missed_count"],
                "extra":     r["extra_count"],
                "total_truth": r["total_truth"],
                "timestamp": r["timestamp"],
            }

    run_docs = list(doc_entries.values())
    tm = sum(d["matched"] for d in run_docs)
    te = sum(d["extra"]   for d in run_docs)
    tt = sum(d["total_truth"] for d in run_docs)
    p  = tm / (tm + te) if (tm + te) else 0
    r  = tm / tt if tt else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0

    sub50 = [k for k, v in doc_entries.items() if v["f1"] < 0.5]

    snapshot = {
        "timestamp":   datetime.now().isoformat(),
        "notes":       notes,
        "pipeline":    os.environ.get("MAIN_MODULE", "main_opus"),
        "aggregate": {
            "docs_run":    len(run_docs),
            "docs_total":  len(all_keys),
            "avg_f1":      round(f1, 4),
            "precision":   round(p, 4),
            "recall":      round(r, 4),
            "sub_50_count": len(sub50),
            "sub_50_docs": sub50,
        },
        "docs": doc_entries,
    }

    ledger = _load_ledger()
    ledger.append(snapshot)
    with open(LEDGER_FILE, "w") as f:
        json.dump(ledger, f, indent=2)

    return {"saved": True, "snapshot_index": len(ledger) - 1, "aggregate": snapshot["aggregate"]}


# ── Map / polyline endpoint ───────────────────────────────────────────────────

MAP_CACHE_DIR = os.path.join(CACHE_DIR, "map_cache")

# Limit tokens that cannot be geocoded as cross streets
_MAP_LIMIT_TOKENS = {
    "END", "BEGIN", "BEGINNING", "START", "STOP",
    "EOP", "EOR", "EOS", "EOC", "EOL", "EOF",
    "CDS", "DEAD END", "DEADEND",
    "N CDS", "S CDS", "E CDS", "W CDS",
    "NE CDS", "NW CDS", "SE CDS", "SW CDS",
}

_MAP_OFFSET_RE = re.compile(
    r"^\d+['\"FT\s]*(FEET|FT|'|NORTH|SOUTH|EAST|WEST|N|S|E|W|OF|E/O|W/O|N/O|S/O)",
    re.IGNORECASE,
)


def _is_map_limit(s: str) -> bool:
    if not s:
        return True
    u = s.strip().upper()
    if u in _MAP_LIMIT_TOKENS:
        return True
    if _MAP_OFFSET_RE.match(u):
        return True
    return False


def _map_geocode(address: str, api_key: str) -> tuple:
    """Geocode an address string → (lat, lng) or raise ValueError."""
    import urllib.parse, urllib.request
    os.makedirs(MAP_CACHE_DIR, exist_ok=True)
    cache_key = address.upper().strip()
    cache_file = os.path.join(MAP_CACHE_DIR, f"{abs(hash(cache_key)) % 10**12}.json")

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cached = json.load(f)
        if cached.get("ok"):
            return cached["lat"], cached["lng"]
        raise ValueError(cached.get("error", "cached failure"))

    query = urllib.parse.quote(address)
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={query}&key={api_key}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())

    if data.get("status") != "OK" or not data.get("results"):
        err = f"Geocoding failed: {data.get('status')} for {address}"
        with open(cache_file, "w") as f:
            json.dump({"ok": False, "error": err}, f)
        raise ValueError(err)

    loc = data["results"][0]["geometry"]["location"]
    with open(cache_file, "w") as f:
        json.dump({"ok": True, "lat": loc["lat"], "lng": loc["lng"]}, f)
    return loc["lat"], loc["lng"]


def _map_route(start: tuple, end: tuple, api_key: str) -> tuple:
    """Get encoded polyline between two (lat,lng) points via Routes API v2.
    Returns (encoded_polyline_str, length_m).
    Matches the exact same call as Parse/takeoff/core.py _get_directions.
    """
    import urllib.request
    cache_key = f"{start[0]:.6f},{start[1]:.6f}|{end[0]:.6f},{end[1]:.6f}"
    cache_file = os.path.join(MAP_CACHE_DIR, f"r_{abs(hash(cache_key)) % 10**12}.json")

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cached = json.load(f)
        if cached.get("ok"):
            return cached["encoded"], cached["length_m"]
        raise ValueError(cached.get("error", "cached failure"))

    import json as _json
    payload = _json.dumps({
        "origin": {"location": {"latLng": {"latitude": start[0], "longitude": start[1]}}},
        "destination": {"location": {"latLng": {"latitude": end[0], "longitude": end[1]}}},
        "travelMode": "DRIVE",
        "polylineEncoding": "ENCODED_POLYLINE",
    }).encode()

    req = urllib.request.Request(
        "https://routes.googleapis.com/directions/v2:computeRoutes",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "routes.polyline.encodedPolyline,routes.distanceMeters",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    routes = data.get("routes") or []
    if not routes:
        err = "Routes API returned no routes"
        with open(cache_file, "w") as f:
            json.dump({"ok": False, "error": err}, f)
        raise ValueError(err)

    encoded = routes[0].get("polyline", {}).get("encodedPolyline", "")
    length_m = routes[0].get("distanceMeters", 0)
    if not encoded:
        err = "Routes API response missing encodedPolyline"
        with open(cache_file, "w") as f:
            json.dump({"ok": False, "error": err}, f)
        raise ValueError(err)

    with open(cache_file, "w") as f:
        json.dump({"ok": True, "encoded": encoded, "length_m": length_m}, f)
    return encoded, length_m


def _classify_street(encoded: str, length_m: float) -> str:
    if length_m < 50:
        return "yellow"   # suspiciously short
    if length_m > 8000:
        return "yellow"   # suspiciously long (>5 miles)
    return "green"


@router.post("/scan-cities")
async def scan_cities():
    """
    For every PDF in tests/pdfs/, send the first 5 pages through claude-haiku
    to extract city + state. Saves results to tests/city_state.json and
    backfills parser caches (adds _meta.city/state if missing).
    Returns a per-doc summary.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set"}

    import anthropic as _ant

    _HEADER_PROMPT = (
        "You are parsing a road construction bid document. Extract project-level fields only.\n"
        "Return ONLY valid JSON with these fields:\n"
        "- bid_number, project_name, city, state, work_type, estimated_cost, bid_due_date\n"
        "Use null for any field not found. city and state are the city/state where the work will be performed."
    )

    cfg_path = os.path.join(TESTS_DIR, "city_state.json")
    existing: dict = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            existing = json.load(f)

    results = {}
    pdfs = sorted(f for f in os.listdir(PDF_DIR) if f.endswith(".pdf"))

    def _scan_one(pdf_name: str) -> dict:
        doc_key = pdf_name[:-4]
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                blocks = []
                for i in range(min(5, len(pdf.pages))):
                    text = pdf.pages[i].extract_text() or ""
                    blocks.append({"type": "text", "text": f"\n--- Page {i+1} ---\n{text}"})

            client = _ant.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                system=_HEADER_PROMPT,
                messages=[{"role": "user", "content": blocks}],
            )
            raw = resp.content[0].text.strip()
            # strip markdown fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            schema = json.loads(raw)
            city  = schema.get("city") or ""
            state = schema.get("state") or ""
            return {"doc_key": doc_key, "city": city, "state": state,
                    "city_state": f"{city}, {state}" if (city and state) else city or state or "",
                    "status": "ok"}
        except Exception as e:
            return {"doc_key": doc_key, "city": "", "state": "", "city_state": "", "status": f"error: {e}"}

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [loop.run_in_executor(pool, _scan_one, p) for p in pdfs]
        scan_results = await asyncio.gather(*futures)

    # Merge into city_state.json and backfill parser caches
    for r in scan_results:
        doc_key = r["doc_key"]
        results[doc_key] = r
        if r["city_state"]:
            existing[doc_key] = r["city_state"]
            # Backfill parser cache with _meta
            parser_cache = _parser_cache_path(doc_key)
            if os.path.exists(parser_cache):
                with open(parser_cache) as f:
                    cached = json.load(f)
                if isinstance(cached, list):
                    # Upgrade plain list → dict with _meta
                    cached = {"streets": cached, "_meta": {"city": r["city"], "state": r["state"]}}
                    with open(parser_cache, "w") as f:
                        json.dump(cached, f, indent=2)
                elif isinstance(cached, dict) and not cached.get("_meta", {}).get("city"):
                    cached.setdefault("_meta", {})["city"] = r["city"]
                    cached["_meta"]["state"] = r["state"]
                    with open(parser_cache, "w") as f:
                        json.dump(cached, f, indent=2)

    with open(cfg_path, "w") as f:
        json.dump(existing, f, indent=2)

    return {"scanned": len(results), "results": results}


@router.get("/city-state-all")
async def get_all_city_states():
    """Return the full city_state.json map for UI display."""
    cfg = os.path.join(os.path.dirname(__file__), "tests", "city_state.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            return json.load(f)
    return {}


@router.get("/city-state/{doc_key:path}")
async def get_city_state(doc_key: str):
    """Return the known city/state for a document (for UI pre-population)."""
    # Check parser cache first
    parser_cache = _parser_cache_path(doc_key)
    if os.path.exists(parser_cache):
        with open(parser_cache) as f:
            cached = json.load(f)
        if isinstance(cached, dict):
            meta = cached.get("_meta", {})
            city, state = meta.get("city", ""), meta.get("state", "")
            if city:
                return {"city_state": f"{city}, {state}" if state else city}
    # Fallback to city_state.json
    cfg = os.path.join(os.path.dirname(__file__), "tests", "city_state.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            mapping = json.load(f)
        if doc_key in mapping:
            return {"city_state": mapping[doc_key]}
    return {"city_state": ""}


@router.get("/map/{doc_key:path}")
async def get_map_data(doc_key: str, city_state: str = ""):
    """
    For each street in the cached parser output, geocode both endpoints and
    fetch the driving route polyline via Routes API v2.
    Returns a list of per-street results for the map view.
    Fully cached — safe to call repeatedly.
    city_state: optional override (e.g. "Irvine, California")
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_MAPS_API_KEY not set on server"}

    parser_cache = _parser_cache_path(doc_key)
    if not os.path.exists(parser_cache):
        return {"error": "No parser output — run eval first"}

    with open(parser_cache) as f:
        cached = json.load(f)

    streets = cached if isinstance(cached, list) else cached.get("streets", [])

    # Get city/state: UI override → parser meta → city_state.json fallback
    meta = cached.get("_meta", {}) if isinstance(cached, dict) else {}
    city  = meta.get("city", "")
    state = meta.get("state", "")
    _city_state_cfg = os.path.join(os.path.dirname(__file__), "tests", "city_state.json")
    _city_state_map: dict = {}
    if os.path.exists(_city_state_cfg):
        with open(_city_state_cfg) as _f:
            _city_state_map = json.load(_f)
    # Priority: UI param > parser _meta > city_state.json
    if city_state.strip():
        _job_city_state = city_state.strip()
    elif city:
        _job_city_state = f"{city}, {state}" if state else city
        city = ""  # prevent double use below
        state = ""
    else:
        _job_city_state = _city_state_map.get(doc_key, "")

    results = []
    for s in streets:
        main = (s.get("main_street") or "").strip()
        fr   = (s.get("from_street") or "").strip()
        to   = (s.get("to_street")   or "").strip()
        # Prefer street-level location → parser meta city/state → city_state.json
        loc  = (s.get("location") or "").strip()
        if not loc and city:
            loc = f"{city}, {state}" if state else city
        if not loc and _job_city_state:
            loc = _job_city_state
        page = s.get("page", "")

        # Classify edge cases without hitting the API
        fr_limit = _is_map_limit(fr)
        to_limit = _is_map_limit(to)

        if not main:
            continue

        if fr_limit or to_limit:
            results.append({
                "main_street": main, "from_street": fr, "to_street": to,
                "page": page,
                "status": "terminus",
                "color": "gray",
                "reason": f"{'from' if fr_limit else 'to'}_is_terminus",
                "from_addr": None, "to_addr": None,
                "encoded_polyline": None, "length_m": None,
            })
            continue

        if not loc:
            results.append({
                "main_street": main, "from_street": fr, "to_street": to,
                "page": page,
                "status": "error",
                "color": "red",
                "reason": "no_city_state",
                "from_addr": None, "to_addr": None,
                "encoded_polyline": None, "length_m": None,
            })
            continue

        from_addr = f"{main} & {fr}, {loc}"
        to_addr   = f"{main} & {to}, {loc}"

        try:
            start_coords = await asyncio.get_running_loop().run_in_executor(
                None, _map_geocode, from_addr, api_key
            )
        except Exception as e:
            results.append({
                "main_street": main, "from_street": fr, "to_street": to,
                "page": page,
                "status": "error", "color": "red",
                "reason": "geocode_failed_from",
                "from_addr": from_addr, "to_addr": to_addr,
                "error": str(e)[:120],
                "encoded_polyline": None, "length_m": None,
            })
            continue

        try:
            end_coords = await asyncio.get_running_loop().run_in_executor(
                None, _map_geocode, to_addr, api_key
            )
        except Exception as e:
            results.append({
                "main_street": main, "from_street": fr, "to_street": to,
                "page": page,
                "status": "error", "color": "red",
                "reason": "geocode_failed_to",
                "from_addr": from_addr, "to_addr": to_addr,
                "error": str(e)[:120],
                "encoded_polyline": None, "length_m": None,
            })
            continue

        try:
            encoded, length_m = await asyncio.get_running_loop().run_in_executor(
                None, _map_route, start_coords, end_coords, api_key
            )
            color = _classify_street(encoded, length_m)
            results.append({
                "main_street": main, "from_street": fr, "to_street": to,
                "page": page,
                "status": "ok", "color": color,
                "reason": "suspicious_length" if color == "yellow" else None,
                "from_addr": from_addr, "to_addr": to_addr,
                "start_coords": list(start_coords),
                "end_coords": list(end_coords),
                "encoded_polyline": encoded,
                "length_m": length_m,
            })
        except Exception as e:
            results.append({
                "main_street": main, "from_street": fr, "to_street": to,
                "page": page,
                "status": "error", "color": "red",
                "reason": "route_failed",
                "from_addr": from_addr, "to_addr": to_addr,
                "start_coords": list(start_coords),
                "end_coords": list(end_coords),
                "error": str(e)[:120],
                "encoded_polyline": None, "length_m": None,
            })

    ok_count      = sum(1 for r in results if r["status"] == "ok")
    error_count   = sum(1 for r in results if r["status"] == "error")
    yellow_count  = sum(1 for r in results if r.get("color") == "yellow")
    terminus_count= sum(1 for r in results if r["status"] == "terminus")

    return {
        "doc_key": doc_key,
        "city_state": loc if loc else None,
        "total": len(results),
        "ok": ok_count,
        "errors": error_count,
        "suspicious": yellow_count,
        "terminus": terminus_count,
        "streets": results,
    }
