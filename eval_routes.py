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
                parsed = json.load(f)
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

            with open(parser_cache, "w") as f:
                json.dump(parsed, f, indent=2)
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
    return FileResponse(os.path.join(STATIC_DIR, "eval.html"))


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
