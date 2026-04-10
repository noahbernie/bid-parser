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
from datetime import datetime

import pdfplumber
from fastapi import APIRouter
from fastapi.responses import FileResponse

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR   = os.path.join(BASE_DIR, "static")
TESTS_DIR    = os.path.join(BASE_DIR, "tests")
GT_DIR       = os.path.join(TESTS_DIR, "ground_truth")
PDF_DIR      = os.path.join(TESTS_DIR, "pdfs")
CACHE_DIR    = os.path.join(TESTS_DIR, "cache")
COL_MAP      = os.path.join(TESTS_DIR, "column_map.json")
HISTORY_FILE = os.path.join(CACHE_DIR, "_history.json")

sys.path.insert(0, TESTS_DIR)
from eval import load_ground_truth, match_streets  # noqa: E402

router = APIRouter(prefix="/eval")

# ── In-memory state ───────────────────────────────────────────────────────────

_jobs    = {}   # job_id  -> job dict
_results = {}   # doc_key -> serialized result dict

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

            from main import run_extraction, documents as docs_store
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
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_eval, job_id, doc_key, force_reparse)
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


@router.delete("/cache/{doc_key:path}")
async def clear_cache(doc_key: str):
    cleared = []
    for path in [_parser_cache_path(doc_key), _eval_cache_path(doc_key)]:
        if os.path.exists(path):
            os.remove(path)
            cleared.append(os.path.basename(path))
    if doc_key in _results:
        del _results[doc_key]
    return {"cleared": cleared}


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
