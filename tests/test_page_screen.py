#!/usr/bin/env python3
"""
Self-test: Gemini Flash page screen precision/recall.

Runs the page screen against every PDF in tests/pdfs/ that has a ground-truth
entry in tests/column_map.json, then reports per-file and aggregate precision/recall.

Usage:
    cd /Users/noahbernstein/bid-parser-form
    python tests/test_page_screen.py [--pdf <stem>] [--no-cache] [--workers N]

The ground-truth expected pages are derived from tab names in column_map.json.
Tab names like "Page 192-193" or "Page 63-74" encode the PDF page range.
Tabs with {"skip": true} are excluded from ground truth.
"""
import os
import sys
import json
import re
import base64
import urllib.request
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow importing from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

TESTS_DIR = Path(__file__).resolve().parent
PDF_DIR   = TESTS_DIR / "pdfs"
COL_MAP   = TESTS_DIR / "column_map.json"
CACHE_DIR = TESTS_DIR / "screen_cache"
CACHE_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Prompt — keep in sync with main_opus.py _GEMINI_PAGE_SCREEN_PROMPT          #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Ground-truth parser                                                          #
# --------------------------------------------------------------------------- #
def _parse_page_range(tab_name: str) -> list:
    """Extract page numbers from a tab name like 'Page 192-193' or 'Page 63-74'."""
    m = re.search(r'(\d+)[-–](\d+)', tab_name)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        # Sanity cap: if the range is huge (e.g. some weird tab name), just use start
        if end - start > 50:
            return [start]
        return list(range(start, end + 1))
    m = re.search(r'(\d+)', tab_name)
    if m:
        return [int(m.group(1))]
    return []


def load_ground_truth(column_map: dict) -> dict:
    """Returns {pdf_stem: {expected_pages}} for each PDF in column_map."""
    gt = {}
    for pdf_stem, file_entry in column_map.items():
        if "_instructions" in pdf_stem:
            continue
        # Explicit screen_pages field takes priority over tab-name parsing
        if "screen_pages" in file_entry:
            pages = set(file_entry["screen_pages"])
        else:
            pages = set()
            tabs = file_entry.get("tabs", {})
            for tab_name, tab_cfg in tabs.items():
                if isinstance(tab_cfg, list):
                    cfgs = tab_cfg
                else:
                    cfgs = [tab_cfg]
                for cfg in cfgs:
                    if cfg.get("skip"):
                        continue
                    pages.update(_parse_page_range(tab_name))
        if pages:
            gt[pdf_stem] = pages
    return gt


# --------------------------------------------------------------------------- #
# Page renderer                                                                #
# --------------------------------------------------------------------------- #
def render_page_b64(pdf_path: Path, page_idx: int, dpi: int = 150) -> str:
    doc = fitz.open(str(pdf_path))
    page = doc[page_idx]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    return base64.b64encode(pix.tobytes("png")).decode()


# --------------------------------------------------------------------------- #
# Gemini Flash page screen                                                     #
# --------------------------------------------------------------------------- #
def screen_page(pdf_path: Path, page_idx: int, gemini_url: str, use_cache: bool = True) -> tuple:
    """Returns (page_num_1indexed, is_street_schedule)."""
    page_num = page_idx + 1
    cache_key = f"{pdf_path.stem}__p{page_num}.json"
    cache_file = CACHE_DIR / cache_key

    if use_cache and cache_file.exists():
        with open(cache_file) as f:
            return page_num, json.load(f)["result"]

    b64 = render_page_b64(pdf_path, page_idx)
    payload = json.dumps({
        "contents": [{"parts": [
            {"text": _GEMINI_PAGE_SCREEN_PROMPT},
            {"inline_data": {"mime_type": "image/png", "data": b64}},
        ]}],
        "generationConfig": {"maxOutputTokens": 512, "temperature": 0},
    }).encode()
    req = urllib.request.Request(
        gemini_url, data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        parts = data["candidates"][0]["content"].get("parts", [])
        text_parts = [p["text"].strip() for p in parts if isinstance(p, dict) and p.get("text") and not p.get("thought")]
        if not text_parts:
            result = False
        else:
            answer = text_parts[-1]
            al = answer.lower().replace(" ", "").replace("\n", "")
            if '"is_street_schedule":true' in al:
                result = True
            elif '"is_street_schedule":false' in al:
                result = False
            elif "true" in al:
                result = True
            else:
                result = False
    except Exception as e:
        print(f"    ⚠ p.{page_num} error: {str(e)[:80]}", flush=True)
        result = False

    if use_cache:
        with open(cache_file, "w") as f:
            json.dump({"result": result}, f)

    return page_num, result


def screen_pdf(pdf_path: Path, gemini_url: str, workers: int, use_cache: bool) -> dict:
    doc = fitz.open(str(pdf_path))
    total = len(doc)
    doc.close()
    print(f"  Screening {total} pages...", flush=True)
    results: dict[int, bool] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(screen_page, pdf_path, i, gemini_url, use_cache): i for i in range(total)}
        for fut in as_completed(futs):
            pn, val = fut.result()
            results[pn] = val
    return results


def apply_expansion(results: dict, total_pages: int) -> set:
    """Mirror the expansion logic from run_extraction."""
    raw_selected = {p for p, v in results.items() if v}
    expanded = set(raw_selected)
    for p in sorted(raw_selected):
        for offset in range(1, 5):
            candidate = p + offset
            if candidate > total_pages:
                break
            if candidate in raw_selected:
                break
            if results.get(candidate) is False:
                break
            expanded.add(candidate)
    return expanded


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def compute_metrics(predicted: set, expected: set) -> dict:
    tp = predicted & expected
    fp = predicted - expected
    fn = expected - predicted
    precision = len(tp) / len(predicted) if predicted else 0.0
    recall    = len(tp) / len(expected)  if expected  else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": sorted(tp), "fp": sorted(fp), "fn": sorted(fn),
            "precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf",      help="Only test this PDF stem (partial match ok)")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached screen results")
    parser.add_argument("--workers",  type=int, default=20, help="Parallel Gemini calls per PDF")
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"

    with open(COL_MAP) as f:
        column_map = json.load(f)

    gt = load_ground_truth(column_map)
    use_cache = not args.no_cache

    # Filter to requested PDF stem
    if args.pdf:
        gt = {k: v for k, v in gt.items() if args.pdf.lower() in k.lower()}
        if not gt:
            print(f"No entries match --pdf '{args.pdf}'")
            sys.exit(1)

    all_results = []

    for stem, expected_pages in sorted(gt.items()):
        pdf_path = PDF_DIR / f"{stem}.pdf"
        if not pdf_path.exists():
            print(f"⚠  SKIP {stem} — PDF not found")
            continue

        print(f"\n{'='*70}")
        print(f"📄 {stem}")
        print(f"   Expected pages: {sorted(expected_pages)}")

        raw_results = screen_pdf(pdf_path, gemini_url, args.workers, use_cache)

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        doc.close()

        predicted_pages = apply_expansion(raw_results, total_pages)
        m = compute_metrics(predicted_pages, expected_pages)

        status = "✅" if m["recall"] >= 0.95 and m["precision"] >= 0.70 else ("⚠" if m["recall"] >= 0.80 else "❌")
        print(f"   Predicted pages: {sorted(predicted_pages)}")
        print(f"   {status} Precision={m['precision']:.2f}  Recall={m['recall']:.2f}  F1={m['f1']:.2f}")
        if m["fp"]:
            print(f"   False positives (should be NO): {m['fp']}")
        if m["fn"]:
            print(f"   False negatives (should be YES): {m['fn']}")

        all_results.append({"stem": stem, **m,
                            "predicted": sorted(predicted_pages),
                            "expected": sorted(expected_pages)})

    if not all_results:
        print("No PDFs tested.")
        return

    # Aggregate over all pages across all files
    total_tp = sum(len(r["tp"]) for r in all_results)
    total_fp = sum(len(r["fp"]) for r in all_results)
    total_fn = sum(len(r["fn"]) for r in all_results)
    agg_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    agg_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    agg_f1 = 2 * agg_p * agg_r / (agg_p + agg_r) if (agg_p + agg_r) > 0 else 0.0

    print(f"\n{'='*70}")
    print(f"AGGREGATE  ({len(all_results)} PDFs)")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Precision={agg_p:.3f}  Recall={agg_r:.3f}  F1={agg_f1:.3f}")

    # Per-file summary table
    print(f"\n{'PDF':<55} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 75)
    for r in sorted(all_results, key=lambda x: x["f1"]):
        flag = "❌" if r["recall"] < 0.80 else ("⚠" if r["recall"] < 0.95 or r["precision"] < 0.70 else "✅")
        name = r["stem"][-54:]
        print(f"{flag} {name:<54} {r['precision']:>5.2f} {r['recall']:>5.2f} {r['f1']:>5.2f}")


if __name__ == "__main__":
    main()
