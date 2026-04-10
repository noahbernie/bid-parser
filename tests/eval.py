"""
Bid Parser Evaluation Suite
Usage:
  python tests/eval.py                        # run all documents
  python tests/eval.py --doc "K-26-2431"      # run one document
  python tests/eval.py --no-cache             # force re-parse everything
"""
import argparse
import csv
import json
import os
import re
import sys
import requests

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL   = os.environ.get("PARSER_URL", "http://localhost:8000")
API_KEY    = os.environ.get("PARSER_API_KEY", "")
TESTS_DIR  = os.path.dirname(os.path.abspath(__file__))
GT_DIR     = os.path.join(TESTS_DIR, "ground_truth")
PDF_DIR    = os.path.join(TESTS_DIR, "pdfs")
CACHE_DIR  = os.path.join(TESTS_DIR, "cache")
COL_MAP    = os.path.join(TESTS_DIR, "column_map.json")

FUZZY_THRESHOLD = 0.80   # 80% similarity to count as a match

# ── Normalization ─────────────────────────────────────────────────────────────

_SUFFIX_MAP = {
    "STREET": "ST", "AVENUE": "AV", "DRIVE": "DR", "BOULEVARD": "BL",
    "ROAD": "RD",   "COURT": "CT",  "LANE": "LN",  "PLACE": "PL",
    "WAY": "WY",    "CIRCLE": "CIR","TERRACE": "TER","TRAIL": "TRL",
    "AVE": "AV",    "BLVD": "BL",
}

def norm(v: str) -> str:
    if not v:
        return ""
    parts = re.sub(r"[^\w\s]", "", v.strip().upper()).split()
    if parts and parts[-1] in _SUFFIX_MAP:
        parts[-1] = _SUFFIX_MAP[parts[-1]]
    return " ".join(parts)

def similarity(a: str, b: str) -> float:
    """Simple token overlap similarity."""
    if not a or not b:
        return 0.0
    ta, tb = set(norm(a).split()), set(norm(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))

def street_key(s: dict) -> tuple:
    return (norm(s.get("main_street", "")),
            norm(s.get("from_street", "")),
            norm(s.get("to_street", "")))

# ── Ground truth loader ───────────────────────────────────────────────────────

def load_ground_truth(doc_key: str, col_map: dict) -> list[dict]:
    csv_path = os.path.join(GT_DIR, doc_key + ".csv")
    if not os.path.exists(csv_path):
        print(f"  ⚠  No ground truth CSV found: {csv_path}")
        return []

    mapping = col_map.get(doc_key, {})
    if not mapping or any(v == "FILL IN" for v in mapping.values()):
        print(f"  ⚠  column_map.json not filled in for '{doc_key}' — skipping")
        return []

    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "main_street": row.get(mapping.get("main_street", ""), ""),
                "from_street":  row.get(mapping.get("from_street", ""), ""),
                "to_street":    row.get(mapping.get("to_street", ""), ""),
                "work_type":    row.get(mapping.get("work_type", ""), ""),
            })
    return [r for r in rows if r["main_street"].strip()]

# ── Parser runner ─────────────────────────────────────────────────────────────

def run_parser(doc_key: str, use_cache: bool = True) -> list[dict]:
    cache_path = os.path.join(CACHE_DIR, doc_key + ".json")

    if use_cache and os.path.exists(cache_path):
        print(f"  📦 Using cached result")
        with open(cache_path) as f:
            return json.load(f)

    # Find PDF
    pdf_path = None
    for fname in os.listdir(PDF_DIR):
        if doc_key.lower() in fname.lower() and fname.endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, fname)
            break

    if not pdf_path:
        print(f"  ⚠  No PDF found in tests/pdfs/ matching '{doc_key}'")
        return []

    print(f"  🔄 Parsing {os.path.basename(pdf_path)} ...")
    headers = {"X-Api-Key": API_KEY} if API_KEY else {}
    with open(pdf_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/parse",
            headers=headers,
            files={"file": (os.path.basename(pdf_path), f, "application/pdf")},
            timeout=600,
        )
    resp.raise_for_status()
    streets = resp.json().get("streets", [])

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(streets, f, indent=2)

    return streets

# ── Matching ──────────────────────────────────────────────────────────────────

def match_streets(truth: list[dict], parsed: list[dict]) -> dict:
    matched, missed, extra = [], [], []
    used = set()

    for t in truth:
        tk = street_key(t)
        best_score, best_idx = 0.0, -1
        for i, p in enumerate(parsed):
            if i in used:
                continue
            score = similarity(tk[0], street_key(p)[0])
            # Also require from/to to be in the ballpark
            from_ok = similarity(tk[1], street_key(p)[1]) > 0.5 or not tk[1] or not street_key(p)[1]
            to_ok   = similarity(tk[2], street_key(p)[2]) > 0.5 or not tk[2] or not street_key(p)[2]
            if score > best_score and from_ok and to_ok:
                best_score, best_idx = score, i
        if best_score >= FUZZY_THRESHOLD and best_idx >= 0:
            matched.append({"truth": t, "parsed": parsed[best_idx], "score": best_score})
            used.add(best_idx)
        else:
            missed.append(t)

    for i, p in enumerate(parsed):
        if i not in used:
            extra.append(p)

    total = len(truth)
    precision = len(matched) / (len(matched) + len(extra)) if (matched or extra) else 0
    recall    = len(matched) / total if total else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "matched": matched, "missed": missed, "extra": extra,
        "precision": precision, "recall": recall, "f1": f1,
        "total_truth": total, "total_parsed": len(parsed),
    }

# ── Report ────────────────────────────────────────────────────────────────────

def print_report(doc_key: str, r: dict):
    p, re_, f1 = r["precision"], r["recall"], r["f1"]
    print(f"\n{'='*60}")
    print(f"  {doc_key}")
    print(f"{'='*60}")
    print(f"  Ground truth:  {r['total_truth']} streets")
    print(f"  Parser output: {r['total_parsed']} streets")
    print(f"  Matched:  {len(r['matched'])}   Missed: {len(r['missed'])}   Extra: {len(r['extra'])}")
    print(f"  Precision: {p:.0%}   Recall: {re_:.0%}   F1: {f1:.0%}")

    if r["missed"]:
        print(f"\n  ❌ MISSED ({len(r['missed'])}):")
        for s in r["missed"][:20]:
            print(f"     {s['main_street']}  |  {s['from_street']} → {s['to_street']}")
        if len(r["missed"]) > 20:
            print(f"     ... and {len(r['missed']) - 20} more")

    if r["extra"]:
        print(f"\n  ➕ EXTRA / false positives ({len(r['extra'])}):")
        for s in r["extra"][:10]:
            print(f"     {s.get('main_street')}  |  {s.get('from_street')} → {s.get('to_street')}")
        if len(r["extra"]) > 10:
            print(f"     ... and {len(r['extra']) - 10} more")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", help="Run only this document key")
    parser.add_argument("--no-cache", action="store_true", help="Force re-parse")
    args = parser.parse_args()

    with open(COL_MAP) as f:
        col_map = json.load(f)

    doc_keys = [k for k in col_map if not k.startswith("_")]
    if args.doc:
        doc_keys = [k for k in doc_keys if args.doc.lower() in k.lower()]
        if not doc_keys:
            print(f"No doc matching '{args.doc}' found in column_map.json")
            sys.exit(1)

    all_results = {}
    for doc_key in doc_keys:
        print(f"\n▶ {doc_key}")
        truth  = load_ground_truth(doc_key, col_map)
        if not truth:
            continue
        parsed = run_parser(doc_key, use_cache=not args.no_cache)
        result = match_streets(truth, parsed)
        all_results[doc_key] = result
        print_report(doc_key, result)

    if len(all_results) > 1:
        total_matched = sum(len(r["matched"]) for r in all_results.values())
        total_truth   = sum(r["total_truth"]  for r in all_results.values())
        total_parsed  = sum(r["total_parsed"] for r in all_results.values())
        overall_p = total_matched / (total_matched + sum(len(r["extra"]) for r in all_results.values())) if total_parsed else 0
        overall_r = total_matched / total_truth if total_truth else 0
        overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) else 0
        print(f"\n{'='*60}")
        print(f"  OVERALL ({len(all_results)} documents)")
        print(f"{'='*60}")
        print(f"  Precision: {overall_p:.0%}   Recall: {overall_r:.0%}   F1: {overall_f1:.0%}")
        print(f"  {total_matched}/{total_truth} ground truth streets matched")

if __name__ == "__main__":
    main()
