"""
Bid Parser Evaluation Suite
Usage:
  python tests/eval.py                        # run all documents
  python tests/eval.py --doc "K-26-2431"      # run one document (substring match)
  python tests/eval.py --no-cache             # force re-parse everything
  python tests/eval.py --show-matched         # also print matched rows (verbose)
"""
import argparse
import json
import os
import re
import sys
import requests
import openpyxl

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL   = os.environ.get("PARSER_URL", "http://localhost:8000")
API_KEY    = os.environ.get("PARSER_API_KEY", "")
TESTS_DIR  = os.path.dirname(os.path.abspath(__file__))
GT_DIR     = os.path.join(TESTS_DIR, "ground_truth")
PDF_DIR    = os.path.join(TESTS_DIR, "pdfs")
CACHE_DIR  = os.path.join(TESTS_DIR, "cache")
COL_MAP    = os.path.join(TESTS_DIR, "column_map.json")

FUZZY_THRESHOLD = 0.80

# ── Normalization ─────────────────────────────────────────────────────────────

_SUFFIX_MAP = {
    "STREET": "ST", "AVENUE": "AV", "DRIVE": "DR", "BOULEVARD": "BL",
    "ROAD": "RD",   "COURT": "CT",  "LANE": "LN",  "PLACE": "PL",
    "WAY": "WY",    "CIRCLE": "CIR","TERRACE": "TER","TRAIL": "TRL",
    "AVE": "AV",    "BLVD": "BL",   "PKWY": "PKWY",
}

def norm(v) -> str:
    if not v:
        return ""
    s = re.sub(r"\s+", " ", str(v).strip().upper())
    s = re.sub(r"[^\w\s]", "", s)
    parts = s.split()
    if parts and parts[-1] in _SUFFIX_MAP:
        parts[-1] = _SUFFIX_MAP[parts[-1]]
    return " ".join(parts)

def similarity(a: str, b: str) -> float:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    ta, tb = set(na.split()), set(nb.split())
    return len(ta & tb) / max(len(ta), len(tb))

def street_key(s: dict) -> tuple:
    return (norm(s.get("main_street", "")),
            norm(s.get("from_street", "")),
            norm(s.get("to_street", "")))

# ── Ground truth loader ───────────────────────────────────────────────────────

def _norm_header(h) -> str:
    """Normalize header for matching: collapse whitespace, uppercase."""
    return re.sub(r"\s+", " ", str(h or "").strip())

def load_ground_truth(doc_key: str, col_map: dict) -> list:
    xlsx_path = os.path.join(GT_DIR, doc_key + ".xlsx")
    if not os.path.exists(xlsx_path):
        print(f"  ⚠  No ground truth xlsx: {xlsx_path}")
        return []

    cfg = col_map.get(doc_key)
    if not cfg:
        print(f"  ⚠  No entry in column_map.json for '{doc_key}' — skipping")
        return []

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    all_rows = []

    for tab_name, tab_cfg in cfg.get("tabs", {}).items():
        if tab_cfg.get("skip"):
            continue
        if tab_name not in wb.sheetnames:
            print(f"  ⚠  Tab '{tab_name}' not found in {doc_key}.xlsx")
            continue

        ws = wb[tab_name]
        header_row_idx = tab_cfg.get("header_row", 2)

        # Build column index map from header row
        col_idx = {}
        for cell in ws[header_row_idx]:
            h = _norm_header(cell.value)
            if h:
                col_idx[h] = cell.column - 1  # 0-based

        ms_col   = _norm_header(tab_cfg.get("main_street"))
        from_col = _norm_header(tab_cfg.get("from_street")) if tab_cfg.get("from_street") else None
        to_col   = _norm_header(tab_cfg.get("to_street"))   if tab_cfg.get("to_street")   else None
        wt_col   = _norm_header(tab_cfg.get("work_type"))   if tab_cfg.get("work_type")   else None
        lim_col  = _norm_header(tab_cfg.get("limits_col"))  if tab_cfg.get("limits_col")  else None

        if ms_col not in col_idx:
            print(f"  ⚠  main_street col '{ms_col}' not found in tab '{tab_name}' — skipping tab")
            continue

        data_start = tab_cfg.get("data_start_row", header_row_idx + 1)
        for row in ws.iter_rows(min_row=data_start, values_only=True):
            vals = list(row)

            def get(col_name):
                if not col_name or col_name not in col_idx:
                    return ""
                idx = col_idx[col_name]
                return str(vals[idx]).strip() if idx < len(vals) and vals[idx] is not None else ""

            main = get(ms_col)
            if not main or main in ("None", ""):
                continue

            from_v, to_v = "", ""
            if lim_col:
                # "FROM to TO" or "FROM TO TO" combined column
                limits = get(lim_col)
                parts = re.split(r"\s+to\s+", limits, maxsplit=1, flags=re.IGNORECASE)
                from_v = parts[0].strip() if len(parts) > 0 else ""
                to_v   = parts[1].strip() if len(parts) > 1 else ""
            else:
                from_v = get(from_col) if from_col else ""
                to_v   = get(to_col)   if to_col   else ""

            wt = get(wt_col) if wt_col else ""

            all_rows.append({
                "main_street": main,
                "from_street": from_v,
                "to_street":   to_v,
                "work_type":   wt,
                "_tab": tab_name,
            })

    return all_rows

# ── Parser runner ─────────────────────────────────────────────────────────────

def run_parser(doc_key: str, use_cache: bool = True) -> list:
    cache_path = os.path.join(CACHE_DIR, doc_key + ".json")

    if use_cache and os.path.exists(cache_path):
        print(f"  📦 Using cached result")
        with open(cache_path) as f:
            return json.load(f)

    # PDF has same base name as xlsx
    pdf_path = os.path.join(PDF_DIR, doc_key + ".pdf")
    if not os.path.exists(pdf_path):
        print(f"  ⚠  PDF not found: {pdf_path}")
        return []

    print(f"  🔄 Parsing {doc_key}.pdf ...")
    headers = {"X-Api-Key": API_KEY} if API_KEY else {}
    with open(pdf_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/parse",
            headers=headers,
            files={"file": (doc_key + ".pdf", f, "application/pdf")},
            timeout=600,
        )
    resp.raise_for_status()
    streets = resp.json().get("streets", [])

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(streets, f, indent=2)

    return streets

# ── Matching ──────────────────────────────────────────────────────────────────

def match_streets(truth: list, parsed: list) -> dict:
    matched, missed, extra = [], [], []
    used = set()

    for t in truth:
        tk = street_key(t)
        best_score, best_idx = 0.0, -1
        for i, p in enumerate(parsed):
            if i in used:
                continue
            ms_score = similarity(tk[0], street_key(p)[0])
            from_ok  = (not tk[1] or not street_key(p)[1] or
                        similarity(tk[1], street_key(p)[1]) > 0.5)
            to_ok    = (not tk[2] or not street_key(p)[2] or
                        similarity(tk[2], street_key(p)[2]) > 0.5)
            if ms_score > best_score and from_ok and to_ok:
                best_score, best_idx = ms_score, i
        if best_score >= FUZZY_THRESHOLD and best_idx >= 0:
            matched.append({"truth": t, "parsed": parsed[best_idx], "score": round(best_score, 2)})
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

def print_report(doc_key: str, r: dict, show_matched: bool = False):
    p, re_, f1 = r["precision"], r["recall"], r["f1"]
    bar = "█" * int(f1 * 20) + "░" * (20 - int(f1 * 20))
    print(f"\n{'='*60}")
    print(f"  {doc_key}")
    print(f"{'='*60}")
    print(f"  Truth: {r['total_truth']}  |  Parsed: {r['total_parsed']}  |  "
          f"Matched: {len(r['matched'])}  Missed: {len(r['missed'])}  Extra: {len(r['extra'])}")
    print(f"  Precision: {p:.0%}   Recall: {re_:.0%}   F1: {f1:.0%}  [{bar}]")

    if show_matched and r["matched"]:
        print(f"\n  ✅ MATCHED ({len(r['matched'])}):")
        for m in r["matched"]:
            t, parsed = m["truth"], m["parsed"]
            print(f"     {t['main_street']} | {t.get('from_street','')} → {t.get('to_street','')}  "
                  f"(score {m['score']}, tab: {t.get('_tab','')})")

    if r["missed"]:
        print(f"\n  ❌ MISSED ({len(r['missed'])}):")
        for s in r["missed"][:30]:
            print(f"     {s['main_street']}  |  {s.get('from_street','')} → {s.get('to_street','')}  "
                  f"[{s.get('_tab','')}]")
        if len(r["missed"]) > 30:
            print(f"     ... and {len(r['missed']) - 30} more")

    if r["extra"]:
        print(f"\n  ➕ EXTRA / false positives ({len(r['extra'])}):")
        for s in r["extra"][:15]:
            print(f"     {s.get('main_street','')}  |  {s.get('from_street','')} → {s.get('to_street','')}  "
                  f"[p.{s.get('page','')}]")
        if len(r["extra"]) > 15:
            print(f"     ... and {len(r['extra']) - 15} more")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", help="Run only docs whose key contains this string")
    parser.add_argument("--no-cache", action="store_true", help="Force re-parse all docs")
    parser.add_argument("--show-matched", action="store_true", help="Print matched rows too")
    args = parser.parse_args()

    with open(COL_MAP) as f:
        col_map = json.load(f)

    # All doc keys = xlsx filenames without extension
    all_keys = [
        os.path.splitext(fn)[0]
        for fn in sorted(os.listdir(GT_DIR))
        if fn.endswith(".xlsx")
    ]

    if args.doc:
        all_keys = [k for k in all_keys if args.doc.lower() in k.lower()]
        if not all_keys:
            print(f"No xlsx file matching '{args.doc}' found in tests/ground_truth/")
            sys.exit(1)

    all_results = {}
    for doc_key in all_keys:
        print(f"\n▶ {doc_key}")
        truth  = load_ground_truth(doc_key, col_map)
        if not truth:
            print(f"  ⚠  No ground truth rows loaded — skipping")
            continue
        print(f"  📋 {len(truth)} ground truth rows loaded")
        parsed = run_parser(doc_key, use_cache=not args.no_cache)
        result = match_streets(truth, parsed)
        all_results[doc_key] = result
        print_report(doc_key, result, show_matched=args.show_matched)

    if len(all_results) > 1:
        total_matched = sum(len(r["matched"]) for r in all_results.values())
        total_missed  = sum(len(r["missed"])  for r in all_results.values())
        total_extra   = sum(len(r["extra"])   for r in all_results.values())
        total_truth   = sum(r["total_truth"]  for r in all_results.values())
        total_parsed  = sum(r["total_parsed"] for r in all_results.values())
        op = total_matched / (total_matched + total_extra) if (total_matched + total_extra) else 0
        or_ = total_matched / total_truth if total_truth else 0
        of1 = 2 * op * or_ / (op + or_) if (op + or_) else 0
        bar = "█" * int(of1 * 20) + "░" * (20 - int(of1 * 20))
        print(f"\n{'='*60}")
        print(f"  OVERALL  ({len(all_results)} documents)")
        print(f"{'='*60}")
        print(f"  Truth: {total_truth}  |  Parsed: {total_parsed}  |  "
              f"Matched: {total_matched}  Missed: {total_missed}  Extra: {total_extra}")
        print(f"  Precision: {op:.0%}   Recall: {or_:.0%}   F1: {of1:.0%}  [{bar}]")

        # Per-doc summary table
        print(f"\n  {'Document':<55} {'P':>5} {'R':>5} {'F1':>5}")
        print(f"  {'-'*70}")
        for dk, r in sorted(all_results.items(), key=lambda x: x[1]['f1']):
            print(f"  {dk:<55} {r['precision']:>4.0%} {r['recall']:>4.0%} {r['f1']:>4.0%}")

if __name__ == "__main__":
    main()
