import urllib.request
import urllib.parse
import json
import io
import math
import base64
import pdfplumber
import fitz
import anthropic

CALTRANS_LRS_URL = "https://caltrans-gis.dot.ca.gov/arcgis/rest/services/RH/RestAPI/FeatureServer/0/query"
POSTMILES_TENTH_URL = "https://caltrans-gis.dot.ca.gov/arcgis/rest/services/CHhighway/SHN_Postmiles_Tenth/FeatureServer/0/query"

COUNTY_CODES = {
    "alameda": "ALA", "alpine": "ALP", "amador": "AMA", "butte": "BUT",
    "calaveras": "CAL", "colusa": "COL", "contra costa": "CC", "del norte": "DN",
    "el dorado": "ED", "fresno": "FRE", "glenn": "GLE", "humboldt": "HUM",
    "imperial": "IMP", "inyo": "INY", "kern": "KER", "kings": "KIN",
    "lake": "LAK", "lassen": "LAS", "los angeles": "LA", "madera": "MAD",
    "marin": "MRN", "mariposa": "MPA", "mendocino": "MEN", "merced": "MER",
    "modoc": "MOD", "mono": "MNO", "monterey": "MOT", "napa": "NAP",
    "nevada": "NEV", "orange": "ORA", "placer": "PLA", "plumas": "PLU",
    "riverside": "RIV", "sacramento": "SAC", "san benito": "SBT",
    "san bernardino": "SBD", "san diego": "SD", "san francisco": "SF",
    "san joaquin": "SJ", "san luis obispo": "SLO", "san mateo": "SM",
    "santa barbara": "SB", "santa clara": "SCL", "santa cruz": "SCZ",
    "shasta": "SHA", "sierra": "SIE", "siskiyou": "SIS", "solano": "SOL",
    "sonoma": "SON", "stanislaus": "STA", "sutter": "SUT", "tehama": "TEH",
    "trinity": "TRI", "tulare": "TUL", "tuolumne": "TUO", "ventura": "VEN",
    "yolo": "YOL", "yuba": "YUB",
}

HIGHWAY_PROMPT = """You are parsing a California Caltrans highway construction plan set. Extract ALL of the following from the plan sheets.
Return ONLY valid JSON with this exact structure:

{
  "contract_number": "02-2K2304",
  "project_id": "0225000053",
  "route": "3",
  "county": "Trinity",
  "district": "02",
  "direction": "BOTH",
  "work_type": "micro-surfacing",
  "description": "brief description of work location",
  "plans_date": "December 15, 2025",
  "pm_segments": [
    {"prefix": "L", "start": 0.0, "end": 4.79},
    {"prefix": null, "start": 0.0, "end": 5.0}
  ],
  "work_segments": [
    {"start_prefix": "L", "start": 0.0, "end_prefix": null, "end": 5.0}
  ],
  "features": [
    {
      "type": "road_connection",
      "pm": 0.54,
      "pm_prefix": null,
      "side": "Rt",
      "road_name": "DOBBINS CREEK Rd"
    }
  ]
}

Rules for header fields:
- route: just the number (e.g. "3" not "Route 3")
- county: full county name
- direction: "BOTH", "NB", "SB", "EB", "WB", or null
- work_type: primary pavement work type

Rules for pm_segments — list EVERY distinct PM prefix range used in this project:
- prefix: the letter prefix ("L", "R", "M", "T", etc.) or null for no prefix
- start: numeric start PM for that prefix segment
- end: numeric end PM for that prefix segment
- Look for station equations (e.g. "PM L4.790 EQUATES TO PM 0.000", "PM R52.41 Bk = PM 51.63 Ahd") to find segment boundaries

Rules for work_segments — list each CONTINUOUS geographic paving run:
- One entry per continuous run. Only split at explicit "BEGIN BREAK IN CONSTRUCTION" / "END BREAK IN CONSTRUCTION" labels.
- A run that spans a PM prefix change is still ONE entry (e.g. L-prefix connects to null-prefix via station equation → single entry).
- start_prefix: PM prefix letter at the beginning of this run (null if none)
- start: numeric PM at the beginning of this run
- end_prefix: PM prefix letter at the end of this run (null if none)
- end: numeric PM at the end of this run
- Example with no breaks (Trinity-style): [{"start_prefix":"L","start":0.0,"end_prefix":null,"end":5.0}]
- Example with one break (SB Route 25-style): [{"start_prefix":"R","start":49.8,"end_prefix":"R","end":51.355}, {"start_prefix":"R","start":51.924,"end_prefix":null,"end":51.7}]

Rules for features — extract ALL road connections, driveways, intersections, and pullouts:
- From driveway/road connection/pullout tables if present
- From construction detail sheet labels (e.g. "HILLCREST ROAD REPAIRS AND CONFORMS PM R50.52", "SANTA ANA ROAD REPAIRS AND CONFORMS PM R51.10")
- From intersection labels on plan and detail sheets
- road_name: the exact road/street name. Use null ONLY if it is a plain unnamed driveway.
- type:
  - "pullout" if the row is from a pullouts table
  - "driveway" if road_name is null (unnamed driveway)
  - "road_connection" if road_name is non-null (any named road, street, or intersection)
- pm: numeric postmile value as a number (e.g. 0.54, 50.52)
- pm_prefix: the prefix letter on THIS feature's PM (e.g. "R" for R50.52, "L" for L3.98) — null if no prefix
- side: "Lt" or "Rt" if shown, otherwise null

Use null for any field not found."""


def county_to_code(county_name: str) -> str:
    return COUNTY_CODES.get(county_name.lower().strip(), county_name.upper()[:3])


def get_polyline_points(route: str, county_code: str, pm_prefix: str = None, alignment: str = "R") -> list:
    """Fetch all [lng, lat] points for a route segment from Caltrans LRS."""
    route_padded = str(route).zfill(3)
    prefix_part = pm_prefix if pm_prefix else "."
    route_id = f"{county_code}{route_padded}.{prefix_part}.{alignment}"

    params = urllib.parse.urlencode({
        "where": f"RouteId='{route_id}'",
        "outFields": "RouteId",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "json",
    })
    url = f"{CALTRANS_LRS_URL}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        all_points = []
        for feat in data.get("features", []):
            for path in feat.get("geometry", {}).get("paths", []):
                all_points.extend(path)
        return all_points
    except Exception:
        return []


def pm_to_coords(route: str, county_code: str, pm_value: float, pm_prefix: str = "") -> dict:
    """Get hundredths-precise lat/lng by fetching the two surrounding tenth-mile points
    from Caltrans Postmiles_Tenth and linearly interpolating between them."""
    floor_tenth = math.floor(pm_value * 10) / 10
    ceil_tenth = round(floor_tenth + 0.1, 1)
    lo = round(floor_tenth - 0.01, 3)
    hi = round(ceil_tenth + 0.01, 3)
    prefix_sql = pm_prefix if pm_prefix else ""

    params = urllib.parse.urlencode({
        "where": f"Route={int(route)} AND County='{county_code}' AND PMPrefix='{prefix_sql}' AND PM >= {lo} AND PM <= {hi}",
        "outFields": "PM,PMPrefix,AlignCode",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "json",
    })
    url = f"{POSTMILES_TENTH_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        features = data.get("features", [])
        if not features:
            return None

        # Prefer AlignCode='Right'
        right = [f for f in features if f.get("attributes", {}).get("AlignCode") == "Right"]
        candidates = right if right else features

        def get_pt(f):
            g = f.get("geometry", {})
            return f["attributes"].get("PM"), g.get("x"), g.get("y")

        pts = [get_pt(f) for f in candidates if f.get("geometry")]
        pts = [(pm, x, y) for pm, x, y in pts if pm is not None and x is not None and y is not None]
        if not pts:
            return None

        # Find the two bracketing points
        below = [(pm, x, y) for pm, x, y in pts if pm <= pm_value + 0.001]
        above = [(pm, x, y) for pm, x, y in pts if pm >= pm_value - 0.001]

        if below and above:
            p0 = max(below, key=lambda p: p[0])
            p1 = min(above, key=lambda p: p[0])
            if abs(p1[0] - p0[0]) < 0.001:
                return {"lat": p0[2], "lng": p0[1]}
            t = (pm_value - p0[0]) / (p1[0] - p0[0])
            lng = p0[1] + t * (p1[1] - p0[1])
            lat = p0[2] + t * (p1[2] - p0[2])
            return {"lat": lat, "lng": lng}

        # Fallback: nearest single point
        best = min(pts, key=lambda p: abs(p[0] - pm_value))
        return {"lat": best[2], "lng": best[1]}
    except Exception:
        pass
    return None


def extract_text_all_pages(pdf_bytes: bytes) -> str:
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            is_large = max(page.width, page.height) > 1008
            if is_large:
                try:
                    fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    text = fitz_doc[i].get_text("text")
                    fitz_doc.close()
                except Exception:
                    text = page.extract_text() or ""
            else:
                text = page.extract_text() or ""
            pages.append(f"--- Page {i+1} ---\n{text}")
    return "\n\n".join(pages)


def render_pages_as_images(pdf_bytes: bytes, page_indices: list, dpi: int = 150) -> list:
    """Render specified page indices as base64 PNG images."""
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for i in page_indices:
        if i >= len(doc):
            continue
        pix = doc[i].get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        images.append(base64.standard_b64encode(img_bytes).decode())
    doc.close()
    return images


def call_claude(client, prompt: str, text: str, images: list = None, max_tokens: int = 4096) -> dict:
    content = [{"type": "text", "text": prompt}]
    if images:
        for b64 in images:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}})
    content.append({"type": "text", "text": text})
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        msg = stream.get_final_message()
    raw = msg.content[0].text.strip()
    if not raw:
        stop_reason = getattr(msg, "stop_reason", "unknown")
        raise ValueError(f"Claude returned empty response (stop_reason={stop_reason})")
    if "```" in raw:
        for part in raw.split("```"):
            if part.startswith("json"):
                raw = part[4:].strip()
                break
            elif part.strip().startswith("{"):
                raw = part.strip()
                break
    return json.loads(raw)


def run_highway_extraction(doc_id: str, api_key: str, highway_docs: dict):
    doc = highway_docs[doc_id]

    def log(msg):
        doc["progress"]["logs"].append(msg)

    log("Extracting text from PDF...")
    try:
        text = extract_text_all_pages(doc["bytes"])
    except Exception as e:
        log(f"✗ Text extraction failed: {e}")
        doc["progress"]["done"] = True
        return

    log("Rendering overview pages as images...")
    try:
        page_images = render_pages_as_images(doc["bytes"], [0], dpi=100)
        log(f"  Rendered {len(page_images)} page images")
    except Exception as e:
        log(f"  ⚠ Image render failed, continuing text-only: {e}")
        page_images = []

    log("Calling Claude to extract project info and features...")
    client = anthropic.Anthropic(api_key=api_key)
    try:
        schema = call_claude(client, HIGHWAY_PROMPT, text[:40000], images=page_images, max_tokens=8192)
        log(f"✓ Route {schema.get('route')} | {schema.get('county')} | {schema.get('contract_number')}")
        features = schema.get("features", [])
        # Enforce type based on road_name: named road = road_connection, "DRIVEWAY"/null = driveway
        for feat in features:
            if feat.get("type") == "pullout":
                continue
            rn = feat.get("road_name")
            feat["type"] = "driveway" if (rn is None or str(rn).strip().upper() == "DRIVEWAY") else "road_connection"
        log(f"  {len(features)} point features found (driveways/connections/pullouts)")
    except Exception as e:
        log(f"✗ Claude extraction failed: {e}")
        doc["progress"]["done"] = True
        return

    # --- Resolve coordinates ---
    county_code = county_to_code(schema.get("county", ""))
    route = schema.get("route", "")
    pm_segments = schema.get("pm_segments") or []

    # Main segment endpoints — use work_segments for the actual paving run extents
    log("Resolving main segment endpoints from Caltrans Postmiles...")
    work_segs = schema.get("work_segments") or []
    first_wseg = work_segs[0] if work_segs else None
    last_wseg = work_segs[-1] if work_segs else None
    start_prefix = (first_wseg.get("start_prefix") or "") if first_wseg else ""
    start_pm = float(first_wseg.get("start") or 0) if first_wseg else 0.0
    end_prefix = (last_wseg.get("end_prefix") or "") if last_wseg else ""
    end_pm = float(last_wseg.get("end") or 0) if last_wseg else 0.0

    start_pt = pm_to_coords(route, county_code, start_pm, start_prefix)
    if not start_pt:
        for delta in [0.1, 0.2, 0.5, 1.0]:
            start_pt = pm_to_coords(route, county_code, start_pm + delta, start_prefix)
            if start_pt:
                break

    end_pt = pm_to_coords(route, county_code, end_pm, end_prefix)
    if not end_pt:
        for delta in [0.1, 0.2, 0.5, 1.0]:
            end_pt = pm_to_coords(route, county_code, end_pm - delta, end_prefix)
            if end_pt:
                break
    if start_pt and end_pt:
        schema["coordinates"] = {
            "start": {"lat": round(start_pt["lat"], 6), "lng": round(start_pt["lng"], 6)},
            "end": {"lat": round(end_pt["lat"], 6), "lng": round(end_pt["lng"], 6)},
        }
        log(f"✓ Start (PM {start_prefix}{start_pm}): {start_pt['lat']:.5f}, {start_pt['lng']:.5f}")
        log(f"✓ End   (PM {end_prefix}{end_pm}): {end_pt['lat']:.5f}, {end_pt['lng']:.5f}")
    else:
        schema["coordinates"] = None
        log("⚠ Could not resolve main segment endpoints")

    # Work segment coordinates — one start/end pair per continuous paving run
    if work_segs:
        seg_coords = []
        for seg in work_segs:
            s_start_prefix = seg.get("start_prefix") or ""
            s_end_prefix = seg.get("end_prefix") or ""
            s_start = float(seg.get("start") or 0)
            s_end = float(seg.get("end") or 0)
            sp = pm_to_coords(route, county_code, s_start, s_start_prefix)
            ep = pm_to_coords(route, county_code, s_end, s_end_prefix)
            if sp and ep:
                seg_coords.append({
                    "start": {"lat": round(sp["lat"], 6), "lng": round(sp["lng"], 6)},
                    "end": {"lat": round(ep["lat"], 6), "lng": round(ep["lng"], 6)},
                })
        if seg_coords:
            schema["segment_coords"] = seg_coords
            log(f"✓ Resolved {len(seg_coords)}/{len(work_segs)} work segment polylines")

    # Point features — look up each PM directly from Postmiles_Tenth
    log(f"Resolving coordinates for {len(features)} features...")
    resolved = 0
    for feat in features:
        feat_prefix = feat.get("pm_prefix") or ""
        feat_pm = feat.get("pm")
        if feat_pm is None:
            feat["lat"] = None
            feat["lng"] = None
            continue
        pt = pm_to_coords(route, county_code, float(feat_pm), feat_prefix)
        if pt:
            feat["lat"] = round(pt["lat"], 6)
            feat["lng"] = round(pt["lng"], 6)
            resolved += 1
        else:
            feat["lat"] = None
            feat["lng"] = None

    schema["features"] = features
    log(f"✓ Resolved {resolved}/{len(features)} feature coordinates")

    doc["schema"] = schema
    doc["progress"]["done"] = True
    log("✓ Done!")
