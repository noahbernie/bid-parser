import urllib.request
import urllib.parse
import json
import io
import pdfplumber
import fitz
import anthropic

CALTRANS_LRS_URL = "https://caltrans-gis.dot.ca.gov/arcgis/rest/services/RH/RestAPI/FeatureServer/0/query"

COUNTY_CODES = {
    "alameda": "ALA", "alpine": "ALP", "amador": "AMA", "butte": "BUT",
    "calaveras": "CAL", "colusa": "COL", "contra costa": "CC", "del norte": "DN",
    "el dorado": "ED", "fresno": "FRE", "glenn": "GLE", "humboldt": "HUM",
    "imperial": "IMP", "inyo": "INY", "kern": "KER", "kings": "KIN",
    "lake": "LAK", "lassen": "LAS", "los angeles": "LA", "madera": "MAD",
    "marin": "MRN", "mariposa": "MPA", "mendocino": "MEN", "merced": "MER",
    "modoc": "MOD", "mono": "MON", "monterey": "MOT", "napa": "NAP",
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

HIGHWAY_PROMPT = """You are parsing a California Caltrans highway construction plan set. Extract project-level fields from the title sheet and first few pages.
Return ONLY valid JSON with these exact fields:
{
  "contract_number": "02-2K2304",
  "project_id": "0225000053",
  "route": "3",
  "county": "Trinity",
  "district": "02",
  "pm_start": "0.0",
  "pm_end": "5.0",
  "pm_prefix": "L",
  "direction": "BOTH",
  "work_type": "micro-surfacing",
  "description": "brief description of work location",
  "plans_date": "December 15, 2025"
}
Rules:
- route: just the number (e.g. "3" not "Route 3")
- county: full county name (e.g. "Trinity" not "TRI")
- pm_start/pm_end: numeric portion only, no prefix letters
- pm_prefix: letter prefix before the PM number ("L", "R", "M", "T", etc.) — null if no prefix
- direction: "BOTH" if work in both directions, else "NB"/"SB"/"EB"/"WB", or null
- work_type: primary pavement work type (e.g. "micro-surfacing", "overlay", "AC replacement", "slurry seal")
Use null for any field not found."""


def county_to_code(county_name: str) -> str:
    return COUNTY_CODES.get(county_name.lower().strip(), county_name.upper()[:3])


def get_route_coords(route: str, county_code: str, pm_prefix: str = None, alignment: str = "R") -> dict:
    """Get start/end lat/lng for a route segment from the Caltrans LRS polyline."""
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

        features = data.get("features", [])
        if not features:
            return {"error": f"No features found for RouteId={route_id}"}

        all_points = []
        for feat in features:
            for path in feat.get("geometry", {}).get("paths", []):
                all_points.extend(path)

        if not all_points:
            return {"error": "Geometry had no path points"}

        return {
            "route_id": route_id,
            "start": {"lat": all_points[0][1], "lng": all_points[0][0]},
            "end": {"lat": all_points[-1][1], "lng": all_points[-1][0]},
            "point_count": len(all_points),
        }
    except Exception as e:
        return {"error": str(e)}


def extract_text_first_pages(pdf_bytes: bytes, max_pages: int = 5) -> str:
    """Extract text from first N pages for header extraction."""
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages[:max_pages]):
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


def call_claude(client, prompt: str, text: str, max_tokens: int = 2048) -> dict:
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "text", "text": text},
        ]}],
    ) as stream:
        msg = stream.get_final_message()
    raw = msg.content[0].text.strip()
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
        text = extract_text_first_pages(doc["bytes"], max_pages=5)
    except Exception as e:
        log(f"✗ Text extraction failed: {e}")
        doc["progress"]["done"] = True
        return

    log("Calling Claude to extract project info...")
    client = anthropic.Anthropic(api_key=api_key)
    try:
        schema = call_claude(client, HIGHWAY_PROMPT, text[:20000])
        log(f"✓ Route {schema.get('route')} | {schema.get('county')} | {schema.get('contract_number')}")
        prefix = schema.get("pm_prefix") or ""
        log(f"  PM {prefix}{schema.get('pm_start')} → {prefix}{schema.get('pm_end')}")
    except Exception as e:
        log(f"✗ Claude extraction failed: {e}")
        doc["progress"]["done"] = True
        return

    log("Resolving coordinates from Caltrans LRS...")
    county_code = county_to_code(schema.get("county", ""))
    route = schema.get("route", "")
    pm_prefix = schema.get("pm_prefix") or None

    coords = get_route_coords(route, county_code, pm_prefix, alignment="R")
    if coords and "error" not in coords:
        schema["coordinates"] = coords
        log(f"✓ Start: {coords['start']['lat']:.5f}, {coords['start']['lng']:.5f}")
        log(f"✓ End:   {coords['end']['lat']:.5f}, {coords['end']['lng']:.5f}")
    else:
        schema["coordinates"] = None
        log(f"⚠ Could not resolve coordinates: {coords.get('error')}")

    doc["schema"] = schema
    doc["progress"]["done"] = True
    log("✓ Done!")
