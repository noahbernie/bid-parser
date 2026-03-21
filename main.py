from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pdfplumber
import anthropic
import io
import base64
import uuid
import os
import json
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

documents = {}

EXTRACT_PROMPT = """You are parsing road construction / bid document pages. Extract the following fields and return ONLY valid JSON — no explanation, no markdown, just the JSON object.

Fields to extract:
- bid_number: the bid or contract number (e.g. "K-26-2431-DBB-3", "R-4656")
- project_name: the name of the project (e.g. "Slurry Seal Group 2627", "2024/2025 Street Preservation")
- city: the city or municipality the work is in
- work_type: type of road work (e.g. "Slurry Seal", "Mill & Overlay", "Preservation", "Full Reconstruction")
- estimated_cost: estimated construction cost if mentioned
- bid_due_date: when bids are due (null if not a bid document)
- streets: a list of ALL street segments found. Each object has:
    - main_street: the street being worked on
    - from_street: the cross street / start point (null if not found)
    - to_street: the cross street / end point (null if not found)
    - work_type: work type for this specific segment if specified (e.g. "Slurry Seal", "Cape Seal", "Crack Fill")
    - location: location group if specified (e.g. "Location 1", "Location 2")

IMPORTANT: Look carefully at any tables. Tables typically have columns like STREET NAME, START, END or FROM, TO. Read every row. There may be dozens or hundreds of street segments — include ALL of them.

If a field cannot be found, use null. Return an empty array [] for streets only if truly none exist.
"""

STREET_KEYWORDS = [
    "street", "ave", "avenue", "blvd", "boulevard", "rd", "road",
    "dr", "drive", "ln", "lane", "ct", "court", "way", "location",
    "limits", "slurry", "overlay", "resurfacing", "mill", "pavement",
    "seal", "attachment a", "exhibit a", "scope of work", "linear feet"
]

def is_relevant_page(text: str) -> bool:
    t = text.lower()
    return sum(1 for kw in STREET_KEYWORDS if kw in t) >= 3

def page_to_base64(pdf_bytes: bytes, page_index: int, resolution: int = 120) -> str:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page = pdf.pages[page_index]
        img = page.to_image(resolution=resolution)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


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
            doc["page_cache"][page_num] = pdf.pages[page_num - 1].extract_text() or ""

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

    full_text = "\n\n--- PAGE BREAK ---\n\n".join(pages_text)
    return {"filename": doc["filename"], "total_pages": doc["total_pages"], "text": full_text}


@app.post("/doc/{doc_id}/extract")
async def extract_schema(doc_id: str):
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    doc = documents[doc_id]

    if doc["extracted_schema"]:
        return doc["extracted_schema"]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    # Scan all pages, find relevant ones
    header_indices = list(range(min(5, doc["total_pages"])))
    street_indices = []

    with pdfplumber.open(io.BytesIO(doc["bytes"])) as pdf:
        for i, page in enumerate(pdf.pages):
            text = doc["page_cache"].get(i + 1) or page.extract_text() or ""
            doc["page_cache"][i + 1] = text
            if i >= 5 and is_relevant_page(text):
                street_indices.append(i)

    # Select pages to send: header pages + up to 15 street pages
    # (vision tokens are expensive — be selective)
    pages_to_send = header_indices + street_indices[:15]
    pages_to_send = sorted(set(pages_to_send))

    # Build vision content blocks — one image per page
    content = [{"type": "text", "text": EXTRACT_PROMPT}]

    for page_idx in pages_to_send:
        content.append({
            "type": "text",
            "text": f"\n--- Page {page_idx + 1} of {doc['total_pages']} ---"
        })
        b64 = page_to_base64(doc["bytes"], page_idx, resolution=120)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64,
            }
        })

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        messages=[{"role": "user", "content": content}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        schema = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Could not parse model response: {raw[:300]}")

    schema["_meta"] = {
        "total_pages": doc["total_pages"],
        "pages_sent_to_ai": len(pages_to_send),
        "street_pages_found": len(street_indices),
    }

    doc["extracted_schema"] = schema
    return schema


@app.get("/documents")
async def list_docs():
    return [
        {"doc_id": k, "filename": v["filename"], "total_pages": v["total_pages"]}
        for k, v in documents.items()
    ]


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
