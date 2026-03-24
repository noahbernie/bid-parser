from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
import uuid
import os
import asyncio

from highway.parser import run_highway_extraction

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

router = APIRouter(prefix="/highway")

highway_docs = {}


@router.get("")
async def highway_index():
    return FileResponse(os.path.join(STATIC_DIR, "highway.html"))


@router.post("/upload")
async def upload_highway_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    contents = await file.read()
    doc_id = str(uuid.uuid4())[:8]
    highway_docs[doc_id] = {
        "filename": file.filename,
        "bytes": contents,
        "schema": None,
        "progress": None,
    }
    return {"doc_id": doc_id, "filename": file.filename}


@router.get("/doc/{doc_id}/pdf")
async def get_pdf(doc_id: str):
    if doc_id not in highway_docs:
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(
        content=highway_docs[doc_id]["bytes"],
        media_type="application/pdf",
    )


@router.post("/doc/{doc_id}/extract")
async def extract_highway(doc_id: str):
    if doc_id not in highway_docs:
        raise HTTPException(status_code=404, detail="Document not found")
    doc = highway_docs[doc_id]
    if doc.get("progress") is not None:
        return {"status": "already_running"}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    doc["progress"] = {"logs": [], "done": False}
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, run_highway_extraction, doc_id, api_key, highway_docs)
    return {"status": "started"}


@router.get("/doc/{doc_id}/status")
async def highway_status(doc_id: str):
    if doc_id not in highway_docs:
        raise HTTPException(status_code=404, detail="Document not found")
    doc = highway_docs[doc_id]
    return {
        "done": doc["progress"]["done"] if doc.get("progress") else False,
        "logs": doc["progress"]["logs"] if doc.get("progress") else [],
        "schema": doc["schema"],
    }


@router.delete("/doc/{doc_id}")
async def clear_highway_doc(doc_id: str):
    if doc_id not in highway_docs:
        raise HTTPException(status_code=404, detail="Document not found")
    highway_docs[doc_id]["schema"] = None
    highway_docs[doc_id]["progress"] = None
    return {"status": "cleared"}
