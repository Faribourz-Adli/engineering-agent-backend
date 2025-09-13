import os
import io
import re
import json
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from pdfminer.high_level import extract_text

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
RAW_BUCKET = os.environ.get("SUPABASE_RAW_BUCKET", "raw")
TEXT_BUCKET = os.environ.get("SUPABASE_TEXT_BUCKET", "text")
JSON_BUCKET = os.environ.get("SUPABASE_JSON_BUCKET", "json")

app = FastAPI(title="Engineering Agent API")

@app.get("/api/debug/env")
def debug_env():
    return {
        "has_url": bool(SUPABASE_URL),
        "has_service_key": bool(SUPABASE_SERVICE_KEY),
        "has_anon_key": bool(SUPABASE_ANON_KEY),
        "buckets": {"raw": RAW_BUCKET, "text": TEXT_BUCKET, "json": JSON_BUCKET}
    }

def supabase_admin() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Missing Supabase env vars")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

class StartUploadReq(BaseModel):
    user_email: str
    org_name: Optional[str] = None
    note: Optional[str] = None

class EnqueueProcessReq(BaseModel):
    upload_session_id: str
    file_path: str   # e.g., "clientA/API_6D.pdf" inside RAW_BUCKET

def simple_extract_metadata(text: str):
    first_line = text.strip().splitlines()[0] if text.strip().splitlines() else ""
    title = first_line[:200]

    m = re.search(r'\b(API\s?\d+[A-Z]?|ISO\s?\d+(?:-\d+)?|ASME\s?[A-Z]?\d[\d\.]*|IEC\s?\d+)\b', text[:2000], re.I)
    code = m.group(0).upper() if m else None

    dm = re.search(r'(20\d{2}|19\d{2})[-/\.]([01]?\d)[-/\.]([0-3]?\d)', text[:4000])
    rev_date = None
    if dm:
        y, mth, d = dm.group(1), dm.group(2), dm.group(3)
        try:
            rev_date = datetime(int(y), int(mth), int(d)).date().isoformat()
        except:
            pass

    pub = None
    for k in ["API", "ISO", "ASME", "IEC"]:
        if re.search(r'\b'+k+r'\b', text[:2000], re.I):
            pub = k
            break

    return {
        "code": code,
        "title": title or (code or "Untitled Standard"),
        "publisher": pub or "Company",
        "rev_date": rev_date
    }

def segment_sections(text: str):
    lines = [l for l in text.splitlines()]
    sections = []
    current = {"path": "0", "heading": "Document", "body": []}
    for ln in lines:
        if re.match(r'^(\d+(\.\d+)*)\s+', ln.strip()):
            sections.append(current)
            current = {"path": ln.strip().split()[0], "heading": ln.strip(), "body": []}
        else:
            current["body"].append(ln)
    sections.append(current)
    return sections

import re as _re
TERM_PATTERNS = {
    "pressure": _re.compile(r'\b(\d{1,4}(?:\.\d+)?)\s*(bar|psi|kPa|MPa)\b', _re.I),
    "temperature": _re.compile(r'\b(-?\d{1,3}(?:\.\d+)?)\s*(°C|°F|C|F)\b', _re.I),
    "class": _re.compile(r'\bClass\s?(150|300|600|900|1500|2500)\b', _re.I),
    "material": _re.compile(r'\b(316L|304L|A105|F316L|F51|Duplex|Superduplex|AISI\s?\d{3})\b', _re.I),
    "xref": _re.compile(r'\b(API\s?\d+[A-Z]?|ISO\s?\d+(?:-\d+)?|ASME\s?[A-Z]?\d[\d\.]*|IEC\s?\d+|NACE\s?\w+\d+)\b', _re.I),
}

def extract_terms(text: str, path: str):
    found = []
    for tname, pat in TERM_PATTERNS.items():
        for m in pat.finditer(text):
            val = m.group(0)
            unit = None
            if tname in ("pressure", "temperature"):
                unit = m.group(2) if m.lastindex and m.lastindex >= 2 else None
            found.append({"term": tname, "value": val, "unit": unit, "context_path": path, "confidence": 0.8})
    return found

def download_from_storage(file_path: str) -> bytes:
    sb = supabase_admin()
    resp = sb.storage.from_(RAW_BUCKET).download(file_path)
    return resp

def extract_text_from_pdf(content: bytes) -> str:
    # Try native PDF text first
    try:
        with io.BytesIO(content) as fh:
            text = extract_text(fh) or ""
            if text.strip():
                return text
    except Exception:
        pass

    # Fallback to OCR only if needed (lazy import to avoid startup issues)
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(io.BytesIO(content))
        return pytesseract.image_to_string(img)
    except Exception:
        return ""

def upsert_standard_and_children(meta, text, session_id) -> str:
    sb = supabase_admin()

    st = sb.table("standards").insert({
        "code": meta.get("code"),
        "title": meta.get("title"),
        "publisher": meta.get("publisher"),
        "revision": None,
        "rev_date": meta.get("rev_date"),
        "status": "current",
    }).execute()
    if not st.data:
        raise HTTPException(500, "Failed to insert standard")
    standard_id = st.data[0]["id"]

    sections = segment_sections(text)
    rows = []
    for idx, s in enumerate(sections):
        rows.append({
            "standard_id": standard_id,
            "path": s["path"],
            "heading": s["heading"][:500],
            "body_text": "\n".join(s["body"])[:200000],
            "order_index": idx
        })
    if rows:
        sb.table("standard_sections").insert(rows).execute()

    term_rows = []
    for s in sections:
        terms = extract_terms("\n".join(s["body"]), s["path"])
        for t in terms:
            t["standard_id"] = standard_id
        term_rows.extend(terms)
    if term_rows:
        sb.table("extracted_terms").insert(term_rows).execute()

    sb.table("processing_jobs").update(
        {"standard_id": standard_id, "status": "done", "finished_at": datetime.utcnow().isoformat()}
    ).eq("upload_session_id", session_id).eq("status", "processing").execute()

    return standard_id

from fastapi.responses import JSONResponse

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/upload/start")
def start_upload(req: StartUploadReq):
    sb = supabase_admin()
    org_id = None
    if req.org_name:
        got = sb.table("organizations").select("id").eq("name", req.org_name).limit(1).execute()
        if got.data:
            org_id = got.data[0]["id"]
        else:
            ins = sb.table("organizations").insert({"name": req.org_name, "type": "Other"}).execute()
            org_id = ins.data[0]["id"]

    session = sb.table("upload_sessions").insert({
        "user_id": None,
        "org_id": org_id,
        "note": req.note or "MVP upload"
    }).execute()
    return {"upload_session_id": session.data[0]["id"]}

@app.post("/api/process/enqueue")
def enqueue(req: EnqueueProcessReq):
    sb = supabase_admin()
    sb.table("processing_jobs").insert({
        "upload_session_id": req.upload_session_id,
        "status": "processing",
        "started_at": datetime.utcnow().isoformat()
    }).execute()

    try:
        content = download_from_storage(req.file_path)
        text = extract_text_from_pdf(content)
        if not text.strip():
            raise HTTPException(400, "Could not extract text (PDF may be fully scanned; OCR config needed).")
        meta = simple_extract_metadata(text)
        std_id = upsert_standard_and_children(meta, text, req.upload_session_id)

        text_key = f"{uuid.uuid4()}.txt"
        sb.storage.from_(TEXT_BUCKET).upload(text_key, text.encode("utf-8"))
        sb.storage.from_(JSON_BUCKET).upload(f"{std_id}.json", json.dumps({"meta": meta}).encode("utf-8"))

        return {"ok": True, "standard_id": std_id}
    except HTTPException as e:
        sb.table("processing_jobs").update(
            {"status": "failed", "finished_at": datetime.utcnow().isoformat(), "log": str(e.detail)}
        ).eq("upload_session_id", req.upload_session_id).execute()
        return JSONResponse(status_code=400, content={"error": str(e.detail)})
    except Exception as e:
        sb.table("processing_jobs").update(
            {"status": "failed", "finished_at": datetime.utcnow().isoformat(), "log": str(e)}
        ).eq("upload_session_id", req.upload_session_id).execute()
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {e}"})

@app.get("/api/standards")
def list_standards(q: Optional[str] = None, publisher: Optional[str] = None, limit: int = 50):
    sb = supabase_admin()
    query = sb.table("standards").select("*").limit(limit).order("created_at", desc=True)
    if publisher:
        query = query.eq("publisher", publisher)
    if q:
        query = query.ilike("title", f"%{q}%")
    res = query.execute()
    return {"items": res.data}

@app.get("/api/standards/{standard_id}")
def standard_detail(standard_id: str):
    sb = supabase_admin()
    std = sb.table("standards").select("*").eq("id", standard_id).single().execute().data
    if not std:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    secs = sb.table("standard_sections").select("*").eq("standard_id", standard_id).order("order_index").execute().data
    terms = sb.table("extracted_terms").select("*").eq("standard_id", standard_id).execute().data
    return {"standard": std, "sections": secs, "terms": terms}
