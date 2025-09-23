import os
import io
import re
import json
import uuid
import glob
import csv
import tempfile
import subprocess
from datetime import datetime
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, Response
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client, Client

# pdf parsing
from pdfminer.high_level import extract_text as pm_extract

# PDF export
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth


# =========================
# ENV / GLOBALS
# =========================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

RAW_BUCKET = os.environ.get("SUPABASE_RAW_BUCKET", "raw")
TEXT_BUCKET = os.environ.get("SUPABASE_TEXT_BUCKET", "text")
JSON_BUCKET = os.environ.get("SUPABASE_JSON_BUCKET", "json")

app = FastAPI(title="Engineering Agent API")
# --- HEALTH & SMOKE ENDPOINTS (do not remove) ---
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/ingest/test")
async def ingest_test(file: UploadFile = File(...)):
    head = await file.read(256)  # read a small chunk only
    return {"filename": file.filename, "bytes": len(head)}
# --- end ---

def supabase_admin() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Missing Supabase env vars")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# =========================
# REQUEST MODELS
# =========================
class StartUploadReq(BaseModel):
    user_email: str
    org_name: Optional[str] = None
    note: Optional[str] = None


class EnqueueProcessReq(BaseModel):
    upload_session_id: str
    file_path: str                  # e.g., "API RP 14C 8TH ED (E1).pdf"
    force_ocr: bool = False         # OCR-first path for scanned PDFs
    pages_sample: int = 0          # how many pages to sample for extraction


class BuildConflictsReq(BaseModel):
    std_a_id: str
    std_b_id: str


# =========================
# TEXT QUALITY / OCR HELPERS
# =========================
def _text_useful(t: str) -> bool:
    """Heuristic check that text looks like real technical content."""
    if not t:
        return False
    t = t.strip()
    if len(t) < 200:
        return False
    return len(re.findall(r'[A-Za-z]{3,}', t)) > 20


def _has_signal_terms(t: str) -> bool:
    """Look for standard-like codes or typical engineering tokens."""
    patterns = [
        r'\bAPI\s*(?:Spec(?:ification)?|Std(?:ard)?|RP|Recommended\s*Practice|Bulletin|Bull|MPMS)?\s*[A-Z]?\d+[A-Z]?\b',
        r'\bISO\s*\d+(?:-\d+)*\b',
        r'\bASME\s*[A-Z]?\d[\d\.]*\b',
        r'\bIEC\s*\d+\b',
        r'\bClass\s*(150|300|600|900|1500|2500)\b',
        r'\b\d+(?:\.\d+)?\s*(bar|psi|kPa|MPa)\b',
        r'\b-?\d+(?:\.\d+)?\s*(°C|°F|C|F)\b',
    ]
    return any(re.search(p, t, re.I) for p in patterns)


def _copyright_heavy(t: str) -> bool:
    t_upper = (t or "").upper()
    return t_upper.count("COPYRIGHT") >= 5 or "INFORMATION HANDLING SERVICES" in t_upper


def _clean_text(t: str) -> str:
    """Normalize common noise + bad encodings."""
    if not t:
        return t
    # remove repeated watermarks
    t = re.sub(r'(?mi)^\s*Accessed by account:.*$', '', t)
    t = re.sub(r'(?mi)^\s*IP address:.*$', '', t)
    # trim many COPYRIGHT lines if we still have enough content
    lines = t.splitlines()
    if sum(1 for ln in lines if "COPYRIGHT" in ln.upper()) >= 5 and len(lines) > 20:
        lines = [ln for ln in lines if "COPYRIGHT" not in ln.upper()]
        t = "\n".join(lines)
    # encoding fixes
    t = (t
         .replace("Â°C", "°C").replace("Â°F", "°F")
         .replace("â€”", "—").replace("â€“", "–")
         .replace("â€", "”").replace("â€œ", "“")
         .replace("â€™", "’").replace("â€¢", "•")
         .replace("â€˜", "‘"))
    return t


def _ocrmypdf(content: bytes) -> bytes:
    """Run OCRmyPDF to add a text layer to scanned PDFs."""
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.pdf")
        out_path = os.path.join(td, "out.pdf")
        with open(in_path, "wb") as f:
            f.write(content)
        cmd = [
            "ocrmypdf", "--skip-text", "--force-ocr",
            "--rotate-pages", "--deskew",
            "--output-type", "pdf",
            in_path, out_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(out_path, "rb") as f:
            return f.read()


def extract_text_from_pdf(content: bytes, pages: int = 15) -> str:
    """
    Balanced path for decent PDFs:
      1) pdftotext (first N pages)
      2) pdfminer fallback
      3) OCRmyPDF → pdftotext
      4) high-DPI image OCR
    """
    # 1) pdftotext
    try:
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "in.pdf")
            out_txt = os.path.join(td, "out.txt")
            with open(in_path, "wb") as f:
                f.write(content)
            subprocess.run(
                ["pdftotext", "-layout", "-f", "1", "-l", str(pages), in_path, out_txt],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if os.path.exists(out_txt):
                with open(out_txt, "r", encoding="utf-8", errors="ignore") as f:
                    t = f.read()
                t = _clean_text(t)
                if t and _text_useful(t) and _has_signal_terms(t) and not _copyright_heavy(t):
                    return t
    except Exception:
        pass

    # 2) pdfminer
    try:
        with io.BytesIO(content) as fh:
            t = pm_extract(fh) or ""
            t = _clean_text(t)
            if t and _text_useful(t) and _has_signal_terms(t):
                return t
    except Exception:
        pass

    # 3) OCRmyPDF → pdftotext
    try:
        ocr_bytes = _ocrmypdf(content)
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "ocr.pdf")
            out_txt = os.path.join(td, "out.txt")
            with open(in_path, "wb") as f:
                f.write(ocr_bytes)
            subprocess.run(
                ["pdftotext", "-layout", "-f", "1", "-l", str(pages), in_path, out_txt],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if os.path.exists(out_txt):
                with open(out_txt, "r", encoding="utf-8", errors="ignore") as f:
                    t = f.read()
                t = _clean_text(t)
                if t and _text_useful(t) and _has_signal_terms(t):
                    return t
            # pdfminer on OCR'd PDF
            with open(in_path, "rb") as fh:
                t = pm_extract(fh) or ""
                t = _clean_text(t)
                if t and _text_useful(t):
                    return t
    except Exception:
        pass

    # 4) high-DPI image OCR (first ~10 pages)
    try:
        from PIL import Image
        import pytesseract
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "in.pdf")
            base = os.path.join(td, "page")
            with open(in_path, "wb") as f:
                f.write(content)
            subprocess.run(
                ["pdftoppm", "-r", "300", "-f", "1", "-l", "10", "-png", in_path, base],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            chunks = []
            for img_path in sorted(glob.glob(base + "-*.png")):
                img = Image.open(img_path)
                chunks.append(pytesseract.image_to_string(img, config="--oem 1 --psm 6 -l eng"))
            t = _clean_text("\n".join(chunks))
            return t
    except Exception:
        pass

    return ""


def extract_text_from_pdf_ocr_first(content: bytes, pages: int = 25) -> str:
    """
    OCR-first path for bad/scanned PDFs:
      1) OCRmyPDF → pdftotext
      2) high-DPI image OCR
      3) pdfminer (last resort)
    """
    # 1) OCRmyPDF → pdftotext
    try:
        ocr_bytes = _ocrmypdf(content)
        with tempfile.TemporaryDirectory() as td:
            ip, out = os.path.join(td, "ocr.pdf"), os.path.join(td, "out.txt")
            with open(ip, "wb") as f:
                f.write(ocr_bytes)
            subprocess.run(["pdftotext", "-layout", "-f", "1", "-l", str(pages), ip, out],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            txt = open(out, "r", encoding="utf-8", errors="ignore").read() if os.path.exists(out) else ""
            txt = _clean_text(txt)
            if txt and _text_useful(txt) and (_has_signal_terms(txt) or len(txt) > 800):
                return txt
    except Exception:
        pass

    # 2) image OCR
    try:
        from PIL import Image
        import pytesseract
        with tempfile.TemporaryDirectory() as td:
            ip, base = os.path.join(td, "in.pdf"), os.path.join(td, "page")
            with open(ip, "wb") as f:
                f.write(content)
            subprocess.run(["pdftoppm", "-r", "300", "-f", "1", "-l", str(min(40, pages)), "-png", ip, base],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            chunks = []
            for img_path in sorted(glob.glob(base + "-*.png")):
                img = Image.open(img_path)
                chunks.append(pytesseract.image_to_string(img, config="--oem 1 --psm 6 -l eng"))
            txt = _clean_text("\n".join(chunks))
            return txt
    except Exception:
        pass

    # 3) pdfminer
    try:
        with io.BytesIO(content) as fh:
            t = pm_extract(fh) or ""
            t = _clean_text(t)
            return t
    except Exception:
        return ""


# =========================
# METADATA / SECTION / TERMS
# =========================
def simple_extract_metadata(text: str, file_name: Optional[str] = None) -> Dict[str, Optional[str]]:
    first_line = text.strip().splitlines()[0] if text.strip().splitlines() else ""
    title = first_line[:200] or "Untitled Standard"

    patterns = [
        r'\b(API\s*(?:Spec(?:ification)?|Std(?:ard)?|RP|Recommended\s*Practice|Bulletin|Bull|MPMS)?\s*[A-Z]?\d+[A-Z]?)\b',
        r'\b(ISO\s*\d+(?:-\d+)*)\b',
        r'\b(ASME\s*[A-Z]?\d[\d\.]*)\b',
        r'\b(IEC\s*\d+)\b',
    ]
    code = None
    for pat in patterns:
        m = re.search(pat, text[:4000], re.I)
        if m:
            code = m.group(1).upper()
            break
    if not code and file_name:
        base = os.path.splitext(os.path.basename(file_name))[0].replace("_", " ")
        for pat in patterns:
            m = re.search(pat, base, re.I)
            if m:
                code = m.group(1).upper()
                break
        if not code:
            code = base.upper()[:100]

    # rough rev date
    dm = re.search(r'(20\d{2}|19\d{2})[-/\.]([01]?\d)[-/\.]([0-3]?\d)', text[:4000])
    rev_date = None
    if dm:
        y, mth, d = dm.group(1), dm.group(2), dm.group(3)
        try:
            rev_date = datetime(int(y), int(mth), int(d)).date().isoformat()
        except Exception:
            pass

    pub = None
    if code:
        if code.startswith("API"): pub = "API"
        elif code.startswith("ISO"): pub = "ISO"
        elif code.startswith("ASME"): pub = "ASME"
        elif code.startswith("IEC"): pub = "IEC"

    return {"code": code, "title": title, "publisher": pub or "Company", "rev_date": rev_date}


def segment_sections(text: str) -> List[Dict[str, str]]:
    """Simple numeric heading splitter: "1", "1.1", etc."""
    lines = text.splitlines()
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


# term patterns (tight temperature to avoid "API 14C/14F")
TERM_PATTERNS = {
    "pressure": re.compile(r'\b(\d{1,4}(?:\.\d+)?)\s*(bar|psi|kPa|MPa)\b', re.I),
    "temperature": re.compile(r'\b(-?\d{1,3}(?:\.\d+)?)\s*(°C|°F)\b', re.I),
    "class": re.compile(r'\bClass\s?(150|300|600|900|1500|2500)\b', re.I),
    "material": re.compile(r'\b(316L|304L|A105|F316L|F51|Duplex|Superduplex|AISI\s?\d{3})\b', re.I),
    "xref": re.compile(
        r'\b(API\s*(?:Spec(?:ification)?|Std(?:ard)?|RP|Recommended\s*Practice|Bulletin|Bull|MPMS)?\s*[A-Z]?\d+[A-Z]?)\b'
        r'|\b(ISO\s*\d+(?:-\d+)*)\b|\b(ASME\s*[A-Z]?\d[\d\.]*)\b|\b(IEC\s*\d+)\b', re.I),
}


def extract_terms(text: str, path: str) -> List[Dict[str, object]]:
    found = []
    for tname, pat in TERM_PATTERNS.items():
        for m in pat.finditer(text):
            val, unit = m.group(0), None
            if tname in ("pressure", "temperature"):
                # use unit group (2) for these
                try:
                    unit = m.group(2)
                except Exception:
                    unit = None
            found.append({
                "term": tname, "value": val, "unit": unit,
                "context_path": path, "confidence": 0.8
            })
    return found


def download_from_storage(file_path: str) -> bytes:
    sb = supabase_admin()
    return sb.storage.from_(RAW_BUCKET).download(file_path)


def _norm_code(s: str) -> str:
    if not s:
        return ""
    s = s.upper()
    for w in ["BULLETIN", "BULL", "SPECIFICATION", "SPEC", "STANDARD", "STD",
              "RECOMMENDED PRACTICE", "RECOMMENDEDPRACTICE", "RP", "MPMS"]:
        s = s.replace(w, "")
    s = re.sub(r"[^A-Z0-9\.]", "", s)
    return s


# =========================
# CONFLICT HELPERS
# =========================
_num = re.compile(r"[-+]?\d+(?:\.\d+)?")

def _to_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = _num.search(s)
    return float(m.group(0)) if m else None

def _to_bar(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    u = (unit or "").lower()
    if u == "bar": return value
    if u == "psi": return value * 0.0689476
    if u == "kpa": return value / 100.0
    if u == "mpa": return value * 10.0
    return value

def _to_celsius(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    u = (unit or "").lower().replace("°", "")
    if u in ("c", "°c"): return value
    if u in ("f", "°f"): return (value - 32.0) * 5.0 / 9.0
    return value

def _pick_first(terms: List[dict], name: str) -> Optional[dict]:
    for t in terms or []:
        if t.get("term") == name:
            return t
    return None


# =========================
# UPSERT PIPELINE
# =========================
def upsert_standard_and_children(meta, text, session_id, file_name: Optional[str] = None) -> str:
    sb = supabase_admin()

    # dedupe by code if available
    standard_id = None
    if meta.get("code"):
        got = sb.table("standards").select("id").eq("code", meta["code"]).limit(1).execute().data
        if got:
            standard_id = got[0]["id"]
            sb.table("standards").update({
                "title": meta.get("title"),
                "publisher": meta.get("publisher"),
                "rev_date": meta.get("rev_date"),
                "status": "current",
                "source_file_url": f"raw://{file_name}" if file_name else None
            }).eq("id", standard_id).execute()

    if not standard_id:
        st = sb.table("standards").insert({
            "code": meta.get("code"),
            "title": meta.get("title"),
            "publisher": meta.get("publisher"),
            "revision": None,
            "rev_date": meta.get("rev_date"),
            "status": "current",
            "source_file_url": f"raw://{file_name}" if file_name else None
        }).execute()
        if not st.data:
            raise HTTPException(500, "Failed to insert standard")
        standard_id = st.data[0]["id"]

    # idempotent cleanup for reprocessing
    sb.table("standard_sections").delete().eq("standard_id", standard_id).execute()
    sb.table("extracted_terms").delete().eq("standard_id", standard_id).execute()

    # sections + terms (unique path using index suffix)
    sections = segment_sections(text)
    rows = []
    term_rows = []
    for idx, s in enumerate(sections):
        path = f"{s['path']}|{idx}"  # ensure uniqueness for (standard_id, path)
        body = "\n".join(s["body"])[:200000]
        rows.append({
            "standard_id": standard_id,
            "path": path,
            "heading": s["heading"][:500],
            "body_text": body,
            "order_index": idx
        })
        for t in extract_terms(body, path):
            t["standard_id"] = standard_id
            term_rows.append(t)

    if rows:
        sb.table("standard_sections").insert(rows).execute()
    if term_rows:
        sb.table("extracted_terms").insert(term_rows).execute()

    # mark job done
    sb.table("processing_jobs").update({
        "standard_id": standard_id,
        "status": "done",
        "finished_at": datetime.utcnow().isoformat()
    }).eq("upload_session_id", session_id).eq("status", "processing").execute()

    return standard_id


# =========================
# ENDPOINTS
# =========================
@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/debug/env")
def debug_env():
    return {
        "has_url": bool(SUPABASE_URL),
        "has_service_key": bool(SUPABASE_SERVICE_KEY),
        "has_anon_key": bool(SUPABASE_ANON_KEY),
        "buckets": {"raw": RAW_BUCKET, "text": TEXT_BUCKET, "json": JSON_BUCKET}
    }


@app.get("/api/debug/extract")
def debug_extract(file_path: str, pages: int = 15):
    """Preview multiple extract paths for a raw PDF."""
    def _preview(txt: str):
        t = (txt or "")[:800]
        return {"len": len(txt or ""), "preview": t}

    content = download_from_storage(file_path)
    result = {}

    # pdftotext
    try:
        with tempfile.TemporaryDirectory() as td:
            ip, out = os.path.join(td, "in.pdf"), os.path.join(td, "out.txt")
            with open(ip, "wb") as f:
                f.write(content)
            subprocess.run(["pdftotext", "-layout", "-f", "1", "-l", str(pages), ip, out],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            txt = open(out, "r", encoding="utf-8", errors="ignore").read() if os.path.exists(out) else ""
            result["pdftotext"] = _preview(_clean_text(txt))
    except Exception as e:
        result["pdftotext_error"] = str(e)

    # OCRmyPDF → pdftotext
    try:
        with tempfile.TemporaryDirectory() as td:
            ip, op, out = os.path.join(td, "in.pdf"), os.path.join(td, "ocr.pdf"), os.path.join(td, "ocr.txt")
            with open(ip, "wb") as f:
                f.write(content)
            subprocess.run(["ocrmypdf", "--skip-text", "--force-ocr", "--rotate-pages", "--deskew", "--output-type", "pdf", ip, op],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["pdftotext", "-layout", "-f", "1", "-l", str(pages), op, out],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            txt = open(out, "r", encoding="utf-8", errors="ignore").read() if os.path.exists(out) else ""
            result["ocrmypdf_pdftotext"] = _preview(_clean_text(txt))
    except Exception as e:
        result["ocrmypdf_error"] = str(e)

    # image OCR
    try:
        from PIL import Image
        import pytesseract
        with tempfile.TemporaryDirectory() as td:
            ip, base = os.path.join(td, "in.pdf"), os.path.join(td, "page")
            with open(ip, "wb") as f:
                f.write(content)
            subprocess.run(["pdftoppm", "-r", "300", "-f", "1", "-l", "10", "-png", ip, base],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            chunks = []
            for img_path in sorted(glob.glob(base + "-*.png")):
                img = Image.open(img_path)
                chunks.append(pytesseract.image_to_string(img, config="--oem 1 --psm 6 -l eng"))
            txt = "\n".join(chunks)
            result["image_ocr"] = _preview(_clean_text(txt))
    except Exception as e:
        result["image_ocr_error"] = str(e)

    return result


@app.post("/api/upload/start")
def start_upload(req: StartUploadReq):
    sb = supabase_admin()
    org_id = None
    if req.org_name:
        got = sb.table("organizations").select("id").eq("name", req.org_name).limit(1).execute()
        org_id = got.data[0]["id"] if got.data else sb.table("organizations").insert({"name": req.org_name, "type": "Other"}).execute().data[0]["id"]
    session = sb.table("upload_sessions").insert({
        "user_id": None,
        "org_id": org_id,
        "note": req.note or "MVP upload"
    }).execute()
    return {"upload_session_id": session.data[0]["id"]}


@app.post("/api/process/enqueue")
def enqueue(req: EnqueueProcessReq):
    sb = supabase_admin()
    # create processing job
    sb.table("processing_jobs").insert({
        "upload_session_id": req.upload_session_id,
        "status": "processing",
        "started_at": datetime.utcnow().isoformat()
    }).execute()
    try:
        content = download_from_storage(req.file_path)
        if req.force_ocr:
            text = extract_text_from_pdf_ocr_first(content, pages=max(10, min(60, req.pages_sample or 25)))
        else:
            text = extract_text_from_pdf(content, pages=max(10, min(60, req.pages_sample or 15)))
        text = _clean_text(text)
        if not text.strip():
            raise HTTPException(400, "Could not extract text (scanned PDF may need OCR settings).")
        meta = simple_extract_metadata(text, os.path.basename(req.file_path))
        std_id = upsert_standard_and_children(meta, text, req.upload_session_id, file_name=os.path.basename(req.file_path))
        # store raw text + meta JSON (best-effort)
        try:
            text_key = f"{uuid.uuid4()}.txt"
            sb.storage.from_(TEXT_BUCKET).upload(text_key, text.encode("utf-8"))
            sb.storage.from_(JSON_BUCKET).upload(f"{std_id}.json", json.dumps({"meta": meta}).encode("utf-8"))
        except Exception:
            pass
        return {"ok": True, "standard_id": std_id}
    except HTTPException as e:
        sb.table("processing_jobs").update({
            "status": "failed", "finished_at": datetime.utcnow().isoformat(), "log": str(e.detail)
        }).eq("upload_session_id", req.upload_session_id).execute()
        return JSONResponse(status_code=400, content={"error": str(e.detail)})
    except Exception as e:
        sb.table("processing_jobs").update({
            "status": "failed", "finished_at": datetime.utcnow().isoformat(), "log": str(e)
        }).eq("upload_session_id", req.upload_session_id).execute()
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


# =========================
# XREFS
# =========================
def _norm_map(all_codes_rows):
    m = {}
    for row in all_codes_rows or []:
        m[_norm_code(row.get("code") or "")] = row["id"]
    return m


@app.post("/api/xrefs/build")
def build_xrefs(standard_id: str):
    sb = supabase_admin()
    this_std = sb.table("standards").select("id,code").eq("id", standard_id).single().execute().data
    if not this_std:
        raise HTTPException(404, "standard not found")

    all_codes = sb.table("standards").select("id,code").execute().data or []
    code_map = _norm_map(all_codes)

    terms = sb.table("extracted_terms").select("value").eq("standard_id", standard_id).eq("term", "xref").execute().data or []
    created = 0
    for t in terms:
        raw = t.get("value") or ""
        target_id = code_map.get(_norm_code(raw))
        if target_id and target_id != standard_id:
            existing = sb.table("cross_references").select("id").eq("from_standard_id", standard_id).eq("to_standard_id", target_id).limit(1).execute().data
            if not existing:
                sb.table("cross_references").insert({
                    "from_standard_id": standard_id,
                    "to_standard_id": target_id,
                    "evidence_text": raw,
                    "confidence": 0.9
                }).execute()
                created += 1
    return {"ok": True, "created": created}


@app.get("/api/xrefs/of/{standard_id}")
def list_xrefs(standard_id: str):
    sb = supabase_admin()
    links = sb.table("cross_references").select("id,to_standard_id,evidence_text,confidence").eq("from_standard_id", standard_id).execute().data or []
    items = []
    for l in links:
        to = sb.table("standards").select("id,code,title,publisher").eq("id", l["to_standard_id"]).single().execute().data
        if to:
            items.append({
                "xref_id": l["id"],
                "to_standard": to,
                "evidence_text": l.get("evidence_text"),
                "confidence": l.get("confidence", 0.0)
            })
    return {"count": len(items), "items": items}


# =========================
# CONFLICTS
# =========================
@app.post("/api/conflicts/build")
def build_conflicts(req: BuildConflictsReq):
    sb = supabase_admin()
    A = sb.table("standards").select("id,code").eq("id", req.std_a_id).single().execute().data
    B = sb.table("standards").select("id,code").eq("id", req.std_b_id).single().execute().data
    if not A or not B:
        raise HTTPException(404, "one or both standards not found")

    termsA = sb.table("extracted_terms").select("term,value,unit,context_path").eq("standard_id", req.std_a_id).execute().data or []
    termsB = sb.table("extracted_terms").select("term,value,unit,context_path").eq("standard_id", req.std_b_id).execute().data or []

    created, rows = 0, []

    # pressure
    pA, pB = _pick_first(termsA, "pressure"), _pick_first(termsB, "pressure")
    if pA and pB:
        vA = _to_bar(_to_float(pA.get("value")), pA.get("unit"))
        vB = _to_bar(_to_float(pB.get("value")), pB.get("unit"))
        if vA is not None and vB is not None and abs(vA - vB) > 1e-6:
            ratio = abs(vA - vB) / max(vA, vB)
            severity = "high" if ratio >= 0.3 else ("medium" if ratio >= 0.1 else "low")
            rows.append({"std_a_id": req.std_a_id, "std_b_id": req.std_b_id, "parameter": "design_pressure",
                         "value_a": f"{vA:.2f}", "value_b": f"{vB:.2f}", "unit": "bar",
                         "section_a": pA.get("context_path"), "section_b": pB.get("context_path"),
                         "rule_id": None, "severity": severity})

    # temperature
    tA, tB = _pick_first(termsA, "temperature"), _pick_first(termsB, "temperature")
    if tA and tB:
        vA = _to_celsius(_to_float(tA.get("value")), tA.get("unit"))
        vB = _to_celsius(_to_float(tB.get("value")), tB.get("unit"))
        if vA is not None and vB is not None and abs(vA - vB) >= 5.0:
            severity = "medium" if abs(vA - vB) < 30 else "high"
            rows.append({"std_a_id": req.std_a_id, "std_b_id": req.std_b_id, "parameter": "design_temperature",
                         "value_a": f"{vA:.1f}", "value_b": f"{vB:.1f}", "unit": "C",
                         "section_a": tA.get("context_path"), "section_b": tB.get("context_path"),
                         "rule_id": None, "severity": severity})

    # class
    cA, cB = _pick_first(termsA, "class"), _pick_first(termsB, "class")
    if cA and cB:
        nA, nB = _to_float(cA.get("value")), _to_float(cB.get("value"))
        if nA is not None and nB is not None and int(nA) != int(nB):
            rows.append({"std_a_id": req.std_a_id, "std_b_id": req.std_b_id, "parameter": "pressure_class",
                         "value_a": str(int(nA)), "value_b": str(int(nB)), "unit": None,
                         "section_a": cA.get("context_path"), "section_b": cB.get("context_path"),
                         "rule_id": None, "severity": "high"})

    # material
    mA, mB = _pick_first(termsA, "material"), _pick_first(termsB, "material")
    if mA and mB and (mA.get("value") or "").strip().upper() != (mB.get("value") or "").strip().upper():
        rows.append({"std_a_id": req.std_a_id, "std_b_id": req.std_b_id, "parameter": "material",
                     "value_a": mA.get("value"), "value_b": mB.get("value"), "unit": None,
                     "section_a": mA.get("context_path"), "section_b": mB.get("context_path"),
                     "rule_id": None, "severity": "low"})

    if rows:
        sb.table("conflicts").insert(rows).execute()
        created = len(rows)

    return {"ok": True, "created": created, "diffs": rows}


@app.get("/api/conflicts/of/{standard_id}")
def list_conflicts_for_standard(standard_id: str):
    sb = supabase_admin()
    confA = sb.table("conflicts").select("*").eq("std_a_id", standard_id).execute().data or []
    confB = sb.table("conflicts").select("*").eq("std_b_id", standard_id).execute().data or []
    conflicts = confA + confB

    def _std_code(sid: str):
        try:
            row = sb.table("standards").select("id,code,title").eq("id", sid).single().execute().data
            return {"id": sid, "code": (row or {}).get("code"), "title": (row or {}).get("title")}
        except Exception:
            return {"id": sid, "code": None, "title": None}

    items = []
    for c in conflicts:
        items.append({
            "parameter": c.get("parameter"),
            "severity": c.get("severity"),
            "unit": c.get("unit"),
            "a": {"standard": _std_code(c.get("std_a_id")), "value": c.get("value_a"), "section": c.get("section_a")},
            "b": {"standard": _std_code(c.get("std_b_id")), "value": c.get("value_b"), "section": c.get("section_b")},
        })
    return {"count": len(items), "items": items}


@app.get("/api/conflicts/csv/{standard_id}")
def conflicts_csv(standard_id: str):
    sb = supabase_admin()
    confA = sb.table("conflicts").select("*").eq("std_a_id", standard_id).execute().data or []
    confB = sb.table("conflicts").select("*").eq("std_b_id", standard_id).execute().data or []
    conflicts = confA + confB

    def _std(sid):
        try:
            r = sb.table("standards").select("id,code,title").eq("id", sid).single().execute().data
            return (r or {}).get("code") or sid, (r or {}).get("title") or ""
        except Exception:
            return sid, ""

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["parameter","severity","unit","A_standard_code","A_standard_title","A_value","A_section","B_standard_code","B_standard_title","B_value","B_section"])
    for c in conflicts:
        acode, atitle = _std(c.get("std_a_id"))
        bcode, btitle = _std(c.get("std_b_id"))
        w.writerow([c.get("parameter"), c.get("severity"), c.get("unit"), acode, atitle, c.get("value_a"), c.get("section_a"), bcode, btitle, c.get("value_b"), c.get("section_b")])
    csv_data = buf.getvalue()
    return Response(content=csv_data, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="conflicts_{standard_id}.csv"'})


def _wrap_lines(text: str, max_width: float, font_name="Helvetica", font_size=11):
    words = (text or "").split()
    lines, line = [], ""
    for w in words:
        trial = (line + " " + w).strip()
        if stringWidth(trial, font_name, font_size) <= max_width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


@app.get("/api/conflicts/pdf/{standard_id}")
def conflicts_pdf(standard_id: str):
    sb = supabase_admin()
    std = sb.table("standards").select("id,code,title").eq("id", standard_id).single().execute().data
    if not std:
        raise HTTPException(404, "standard not found")

    confA = sb.table("conflicts").select("*").eq("std_a_id", standard_id).execute().data or []
    confB = sb.table("conflicts").select("*").eq("std_b_id", standard_id).execute().data or []
    conflicts = confA + confB

    def _std_meta(sid: str):
        try:
            r = sb.table("standards").select("id,code,title").eq("id", sid).single().execute().data
            return (r or {}).get("code") or sid, (r or {}).get("title") or ""
        except Exception:
            return sid, ""

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    lm, rm, tm, bm = 20*mm, 20*mm, 20*mm, 20*mm
    maxw = W - lm - rm
    y = H - tm

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(lm, y, "Conflict Summary")
    y -= 14
    c.setFont("Helvetica", 10)
    c.drawString(lm, y, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    y -= 18

    # Standard info
    c.setFont("Helvetica-Bold", 12)
    c.drawString(lm, y, f"Standard: {std.get('code') or standard_id}")
    y -= 14
    c.setFont("Helvetica", 11)
    for line in _wrap_lines(std.get("title") or "", maxw):
        c.drawString(lm, y, line)
        y -= 13
    y -= 6

    # Summary counts
    c.setFont("Helvetica-Bold", 12)
    c.drawString(lm, y, f"Total conflicts: {len(conflicts)}")
    y -= 18

    # List conflicts
    c.setFont("Helvetica", 11)
    for i, cf in enumerate(conflicts, start=1):
        if y < bm + 50:  # page break
            c.showPage()
            y = H - tm
            c.setFont("Helvetica", 11)

        acode, _ = _std_meta(cf.get("std_a_id"))
        bcode, _ = _std_meta(cf.get("std_b_id"))
        param = cf.get("parameter") or "-"
        sev = (cf.get("severity") or "").upper()
        unit = cf.get("unit") or ""

        c.setFont("Helvetica-Bold", 11)
        c.drawString(lm, y, f"{i}. {param}  [severity: {sev}]")
        y -= 13
        c.setFont("Helvetica", 11)

        a_line = f"A: {acode}  →  value: {cf.get('value_a')}{(' ' + unit) if unit else ''}  (section {cf.get('section_a') or '-'})"
        for ln in _wrap_lines(a_line, maxw):
            c.drawString(lm, y, ln)
            y -= 13

        b_line = f"B: {bcode}  →  value: {cf.get('value_b')}{(' ' + unit) if unit else ''}  (section {cf.get('section_b') or '-'})"
        for ln in _wrap_lines(b_line, maxw):
            c.drawString(lm, y, ln)
            y -= 13

        y -= 6

    c.showPage()
    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="conflicts_{standard_id}.pdf"'}
    )


# =========================
# DASHBOARD
# =========================
@app.get("/api/dashboard")
def dashboard():
    sb = supabase_admin()
    total = len(sb.table("standards").select("id").execute().data or [])
    outdated = len(sb.table("standards").select("id").eq("status", "outdated").execute().data or [])
    conflicts = len(sb.table("conflicts").select("id").execute().data or [])
    xrefs = len(sb.table("extracted_terms").select("id").eq("term", "xref").execute().data or [])
    return {"total_standards": total, "outdated": outdated, "conflicts": conflicts, "xref_mentions": xrefs}
