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

def simple_extract_metadata(text: str, file_name: Optional[str] = None):
    # Very lightweight heuristics; improved to handle file-name fallbacks like "API Bull 6J.pdf"
    first_line = text.strip().splitlines()[0] if text.strip().splitlines() else ""
    title = first_line[:200]

    # Search targets (API / ISO / ASME / IEC) in text first
    patterns = [
        r'\b(API\s*(?:Spec(?:ification)?|Std(?:ard)?|RP|Recommended\s*Practice|Bulletin|Bull|MPMS)?\s*[A-Z]?\d+[A-Z]?)\b',
        r'\b(ISO\s*\d+(?:-\d+)*)\b',
        r'\b(ASME\s*[A-Z]?\d[\d\.]*)\b',
        r'\b(IEC\s*\d+)\b',
    ]
    text_window = text[:4000]
    code = None
    import re as _re
    for pat in patterns:
        m = _re.search(pat, text_window, _re.I)
        if m:
            code = m.group(1).upper()
            break

    # Fallback: try to parse code from file name (e.g., "API Bull 6J.pdf")
    if not code and file_name:
        base = os.path.splitext(os.path.basename(file_name))[0].replace("_", " ")
        for pat in patterns:
            m = _re.search(pat, base, _re.I)
            if m:
                code = m.group(1).upper()
                break
        if not code:
            # last resort: use the filename (uppercased) as code
            code = base.upper()[:100]

    # revision date (very rough)
    dm = _re.search(r'(20\d{2}|19\d{2})[-/\.]([01]?\d)[-/\.]([0-3]?\d)', text_window)
    rev_date = None
    if dm:
        y, mth, d = dm.group(1), dm.group(2), dm.group(3)
        try:
            rev_date = datetime(int(y), int(mth), int(d)).date().isoformat()
        except:
            pass

    # publisher guess from code
    pub = None
    if code:
        if code.startswith("API"): pub = "API"
        elif code.startswith("ISO"): pub = "ISO"
        elif code.startswith("ASME"): pub = "ASME"
        elif code.startswith("IEC"): pub = "IEC"

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
    "xref": _re.compile(r'\b(API\s*(?:Spec(?:ification)?|Std(?:ard)?|RP|Recommended\s*Practice|Bulletin|Bull|MPMS)?\s*[A-Z]?\d+[A-Z]?)\b|\b(ISO\s*\d+(?:-\d+)*)\b|\b(ASME\s*[A-Z]?\d[\d\.]*)\b|\b(IEC\s*\d+)\b', _re.I),
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

# --- add this helper ABOVE extract_text_from_pdf ---
def ocr_pdf_with_poppler(content: bytes, max_pages: int = 5) -> str:
    """
    Renders the first N pages of a PDF to images (via pdftoppm) and OCRs them with Tesseract.
    Works for scanned PDFs where pdfminer finds no text.
    """
    import tempfile, subprocess, glob
    from PIL import Image
    import pytesseract

    with tempfile.TemporaryDirectory() as td:
        pdf_path = os.path.join(td, "in.pdf")
        with open(pdf_path, "wb") as f:
            f.write(content)

        # render first max_pages to PNGs: page-1.png, page-2.png, ...
        out_base = os.path.join(td, "page")
        cmd = ["pdftoppm", "-f", "1", "-l", str(max_pages), "-png", pdf_path, out_base]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        texts = []
        for img_path in sorted(glob.glob(out_base + "-*.png")):
            img = Image.open(img_path)
            texts.append(pytesseract.image_to_string(img))
        return "\n".join(texts)

# --- helpers: text quality check + OCRmyPDF pass ---
def _text_useful(t: str) -> bool:
    import re as _re
    # accept shorter early pages too
    return len(t.strip()) > 200 and len(_re.findall(r'[A-Za-z]{3,}', t)) > 20

def _ocrmypdf(content: bytes) -> bytes:
    """Run OCRmyPDF to add a text layer to scanned PDFs."""
    import tempfile, subprocess, os
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.pdf")
        out_path = os.path.join(td, "out.pdf")
        with open(in_path, "wb") as f:
            f.write(content)
        # --skip-text: only OCR pages without text; --force-ocr as a safety
        cmd = [
            "ocrmypdf", "--skip-text", "--force-ocr",
            "--rotate-pages", "--deskew",
            "--output-type", "pdf",
            in_path, out_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(out_path, "rb") as f:
            return f.read()

# --- replace your entire extract_text_from_pdf with this ---
def extract_text_from_pdf(content: bytes) -> str:
    """
    Order:
      1) pdftotext (first 5 pages)
      2) pdfminer
      3) OCRmyPDF → pdftotext and pdfminer
      4) pdftoppm(300dpi)+tesseract OCR (first 5 pages)
      5) last resort: single-image OCR
    """
    import tempfile, subprocess, os, io, glob
    from pdfminer_high_level_import_fix import extract_text as _pm_extract if False else None
    from pdfminer.high_level import extract_text as _pm_extract

    # 1) pdftotext (Poppler)
    try:
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "in.pdf")
            out_txt = os.path.join(td, "out.txt")
            with open(in_path, "wb") as f:
                f.write(content)
            subprocess.run(["pdftotext", "-layout", "-f", "1", "-l", "5", in_path, out_txt],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if os.path.exists(out_txt):
                with open(out_txt, "r", encoding="utf-8", errors="ignore") as f:
                    t = f.read()
                if _text_useful(t):
                    return t
    except Exception:
        pass

    # 2) pdfminer (Python)
    try:
        with io.BytesIO(content) as fh:
            t = _pm_extract(fh) or ""
            if _text_useful(t):
                return t
    except Exception:
        pass

    # 3) OCRmyPDF → pdftotext (+ pdfminer as backup)
    try:
        ocr_bytes = _ocrmypdf(content)
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "ocr.pdf")
            out_txt = os.path.join(td, "out.txt")
            with open(in_path, "wb") as f:
                f.write(ocr_bytes)
            subprocess.run(["pdftotext", "-layout", "-f", "1", "-l", "5", in_path, out_txt],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if os.path.exists(out_txt):
                with open(out_txt, "r", encoding="utf-8", errors="ignore") as f:
                    t = f.read()
                if _text_useful(t):
                    return t
            # try pdfminer on the OCR'd PDF
            with open(in_path, "rb") as fh:
                t = _pm_extract(fh) or ""
                if t.strip():
                    return t
    except Exception:
        pass

    # 4) High-DPI image OCR (first 5 pages)
    try:
        from PIL import Image
        import pytesseract
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "in.pdf")
            with open(in_path, "wb") as f:
                f.write(content)
            out_base = os.path.join(td, "page")
            subprocess.run(["pdftoppm", "-r", "300", "-f", "1", "-l", "5", "-png", in_path, out_base],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            texts = []
            for img_path in sorted(glob.glob(out_base + "-*.png")):
                img = Image.open(img_path)
                texts.append(pytesseract.image_to_string(img, config="--oem 1 --psm 6 -l eng"))
            t = "\n".join(texts)
            if t.strip():
                return t
    except Exception:
        pass

    # 5) last resort single-image OCR
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(io.BytesIO(content))
        return pytesseract.image_to_string(img, config="--oem 1 --psm 6 -l eng")
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
        meta = simple_extract_metadata(text, os.path.basename(req.file_path))
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
@app.get("/api/dashboard")
def dashboard():
    sb = supabase_admin()
    # simple counts for MVP (no fancy SQL)
    total = len(sb.table("standards").select("id").execute().data or [])
    outdated = len(sb.table("standards").select("id").eq("status", "outdated").execute().data or [])
    conflicts = len(sb.table("conflicts").select("id").execute().data or [])
    xrefs = len(sb.table("extracted_terms").select("id").eq("term", "xref").execute().data or [])
    return {
        "total_standards": total,
        "outdated": outdated,
        "conflicts": conflicts,
        "xref_mentions": xrefs
    }
from typing import Dict

def _norm_code(s: str) -> str:
    if not s:
        return ""
    s = s.upper()
    # Remove descriptor words so API BULLETIN/BULL/SPEC/STD/RP/RECOMMENDED PRACTICE all collapse
    for w in ["BULLETIN", "BULL", "SPECIFICATION", "SPEC", "STANDARD", "STD",
              "RECOMMENDED PRACTICE", "RECOMMENDEDPRACTICE", "RP", "MPMS"]:
        s = s.replace(w, "")
    import re
    s = re.sub(r"[^A-Z0-9\.]", "", s)  # drop spaces and punctuation (keep dot)
    return s

@app.post("/api/xrefs/build")
def build_xrefs(standard_id: str):
    """
    Build cross-references for a single standard:
    - looks at extracted_terms where term='xref'
    - matches term value (e.g., 'API 6D', 'ISO 15156') to other standards' codes
    - writes rows into cross_references
    """
    sb = supabase_admin()

    # 1) fetch this standard and all standards (id, code)
    this_std = sb.table("standards").select("id,code").eq("id", standard_id).single().execute().data
    if not this_std:
        raise HTTPException(404, "standard not found")

    all_codes = sb.table("standards").select("id,code").execute().data or []
    code_map: Dict[str, str] = {}
    for row in all_codes:
        code_map[_norm_code(row.get("code") or "")] = row["id"]

    # 2) fetch xref term values for this standard
    terms = sb.table("extracted_terms").select("value").eq("standard_id", standard_id).eq("term", "xref").execute().data or []

    created = 0
    for t in terms:
        raw = t.get("value") or ""
        target_id = code_map.get(_norm_code(raw))
        if target_id and target_id != standard_id:
            # check if already exists
            existing = sb.table("cross_references").select("id")\
                .eq("from_standard_id", standard_id)\
                .eq("to_standard_id", target_id)\
                .limit(1).execute().data
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
    # fetch links where this standard points to others
    links = sb.table("cross_references")\
        .select("id,to_standard_id,evidence_text,confidence")\
        .eq("from_standard_id", standard_id).execute().data or []
    items = []
    for l in links:
        to = sb.table("standards").select("id,code,title,publisher")\
            .eq("id", l["to_standard_id"]).single().execute().data
        if to:
            items.append({
                "xref_id": l["id"],
                "to_standard": to,
                "evidence_text": l.get("evidence_text"),
                "confidence": l.get("confidence", 0.0)
            })
    return {"count": len(items), "items": items}
from pydantic import BaseModel

# ---------- helpers for conflict detection ----------
_num = _re.compile(r"[-+]?\d+(?:\.\d+)?")

def _to_float(s: str) -> float | None:
    if not s:
        return None
    m = _num.search(s)
    return float(m.group(0)) if m else None

def _to_bar(value: float | None, unit: str | None) -> float | None:
    if value is None:
        return None
    u = (unit or "").lower()
    if u in ("bar",):
        return value
    if u in ("psi",):
        return value * 0.0689476
    if u in ("kpa",):
        return value / 100.0
    if u in ("mpa",):
        return value * 10.0
    # unknown unit → return as-is
    return value

def _to_celsius(value: float | None, unit: str | None) -> float | None:
    if value is None:
        return None
    u = (unit or "").lower().replace("°", "")
    if u in ("c", "°c"):
        return value
    if u in ("f", "°f"):
        return (value - 32.0) * 5.0 / 9.0
    return value

def _pick_first(terms: list[dict], name: str) -> dict | None:
    for t in terms:
        if t.get("term") == name:
            return t
    return None

class BuildConflictsReq(BaseModel):
    std_a_id: str
    std_b_id: str

@app.post("/api/conflicts/build")
def build_conflicts(req: BuildConflictsReq):
    """
    Compare two standards on basic parameters and write differences to `conflicts`:
      - pressure (numeric, normalized to bar)
      - temperature (numeric, normalized to °C)
      - class (Class 150/300/etc.)
      - material (string)
    """
    sb = supabase_admin()

    # fetch A & B
    A = sb.table("standards").select("id,code").eq("id", req.std_a_id).single().execute().data
    B = sb.table("standards").select("id,code").eq("id", req.std_b_id).single().execute().data
    if not A or not B:
        raise HTTPException(404, "one or both standards not found")

    termsA = sb.table("extracted_terms").select("term,value,unit,context_path")\
        .eq("standard_id", req.std_a_id).execute().data or []
    termsB = sb.table("extracted_terms").select("term,value,unit,context_path")\
        .eq("standard_id", req.std_b_id).execute().data or []

    created = 0
    rows = []

    # PRESSURE
    pA = _pick_first(termsA, "pressure")
    pB = _pick_first(termsB, "pressure")
    if pA and pB:
        vA = _to_bar(_to_float(pA.get("value")), pA.get("unit"))
        vB = _to_bar(_to_float(pB.get("value")), pB.get("unit"))
        if vA is not None and vB is not None and abs(vA - vB) > 1e-6:
            ratio = abs(vA - vB) / max(vA, vB)
            severity = "high" if ratio >= 0.3 else ("medium" if ratio >= 0.1 else "low")
            rows.append({
                "std_a_id": req.std_a_id, "std_b_id": req.std_b_id,
                "parameter": "design_pressure",
                "value_a": f"{vA:.2f}", "value_b": f"{vB:.2f}", "unit": "bar",
                "section_a": pA.get("context_path"), "section_b": pB.get("context_path"),
                "rule_id": None, "severity": severity
            })

    # TEMPERATURE
    tA = _pick_first(termsA, "temperature")
    tB = _pick_first(termsB, "temperature")
    if tA and tB:
        vA = _to_celsius(_to_float(tA.get("value")), tA.get("unit"))
        vB = _to_celsius(_to_float(tB.get("value")), tB.get("unit"))
        if vA is not None and vB is not None and abs(vA - vB) >= 5.0:
            severity = "medium" if abs(vA - vB) < 30 else "high"
            rows.append({
                "std_a_id": req.std_a_id, "std_b_id": req.std_b_id,
                "parameter": "design_temperature",
                "value_a": f"{vA:.1f}", "value_b": f"{vB:.1f}", "unit": "C",
                "section_a": tA.get("context_path"), "section_b": tB.get("context_path"),
                "rule_id": None, "severity": severity
            })

    # CLASS
    cA = _pick_first(termsA, "class")
    cB = _pick_first(termsB, "class")
    if cA and cB:
        nA = _to_float(cA.get("value"))
        nB = _to_float(cB.get("value"))
        if nA is not None and nB is not None and int(nA) != int(nB):
            rows.append({
                "std_a_id": req.std_a_id, "std_b_id": req.std_b_id,
                "parameter": "pressure_class",
                "value_a": str(int(nA)), "value_b": str(int(nB)), "unit": None,
                "section_a": cA.get("context_path"), "section_b": cB.get("context_path"),
                "rule_id": None, "severity": "high"
            })

    # MATERIAL
    mA = _pick_first(termsA, "material")
    mB = _pick_first(termsB, "material")
    if mA and mB and (mA.get("value") or "").strip().upper() != (mB.get("value") or "").strip().upper():
        rows.append({
            "std_a_id": req.std_a_id, "std_b_id": req.std_b_id,
            "parameter": "material",
            "value_a": mA.get("value"), "value_b": mB.get("value"), "unit": None,
            "section_a": mA.get("context_path"), "section_b": mB.get("context_path"),
            "rule_id": None, "severity": "low"
        })

    # write conflicts
    if rows:
        sb.table("conflicts").insert(rows).execute()
        created = len(rows)

    return {"ok": True, "created": created, "diffs": rows}

@app.get("/api/conflicts/of/{standard_id}")
def list_conflicts_for_standard(standard_id: str):
    sb = supabase_admin()
    # gather conflicts where this standard is on either side (A or B)
    confA = sb.table("conflicts").select("*").eq("std_a_id", standard_id).execute().data or []
    confB = sb.table("conflicts").select("*").eq("std_b_id", standard_id).execute().data or []
    conflicts = confA + confB

    # hydrate codes for readability
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

from fastapi import Response
import csv, io

@app.get("/api/conflicts/csv/{standard_id}")
def conflicts_csv(standard_id: str):
    sb = supabase_admin()
    confA = sb.table("conflicts").select("*").eq("std_a_id", standard_id).execute().data or []
    confB = sb.table("conflicts").select("*").eq("std_b_id", standard_id).execute().data or []
    conflicts = confA + confB

    # lookup map for codes/titles
    def _std(sid):
        try:
            r = sb.table("standards").select("id,code,title").eq("id", sid).single().execute().data
            return (r or {}).get("code") or sid, (r or {}).get("title") or ""
        except Exception:
            return sid, ""

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["parameter","severity","unit",
                "A_standard_code","A_standard_title","A_value","A_section",
                "B_standard_code","B_standard_title","B_value","B_section"])
    for c in conflicts:
        acode, atitle = _std(c.get("std_a_id"))
        bcode, btitle = _std(c.get("std_b_id"))
        w.writerow([
            c.get("parameter"), c.get("severity"), c.get("unit"),
            acode, atitle, c.get("value_a"), c.get("section_a"),
            bcode, btitle, c.get("value_b"), c.get("section_b")
        ])
    csv_data = buf.getvalue()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="conflicts_{standard_id}.csv"'}
    )

from fastapi import Response
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
import io
from datetime import datetime as _dt

def _wrap_lines(text: str, max_width: float, font_name="Helvetica", font_size=11):
    """Simple word-wrap for reportlab drawString."""
    words = text.split()
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
    # fetch base standard
    std = sb.table("standards").select("id,code,title").eq("id", standard_id).single().execute().data
    if not std:
        raise HTTPException(404, "standard not found")

    # fetch conflicts where this standard is A or B
    confA = sb.table("conflicts").select("*").eq("std_a_id", standard_id).execute().data or []
    confB = sb.table("conflicts").select("*").eq("std_b_id", standard_id).execute().data or []
    conflicts = confA + confB

    # helper to get code/title for sides
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
    c.drawString(lm, y, f"Generated: {_dt.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
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
        # Page break if needed
        if y < bm + 50:
            c.showPage()
            y = H - tm
            c.setFont("Helvetica", 11)

        acode, atitle = _std_meta(cf.get("std_a_id"))
        bcode, btitle = _std_meta(cf.get("std_b_id"))
        param = cf.get("parameter") or "-"
        sev = (cf.get("severity") or "").upper()
        unit = cf.get("unit") or ""

        # Line 1: heading
        line1 = f"{i}. {param}  [severity: {sev}]"
        c.setFont("Helvetica-Bold", 11)
        c.drawString(lm, y, line1)
        y -= 13
        c.setFont("Helvetica", 11)

        # Line 2: A
        a_line = f"A: {acode}  →  value: {cf.get('value_a')}{(' ' + unit) if unit else ''}  (section {cf.get('section_a') or '-'})"
        for ln in _wrap_lines(a_line, maxw):
            c.drawString(lm, y, ln)
            y -= 13

        # Line 3: B
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
