FROM python:3.11-slim

# System deps: Tesseract + Poppler + Ghostscript + OCRmyPDF helpers
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    ghostscript \
    poppler-utils \
    qpdf \
    pngquant \
    unpaper \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

ENV PYTHONUNBUFFERED=1
# Use Render's provided $PORT if present
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]

