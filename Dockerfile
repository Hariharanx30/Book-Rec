FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

ENV PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model at image build-time to avoid runtime cold-start downloads
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
