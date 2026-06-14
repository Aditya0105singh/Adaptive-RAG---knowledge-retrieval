FROM python:3.12-slim

WORKDIR /app

# Build deps for sentence-transformers native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

# Pre-download the embedding model at build time so the first query is instant.
# Remove this RUN line if you need an offline / air-gapped build.
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

# Ensure Streamlit never shows the secrets banner in container environments
RUN mkdir -p .streamlit && touch .streamlit/secrets.toml

# Default: start the FastAPI backend.
# Override with --command for the Streamlit frontend (see docker-compose.yml).
CMD ["python", "main.py"]
