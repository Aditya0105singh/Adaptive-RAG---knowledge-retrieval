FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

COPY . .

# Ensure Streamlit never shows the secrets banner in container environments
RUN mkdir -p .streamlit && touch .streamlit/secrets.toml

# Default: start the FastAPI backend.
CMD ["python", "main.py"]
