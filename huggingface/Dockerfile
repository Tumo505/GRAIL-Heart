FROM python:3.11-slim

WORKDIR /app

# System deps for scanpy / h5py
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libhdf5-dev git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (two stages: torch first, then the rest)
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        torch>=2.0.0 && \
    pip install --no-cache-dir -r requirements.txt

# Streamlit config (upload limit, XSRF/CORS disabled for HF Spaces proxy)
RUN mkdir -p .streamlit
RUN printf '[server]\nmaxUploadSize = 10240\nmaxMessageSize = 10240\nenableXsrfProtection = false\nenableCORS = false\n' > .streamlit/config.toml

# Copy application code and assets
COPY app.py .
COPY model/ model/
COPY demo_data/ demo_data/
COPY src/ src/

# HF Spaces expects port 7860
EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=7860", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
