FROM python:3.11-slim

WORKDIR /app

# System deps for scanpy / h5py
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.0.1 && \
    pip install --no-cache-dir -r requirements.txt

# Streamlit config (upload limit, XSRF/CORS disabled for HF Spaces proxy)
RUN mkdir -p .streamlit
RUN printf '[server]\nmaxUploadSize = 10240\nmaxMessageSize = 10240\nenableXsrfProtection = false\nenableCORS = false\n' > .streamlit/config.toml

# Copy all application code and assets present in this repo
COPY . .

# HF Spaces expects port 7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health')" || exit 1

ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py", \
    "--server.port=7860", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--logger.level=warning", \
    "--browser.gatherUsageStats=false"]
