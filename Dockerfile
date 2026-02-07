FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for Docker cache efficiency)
COPY pyproject.toml README.md ./

# Install PyTorch CPU (smaller image for deployment)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir torch-geometric && \
    pip install --no-cache-dir streamlit plotly scanpy anndata scikit-learn

# Copy source code and app
COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/
COPY .streamlit/ ./.streamlit/

# Create data dir — the L-R database cache is optional;
# the app downloads it from OmniPath on first run if missing.
RUN mkdir -p ./data
# If you have the cache on the VM, mount it via docker-compose volumes.

# Checkpoint is mounted as a volume (see docker-compose.yml)
# To bake it in instead, uncomment:
# COPY outputs/checkpoints/best.pt ./outputs/checkpoints/best.pt

# Install the package
RUN pip install --no-cache-dir -e .

# Set environment
ENV PYTHONPATH=/app/src
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=5000

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
