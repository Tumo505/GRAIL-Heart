FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for Docker cache efficiency)
COPY pyproject.toml README.md MANIFEST.in ./

# Install PyTorch CPU (smaller image; use CUDA base image for GPU)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir torch-geometric && \
    pip install --no-cache-dir streamlit plotly

# Copy source code
COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/
COPY data/lr_database_cache.csv ./data/lr_database_cache.csv

# Copy model checkpoint (if you want it baked in; or mount as volume)
# COPY outputs/final_model/ ./outputs/final_model/

# Install the package
RUN pip install --no-cache-dir -e .

# Set environment
ENV PYTHONPATH=/app/src
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
