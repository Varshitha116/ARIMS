# ============================================================
# ARIMS — Multi-stage Docker Build
# ============================================================
# Stage 1: Install dependencies in a builder image
# Stage 2: Copy into a slim runtime image
# ============================================================

FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim AS runtime

# System metadata
LABEL maintainer="ARIMS Team — Osmania University"
LABEL description="Agentic AI Road Infrastructure Maintenance System"

# Install runtime OS deps (OpenCV needs libgl)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Application code
WORKDIR /app
COPY . .

# Create upload directory
RUN mkdir -p /app/uploads

# Non-root user for security
RUN useradd -m arims && chown -R arims:arims /app
USER arims

# Expose ports
EXPOSE 8000 8501

# Default: start FastAPI
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
