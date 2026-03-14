FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Copy backend runtime requirements only (avoid dev/test/CLI deps in image)
COPY dashboard/backend/requirements.runtime.txt ./requirements.runtime.txt

# Copy repository files early so any build-time packaging that reads repo files
# (e.g. README.md) won't fail while building wheels.
COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefer-binary -r requirements.runtime.txt

# Copy the repository into the image (backend expects repo root relative imports)

EXPOSE 5001

# Use Gunicorn with the Flask factory `create_app()`
# Single worker is required: BacktestStore/DiscoveryStore are in-memory singletons;
# multiple workers would each have their own copy and GET endpoints would return
# 404 after data is written in a different worker process.
CMD ["gunicorn", "dashboard.backend.app:create_app()", "-b", "0.0.0.0:5001", "--workers", "1", "--threads", "8", "--timeout", "300"]
