FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project requirements (top-level) so core deps like pandas are installed
COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

# Copy the repository into the image (backend expects repo root relative imports)
COPY . /app

EXPOSE 5001

# Use Gunicorn with the Flask factory `create_app()`
# Single worker is required: BacktestStore/DiscoveryStore are in-memory singletons;
# multiple workers would each have their own copy and GET endpoints would return
# 404 after data is written in a different worker process.
CMD ["gunicorn", "dashboard.backend.app:create_app()", "-b", "0.0.0.0:5001", "--workers", "1", "--threads", "8", "--timeout", "300"]
