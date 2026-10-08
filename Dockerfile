FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    WEB_CONCURRENCY=2

WORKDIR /app

# CPU-only torch keeps the image a few GB smaller. Remove the first line if you have a GPU.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY app app
COPY scripts scripts

# run as a normal user, not root
RUN useradd --create-home appuser && mkdir -p /app/data /models && chown -R appuser /app /models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# WEB_CONCURRENCY sets the number of workers. Each worker loads the model and index into memory.
CMD ["gunicorn", "app.main:app_factory()", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--timeout", "120"]
