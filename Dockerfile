FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && grep -v '^torch' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt

COPY . .
RUN pip install --no-deps -e . && python examples/generate_datasets.py

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1
CMD ["agentml", "serve", "--host", "0.0.0.0", "--port", "8000"]
