FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DESKBOT_HARDWARE=mock

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY assets ./assets
COPY web ./web
COPY config ./config

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .[api]

# Create a non-root user and switch to it.
RUN groupadd --system deskbot && useradd --system --gid deskbot --home-dir /data deskbot
USER deskbot

EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3)" || exit 1

CMD ["python", "-m", "robot"]
