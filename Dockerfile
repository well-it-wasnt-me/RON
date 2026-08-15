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

EXPOSE 8000
VOLUME ["/data"]

CMD ["python", "-m", "robot"]
