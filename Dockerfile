ARG PYTHON_VERSION=3.11.4
FROM python:${PYTHON_VERSION}-slim as base

COPY --from=ghcr.io/astral-sh/uv:0.5.27 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FFMPEG=/usr/local/bin/ffmpeg
ENV FFPROBE=/usr/local/bin/ffprobe
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV CUDA_VISIBLE_DEVICES=""

RUN mkdir -p /cache/uv
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/cache/uv

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
    && tar -xvf ffmpeg-release-amd64-static.tar.xz \
    && mv ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ffmpeg \
    && mv ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ffprobe \
    && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
    && rm -rf ffmpeg-*-amd64-static*

ENV FFMPEG=/usr/local/bin/ffmpeg
ENV FFPROBE=/usr/local/bin/ffprobe

WORKDIR /app

COPY . /app


RUN --mount=type=cache,target=/cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project


RUN --mount=type=cache,target=/cache/uv \
    uv sync --frozen

RUN chown -R appuser:appuser /cache/uv /app

USER appuser

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]