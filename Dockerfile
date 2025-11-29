FROM python:3.12-slim-bookworm as builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y git

ENV HOST=0.0.0.0
ENV LISTEN_PORT 8080
EXPOSE 8080

WORKDIR /app

COPY pyproject.toml ./

RUN uv venv .venv
RUN /bin/bash -c "source .venv/bin/activate && uv pip compile pyproject.toml > requirements.txt && uv pip install -r requirements.txt"

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --from=builder /app/requirements.txt ./requirements.txt

COPY ./src ./src

CMD ["streamlit", "run", "src/main.py", "--server.port", "8080"]
