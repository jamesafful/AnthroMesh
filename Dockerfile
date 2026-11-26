FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY schemas ./schemas
COPY examples ./examples

RUN pip install --no-cache-dir -e ".[protocols]"

EXPOSE 8081 8082

CMD ["anthromesh", "serve-a2a", "--port", "8082"]
