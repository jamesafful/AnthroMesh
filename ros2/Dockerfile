FROM python:3.12-slim AS dependencies
RUN pip install --only-binary=:all: --target /anthromesh-deps "pydantic>=2" "numpy>=1.24"

FROM ros:jazzy-ros-base
WORKDIR /app
COPY --from=dependencies /anthromesh-deps /anthromesh-deps
COPY src /app/src
COPY ros2/smoke.py /app/ros2_smoke.py
ENV PYTHONPATH=/app/src:/anthromesh-deps
CMD ["python3", "-m", "anthromesh.cli", "--help"]
