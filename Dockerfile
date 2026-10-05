# Reproducible CPU environment for training, evaluation, inference and tests.
# The interactive viewer needs a display and is meant to run on the host;
# inside the container use `--screenshot` for off-screen rendering.
#
#   docker build -t medimg3d .
#   docker run --rm medimg3d                                   # runs the test suite
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" \
#       medimg3d python -m src.training.train --config configs/train.yaml

FROM python:3.12-slim

# libGL/libXrender: required by VTK (PyVista) even for off-screen use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Dependencies first (cached layer), then the code.
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]"
COPY configs ./configs
COPY tests ./tests

ENV CI=true
CMD ["python", "-m", "pytest", "-q"]
