# ProtoSynth Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy source code
COPY . /app

# Install ProtoSynth and dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Run tests and demo evolution by default
CMD ["bash", "-lc", "pytest -q && protosynth-evolve --env markov_k2 --gens 50 --k 2"]
