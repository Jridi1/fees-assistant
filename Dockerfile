# Image for Hugging Face Spaces (Docker SDK) and for running anywhere else.
#   docker build -t fees-assistant .
#   docker run --rm -p 7860:7860 -e GROQ_API_KEY=... fees-assistant
FROM python:3.12-slim

# Spaces run containers as user 1000: create it first, and give it every file it must write.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ANONYMIZED_TELEMETRY=False \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR $HOME/app

# CPU-only PyTorch first. The default wheel bundles ~2 GB of CUDA libraries a CPU Space cannot use.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

# Download the embedding model in its own layer, so a code change does not download it again.
# The name must match EMBEDDING_MODEL in feesbot/settings.py.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-mpnet-base-v2')"

COPY --chown=user feesbot ./feesbot
COPY --chown=user N26 ./N26
COPY --chown=user internal_policy.pdf .

# Build the search index NOW, not at start-up: waking a sleeping Space then takes seconds instead of
# minutes, and it needs no network. No API key is needed for this step.
RUN python -m feesbot ingest

# From here on the embedding model must come from the cache baked into the image.
ENV HF_HUB_OFFLINE=1

EXPOSE 7860
CMD ["python", "-m", "feesbot", "serve", "--host", "0.0.0.0", "--port", "7860"]
