# Template for a miner-published codec image (issue #48).
#
# Build, push, and pin by digest -- a mutable tag is rejected at precheck:
#
#   docker build -t ghcr.io/your-user/your-codec:1.0 .
#   docker push ghcr.io/your-user/your-codec:1.0
#   docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/your-user/your-codec:1.0
#   # -> paste that ghcr.io/your-user/your-codec@sha256:... into manifest.json's "image"
#
# Bake your weights/deps into the image itself where practical -- warmup runs (and its
# cost is paid) fresh for every compress AND every decompress container, since the two
# phases never share state (see docs/MINING.md).
FROM python:3.12-slim

RUN pip install --no-cache-dir zstandard   # stand-in for your real heavier deps (torch, etc.)
WORKDIR /app
COPY compress.py decompress.py warmup.py /app/

# No CMD/ENTRYPOINT override needed here: the runner starts this image with `sleep
# infinity` as a keep-alive and `docker exec`s warmup.py / compress.py / decompress.py
# into it directly (see manifest.json's "warmup.command" and the entrypoints above).
