# Optional codec-execution image for --runner docker (DockerRunner, src/eval/runner_docker.py).
#
# DockerRunner's own default image (python:3.12-slim) is deliberately bare -- it doesn't know
# what any given codec needs. This one adds zstandard, enough to run the repo's reference_codec/
# (and most simple zstd-based codecs) out of the box. Build and point --docker-image at it:
#
#   docker build -f docker/glyph-runner-default.Dockerfile -t glyph-runner-default:latest .
#   glyph-validator --runner docker --docker-image glyph-runner-default:latest ...
#
# A codec needing different/heavier deps (e.g. torch for a neural codec) should use its own
# image built the same way -- DockerRunner doesn't run pip install inside the timed execution
# budget, so whatever the codec's entrypoint needs must already be baked into the image.
FROM python:3.12-slim
RUN pip install --no-cache-dir zstandard
