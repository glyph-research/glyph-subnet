# Mining on Glyph

A miner publishes one permanent **codec** per hotkey — a self-contained compressor +
decompressor that losslessly compresses fresh text better than the field. Score is the
compression ratio (lower is better) with a hard bit-exact round-trip gate.

## 1. Build a codec artifact

A codec is a directory (a HuggingFace repo at a pinned revision) with:

- `manifest.json` declaring two entrypoints using `{input}`/`{output}` placeholders:
  ```json
  {
    "schema_version": 1,
    "name": "my-codec",
    "entrypoints": {
      "compress":   ["python3", "compress.py",   "--input", "{input}", "--output", "{output}"],
      "decompress": ["python3", "decompress.py", "--input", "{input}", "--output", "{output}"]
    },
    "license": "MIT"
  }
  ```
- the entrypoint scripts + any weights.

See [`../reference_codec/`](../reference_codec) for a minimal zstd example. A real miner
ships a neural arithmetic coder (LLM/NN next-token probabilities → range coder).

### Requirements
- **same-system deterministic**: compress and decompress run on the same worker, so output
  must be byte-identical there (use integer-quantized inference).
- bit-exact round-trip on every stream; decompress throughput ≥ 10 KiB/s.
- artifact under the size cap (default 10 GiB); must beat the zstd-19 baseline to take a
  vacant crown, and beat the incumbent by ≥ 5% (relative) to dethrone it.
- resource caps enforced during evaluation: ≤ 24 GiB VRAM, ≤ 32 GiB RAM
  (`core.constants.VRAM_CAP_BYTES` / `RAM_CAP_BYTES`).
- no outbound network, cloud storage, shell download/upload helpers, or hidden fetches.
  Validators statically review artifact source for common exfiltration paths and the
  production Chutes runner executes entrypoints with network isolation and scrubbed secrets.
  (A codec that ships its own `image` -- see below -- gets network access only during its
  sealed-off warmup phase, before any eval data exists anywhere it can reach; static source
  review no longer applies once code and deps are baked into an opaque image.)

## 1b. Shipping your own Docker image (optional, advanced)

Most codecs (an integer-quantized model + arithmetic coder) fit fine as plain
entrypoint scripts inside the default validator-supplied image. If your codec needs
heavier or more specific dependencies (a particular CUDA/torch build, a custom compiled
extension, etc.), ship your own image instead:

1. **Build and push a digest-pinned image.** A mutable tag (`:latest`, `:1.0`) is rejected
   at precheck -- every validator must run the exact same bytes, so the manifest must
   reference the image by its immutable content digest:
   ```bash
   docker build -t ghcr.io/your-user/your-codec:1.0 .
   docker push ghcr.io/your-user/your-codec:1.0
   docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/your-user/your-codec:1.0
   # -> ghcr.io/your-user/your-codec@sha256:<64 hex chars> -- this is what goes in manifest.json
   ```
2. **Declare it in `manifest.json`**, alongside the usual `entrypoints`:
   ```json
   {
     "schema_version": 1,
     "name": "my-codec",
     "entrypoints": {
       "compress":   ["python3", "/app/compress.py",   "--input", "{input}", "--output", "{output}"],
       "decompress": ["python3", "/app/decompress.py", "--input", "{input}", "--output", "{output}"]
     },
     "image": "ghcr.io/your-user/your-codec@sha256:<64 hex chars>",
     "warmup": {
       "command": ["python3", "/app/warmup.py"],
       "timeout_secs": 300
     },
     "license": "MIT"
   }
   ```
   Your entrypoint scripts/weights now live INSIDE the image (`/app/...` above), not in the
   HuggingFace-published artifact dir -- that dir only needs to carry `manifest.json` itself
   (still the thing committed/hashed on-chain).
3. **Understand the validator lifecycle** your image runs under (`DockerRunner`,
   `src/eval/runner_docker.py`):
   - **Warmup (network ON):** your container starts detached, with network access and an
     empty scratch mount -- no eval data exists anywhere yet. If you set `warmup.command`,
     it's run via `docker exec` and must exit 0 when ready (install deps, download/load
     weights into VRAM, etc.) within `timeout_secs`, or the round fails closed. (If you omit
     `warmup.command`, your image's own `CMD`/`ENTRYPOINT` runs instead and must be a
     long-running process that creates `warmup.ready_file` itself when ready -- it must not
     exit, since the scored entrypoint below is `docker exec`'d into the same container.)
   - **Seal:** your container's network is severed the instant it's ready. It has no route
     out from this point on, for either the compress or the decompress phase.
   - **Benchmark:** only now is the eval input written into the scratch mount, and your
     `compress`/`decompress` entrypoint is run via `docker exec` -- offline, GPU/RAM-capped,
     same throughput floor and wall-clock budget as any other codec.
   - Compress and decompress each get their **own fresh container and warmup** -- this
     preserves the existing anti-cheat invariant that a codec cannot stash the raw input
     during compress and read it back during decompress (there is no shared state between
     the two phases), but it does mean warmup cost is paid twice per stream. Design your
     warmup to be fast (cache weights in the image layer itself rather than downloading them
     every time) if that matters for your throughput budget.
   - `glyph-miner check` (local preflight) uses `LocalSubprocessRunner`, which does **not**
     simulate this lifecycle -- it's only a sanity check of your entrypoints' basic
     correctness, not a substitute for testing against a real `DockerRunner`.

## 2. Self-benchmark locally

```bash
glyph-miner check --local-path ./my-codec --sample-bytes 1000000
# reports: roundtrip_ok, ratio, beats_baseline, meets_floor
```

## 3. Publish and commit

```bash
cp .env.example_miners .env          # set HF_TOKEN
glyph-miner publish --path ./my-codec --repo your-user/your-codec
glyph-miner register --netuid 117 --wallet-name w --hotkey-name h   # if not registered
glyph-miner commit  --netuid 117 --wallet-name w --hotkey-name h \
  --model-repo your-user/your-codec --dry-run
glyph-miner commit  --netuid 117 --wallet-name w --hotkey-name h \
  --model-repo your-user/your-codec
```

**Commitments are permanent per hotkey.** A challenger that does not win is excluded
forever — so benchmark against the public incumbent before paying to commit. To submit an
improved codec, register a new hotkey.

`glyph-miner commit` always resolves `--revision` to its pinned HuggingFace commit SHA for
you before committing (or resolves the repo's default branch if `--revision` is omitted) —
validators reject any commitment whose revision isn't a full 40-character commit SHA (issue
#96), since a mutable branch name would let the underlying content change after precheck.
