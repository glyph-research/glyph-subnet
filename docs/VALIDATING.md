# Validating on Glyph

A validator evaluates committed codecs on fresh, beacon-seeded streams and sets weights
with the king-of-the-hill + temporal-burn policy. GPU work runs on one of two backends,
chosen with `--runner`:

- **`docker` (default)**: runs compress/decompress as ephemeral local Docker containers on
  operator-controlled hardware. **`--docker-gpu` is also on by default, and it requires an RTX
  4090** (`core.constants.DOCKER_REFERENCE_GPU`) -- every validator running GPU codecs must use
  identical hardware, or `compress_secs`/`decompress_secs` (gated against `THROUGHPUT_FLOOR_BPS`)
  aren't comparable across validators (DESIGN §4 same-system determinism). This is intentional
  and network-wide, not a bug: **a validator host without Docker + `nvidia-container-toolkit` +
  a matching GPU fails closed** (`DockerRunner` checks the GPU model via `nvidia-smi` at startup
  and refuses to run on anything else, with no bypass flag). Pass `--no-docker-gpu` for CPU-only
  codecs / a testnet host without an RTX 4090. See "Set up the default Docker runner" below.
- `chutes`: bursts compress/decompress to Chutes (SN64) instead. Chutes now *mandates* the
  `pro_6000` GPU SKU for TEE chutes tied to an integrated subnet, and its availability
  fluctuates independent of anything on the validator side.

The always-on parts (chain polling, precheck, weight setting) are cheap CPU either way.

## Setup

```bash
./scripts/install_deps.sh
pytest -q
```

## Set up the default Docker runner

`--runner docker` (`src/eval/runner_docker.py`, the default) is a drop-in alternative to
`--runner chutes`: same contract (`CodecRunner`), same split-worker isolation (compress and
decompress each run in a fresh, ephemeral `docker run --rm` container with `--network none`, so
a codec can't stash the raw input during compress and read it back during decompress), but on
hardware you control instead of Chutes.

```bash
docker build -f docker/glyph-runner-default.Dockerfile -t glyph-runner-default:latest .   # zstandard-enabled base image
glyph-validator --docker-image glyph-runner-default:latest \
  --netuid 117 --wallet-name validator --hotkey-name default \
  --corpus-dir ./corpus --state-dir ./state
```

- Requires Docker **and `nvidia-container-toolkit`** on the validator host (the GPU gate is on
  by default -- see above). Add `--docker-gpu-device 0` to pin a specific card if the host has
  more than one.
- No GPU available, or intentionally testing a CPU-only codec? Pass `--no-docker-gpu` --
  `DockerRunner` then runs with no GPU requirement at all.
- Pre-pull/build whatever image you pass via `--docker-image`; a cold pull happens inside the
  timed wall-clock budget. `docker/glyph-runner-default.Dockerfile` covers the reference codec
  (needs only `zstandard`) -- a codec with heavier deps (e.g. torch for a neural codec) needs
  its own image with those baked in. Throughput timing includes container startup and there is
  no `--cpus` cap, so also use comparable host CPUs across validators for tightest
  same-system determinism, even with the GPU pinned.
- Like `--runner local`, this fetches each codec artifact to local disk first (no
  `--corpus-url` range-fetch path), so `needs_local_artifact`-style runners always get inlined
  streams -- fine for validator-run hardware, unlike the untrusted-worker Chutes path.
- Live-verified against real LLM-driven compression (RWKV-4 + arithmetic coding, ts_zip-style)
  on an actual RTX 4090: bit-exact round trips against the real oracle-produced mixed corpus,
  GPU genuinely used inside the isolated container, and the GPU-model gate both accepts the
  real RTX 4090 and rejects a simulated mismatched card.

## Use Chutes instead (optional)

```bash
cp .env.example .env            # set CHUTES_API_KEY=cpk_...  (chutes keys create --admin)
```

### Deploy the evaluation chutes (once)

Compress and decompress run on **separate** chutes (separate containers), so a codec cannot
stash the raw input during compress and read it back during decompress to fake the ratio — the
decompressor only ever receives the blob.

```bash
./scripts/deploy_runner_chute.sh   # builds + deploys compressor_chute AND decompressor_chute
# note both chute URLs; pass them via --compress-chute-url / --decompress-chute-url
# (or GLYPH_COMPRESS_CHUTE_URL / GLYPH_DECOMPRESS_CHUTE_URL)
```

Each deployed chute downloads the committed artifact, re-runs precheck inside the worker, then
executes only its phase with outbound network disabled and validator secrets removed from the
subprocess environment. If a worker cannot apply network isolation, the benchmark fails closed
instead of running miner code with network access.

If you deploy under a Chutes account other than the default `glyph`, set
`GLYPH_CHUTE_USERNAME=<account>` (it builds the default chute URLs and is the build/deploy
username). All validators must point at the *same* two deployed chutes. These are
deployment-specific; consensus-critical launch values (e.g. the burn-window anchor) are
committed in `src/core/constants.py` and identical network-wide, never per-operator env.

### Validate the live invocation contract

Each chute is invoked as `POST {base}/compress` / `POST {base}/decompress` with
`Authorization: Basic <cpk_...>`; the bodies are `chute_app.CompressRequest` /
`DecompressRequest` and the replies are JSON dumps of `CompressResultModel` /
`DecompressResultModel`. Bit-exactness is gated on the decompress worker's `output_sha256`
matching the hash the validator computed from the trusted corpus — never a worker self-report.
That binding is pinned offline by `tests/test_chute_contract.py` (CI, no GPU). After a deploy,
confirm it end-to-end with the smoke helper, which drives the real `ChutesRunner` across both
chutes for both stream shapes:

```bash
CHUTES_API_KEY=cpk_... python scripts/smoke_chute.py \
  --repo you/glyph-ref-codec --rev main \
  --compress-chute-url https://<acct>-glyph-compressor.chutes.ai \
  --decompress-chute-url https://<acct>-glyph-decompressor.chutes.ai \
  [--corpus-url https://<host>/corpus.bin]   # also exercises the URL/range path
```

It downloads the codec from `--repo@--rev` on the GPU worker, runs a small inline round-trip
(and a URL/range one when `--corpus-url` is given), and exits non-zero unless every round-trip
is bit-exact. The codec must be published on HuggingFace first (`glyph-miner publish`).

## Provide a corpus

By default, the validator reads the mixed launch corpus from `/tmp/glyph_mixed_8x2mb`
or the directory named by `GLYPH_MIXED_CORPUS_DIR`. The launch mix is 8 x 2 MiB:
3x FineWeb, 3x Pile-derived, and 2x enwiki9.

To override it, run the data oracle or point at any corpus directory:

```bash
glyph-oracle --out-dir ./corpus --target-bytes 268435456   # 256 MiB of fresh text
glyph-validator --corpus-dir ./corpus ...
```

With the default `--runner docker` (or `--runner local`), the validator's own host executes
compress/decompress, so streams are always inlined from `--corpus-dir` -- no publishing step
needed. `--runner chutes` is the one exception: also publish the same corpus as one contiguous
blob (chunk order == sorted manifest order) and pass its URL via `--corpus-url`, so the deployed
runner range-fetches each stream itself instead of the validator inlining the 256 MiB sample.

## Run

All-in-one — `glyph-validator` is a console entry point; wrap it in PM2 (edit wallet/netuid).
`--runner docker` and `--docker-gpu` are both the default, so a plain invocation is the RTX
4090 + Docker path:

```bash
pm2 start glyph-validator --name glyph-validator -- \
  --netuid 117 --wallet-name validator --hotkey-name default \
  --docker-image glyph-runner-default:latest --corpus-dir ./corpus --state-dir ./state
```

Or split into services (each is a console entry point — same `pm2 start <script> -- <args>` form):

```bash
pm2 start glyph-reign-worker  --name glyph-reign-worker  -- --netuid 117 --wallet-name validator --hotkey-name default --docker-image glyph-runner-default:latest --corpus-dir ./corpus   # evaluate + update crown
pm2 start glyph-weight-setter --name glyph-weight-setter -- --netuid 117 --wallet-name validator --hotkey-name default                                          # temporal-burn weights every tempo
pm2 start glyph-oracle        --name glyph-oracle        -- --out-dir ./corpus --target-bytes 268435456                                                         # daily fresh corpus
```

Auto-updating validator (tracks `glyph-research/glyph-subnet`):

```bash
./scripts/setup_hooks.sh
./scripts/run_auto_validator.sh --network finney --netuid 117 \
  --wallet-name validator --hotkey-name default \
  --docker-image glyph-runner-default:latest --corpus-dir ./corpus --state-dir ./state
```

Using Chutes instead:

```bash
./scripts/run_auto_validator.sh --network finney --netuid 117 \
  --wallet-name validator --hotkey-name default --runner chutes \
  --corpus-url https://<host>/corpus.bin --state-dir ./state
```

## Notes

- **Version safety**: the validator fail-closes if `core.__version_key__` ≠ the
  on-chain `weights_version`. Bump both together on breaking changes.
- **Commit-reveal** must be enabled on the subnet for the anti-copy burn schedule to bite.
- **Offline check** (no chain, no Docker/GPU needed): `glyph-validator --offline-demo --runner local ...`
  uses the mixed corpus by default; pass `--corpus-dir samples/corpus` for the tiny
  bundled sample.
