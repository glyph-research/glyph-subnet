# Validating on Glyph

A validator evaluates committed codecs on fresh, beacon-seeded streams and sets weights
with the king-of-the-hill + temporal-burn policy. GPU work runs on one of two backends,
chosen with `--runner`:

- **`docker` (default)**: runs compress/decompress as ephemeral local Docker containers on
  operator-controlled hardware. **`--docker-gpu` is also on by default, and it requires an RTX
  4090** (`core.constants.DOCKER_REFERENCE_GPU`) -- every validator running GPU codecs must use
  identical hardware, or `compress_secs`/`decompress_secs` (gated against `THROUGHPUT_FLOOR_BPS`)
  aren't comparable across validators (same-system determinism). This is intentional
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

Every round streams a fresh corpus slice straight from HuggingFace
(`eval.live_corpus.resolve_live_corpus`). This works fully anonymously, but HF's Xet-backed
CDN increasingly throttles/denies anonymous traffic with an intermittent `403 Forbidden` --
set `HF_TOKEN` in `.env` (a free-tier read token, no special dataset permissions needed;
create one at https://huggingface.co/settings/tokens) to avoid it. See `.env.example`.

## Set up the default Docker runner

`--runner docker` (`src/eval/runner_docker.py`, the default) is a drop-in alternative to
`--runner chutes`: same contract (`CodecRunner`), same split-worker isolation (compress and
decompress each run in a fresh, ephemeral `docker run --rm` container with `--network none`, so
a codec can't stash the raw input during compress and read it back during decompress), but on
hardware you control instead of Chutes.

`./scripts/install_deps.sh` already builds `glyph-runner-default:latest` and it's the
`--docker-image` default, so a plain invocation just works:

```bash
glyph-validator --netuid 117 --wallet-name validator --hotkey-name default
```

- Requires Docker **and `nvidia-container-toolkit`** on the validator host (the GPU gate is on
  by default -- see above). Add `--docker-gpu-device 0` to pin a specific card if the host has
  more than one.
- No GPU available, or intentionally testing a CPU-only codec? Pass `--no-docker-gpu` --
  `DockerRunner` then runs with no GPU requirement at all.
- Pass `--docker-image` to override the default with your own image; pre-pull/build it first --
  a cold pull happens inside the timed wall-clock budget. `docker/glyph-runner-default.Dockerfile`
  covers the reference codec (needs only `zstandard`) -- a codec with heavier deps (e.g. torch
  for a neural codec) needs its own image with those baked in. Throughput timing includes
  container startup and there is no `--cpus` cap, so also use comparable host CPUs across
  validators for tightest same-system determinism, even with the GPU pinned.
- Codec containers run as non-root UID/GID `65534:65534`, drop all Linux capabilities, set
  `no-new-privileges`, and use Docker's default seccomp profile. To test a reviewed stricter
  profile, pass `--docker-seccomp-profile /path/to/seccomp-codec.json`. See
  [`CODEC_SANDBOX_HARDENING.md`](CODEC_SANDBOX_HARDENING.md).
- Like `--runner local`, this fetches each codec artifact to local disk first, so
  `needs_local_artifact`-style runners always get inlined streams -- fine for
  validator-run hardware, unlike the untrusted-worker Chutes path.
- Live-verified against real LLM-driven compression (RWKV-4 + arithmetic coding, ts_zip-style)
  on an actual RTX 4090: bit-exact round trips against the real live-streamed mixed corpus,
  GPU genuinely used inside the isolated container, and the GPU-model gate both accepts the
  real RTX 4090 and rejects a simulated mismatched card.

### Miner-published images (issue #48)

A codec whose manifest declares its own digest-pinned `image` (see `docs/MINING.md`) runs a
different lifecycle than the default single-shot path above: `DockerRunner` starts that
container **detached with network access and no eval data present**, waits for it to signal
ready (bounded by the manifest's `warmup.timeout_secs`, default 300s), **severs its network**,
and only then writes the eval input into its scratch mount and `docker exec`s the scored
compress/decompress entrypoint -- offline, same GPU/RAM caps and wall-clock budget as any
other codec. Operationally:

- This is why `--docker-gpu`/RTX-4090 host requirements still apply identically -- the GPU
  cap attaches at container start and persists through warmup and the sealed benchmark.
- A validator host firewall that normally blocks all outbound Docker traffic needs to permit
  it during this specific warmup window (a per-invocation dedicated bridge network, not the
  shared default bridge); the runner severs it again before any eval byte is ever written to
  disk inside the container, so the exposure window is bounded and never overlaps with the
  corpus/blob being present.
- A codec whose manifest omits `image` is entirely unaffected -- it keeps the original
  `--network none`-from-start, no-warmup path with no behavior change.
- See [`CODEC_SANDBOX_HARDENING.md`](CODEC_SANDBOX_HARDENING.md) for the security rationale
  (why the networked window can't leak eval data) and current test coverage.

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

## `--runner local` (dev/CI fallback, not recommended for production)

Runs compress/decompress as a subprocess on the validator's own OS user rather than an
isolated Docker container or a Chutes worker. Since this still evaluates real on-chain
commitments (untrusted miner code), it is **network-isolated by default**
(`unshare --net`) and fails closed (`RunnerError`) if `unshare` is unavailable on the host,
rather than silently running the codec unisolated with full network + wallet-key access.

There is no way to opt out of this for real commitments: `--unsafe-local-no-sandbox` is
refused outright with a `SystemExit` if passed alongside `--runner local`. To run your own
codec unsandboxed for local testing, use the offline demo instead (no chain, no real
commitments involved):

```bash
glyph-validator --offline-demo --runner local --local-codec mine=./my-codec ...
```

## Corpus

There is no owner-run corpus process and nothing to configure: every validator builds its
own copy of the evaluation corpus directly from HuggingFace at the start of each round
(`eval.live_corpus.resolve_live_corpus`, issue #71) -- the launch mix is the same 8 x 2 MiB
(3x FineWeb, 3x Pile-derived, 2x enwik9) it always was.

The corpus is keyed off the same beacon (derived from the round's chain block hash) already
used to pick which stream windows get scored, via a seed-derived skip offset
(`_skip_for_seed`) into each dataset -- never the fixed dataset prefix, so a miner cannot
memorise it in advance. Because that skip offset is a pure function of `(seed, dataset)`,
independent validators land on byte-identical bytes with no shared file and no coordination
between them; `tests/test_live_corpus.py` proves this directly (same seed -> identical
corpus, different seed -> different corpus, slice is never the dataset prefix). Results are
cached locally per seed (`eval.live_corpus.local_corpus_cache_dir`) so re-evaluating the same
round doesn't re-stream from HuggingFace.

`--corpus-dir` still exists but is **offline-demo only** (see below) -- passing it to a real
round is refused outright with a `SystemExit`, since it would otherwise be silently ignored.

With the default `--runner docker` (or `--runner local`), the validator's own host executes
compress/decompress, so streams are always inlined -- no publishing step needed. `--runner
chutes` inlines too now that there is no shared corpus file to range-fetch from a public URL.

## Run

All-in-one — `glyph-validator` is a console entry point; wrap it in PM2 (edit wallet/netuid).
`--runner docker` and `--docker-gpu` are both the default, so a plain invocation is the RTX
4090 + Docker path:

```bash
pm2 start glyph-validator --name glyph-validator -- \
  --netuid 117 --wallet-name validator --hotkey-name default
```

Or split into services (each is a console entry point — same `pm2 start <script> -- <args>` form):

```bash
pm2 start glyph-reign-worker  --name glyph-reign-worker  -- --netuid 117 --wallet-name validator --hotkey-name default   # evaluate + update crown
pm2 start glyph-weight-setter --name glyph-weight-setter -- --netuid 117 --wallet-name validator --hotkey-name default   # temporal-burn weights every tempo
```

Auto-updating validator (tracks `glyph-research/glyph-subnet`):

```bash
./scripts/setup_hooks.sh
./scripts/run_auto_validator.sh --network finney --netuid 117 --wallet-name validator --hotkey-name default
```

Using Chutes instead:

```bash
./scripts/run_auto_validator.sh --network finney --netuid 117 \
  --wallet-name validator --hotkey-name default --runner chutes
```

## Console logging

`glyph-validator`, `glyph-weight-setter`, and `glyph-reign-worker` log through
`bittensor.utils.btlogging` (issue #80), so every line gets a timestamp and level
(INFO/WARNING/ERROR) instead of a bare `print()`. INFO is the default (matching what these
services always printed); pass `--logging.debug` or `--logging.trace` for more verbosity.

A real round with challengers now logs each stage as it happens (issue #81), not just a
post-hoc summary once everything is already done: each commitment's precheck result
(hotkey + repo/rev + valid/invalid + reason), a "round: evaluating incumbent=..., N
challenger(s): [...]" line before evaluation starts, and every candidate's ratio/validity
once scored -- including challengers that lose, not only the eventual winner. Within a single
candidate's evaluation, every stream also logs before it starts and again once it finishes
(ratio/roundtrip/timing) or fails (issue #86) -- each stream can legitimately take up to the
full compress/decompress wall-clock budget, so this is the difference between the log going
silent for many minutes and showing exactly which stream is running and how long each took.

## Observability (wandb)

Every validator process (`glyph-validator`, and `glyph-reign-worker` in the split-service
form) streams per-round metrics to [Weights & Biases](https://wandb.ai) by default —
per-challenger ratio/throughput/validity, the scored FineWeb/Pile breakdown, the
enwik9 benchmark-only ratio, crown changes, and (all-in-one path only) the weights/burn
decision. It also mirrors the process's own stdout/stderr into the run's Logs tab. This is
pure observability: wandb is never read back into scoring, promotion, or weight-setting, and
a wandb outage (network down, bad credentials, quota) only disables further logging — it
never crashes or delays a round.

With no `WANDB_API_KEY` set, the validator falls back to an **anonymous run**: a real,
shareable wandb run created without requiring a login to view. The run URL is printed to
stdout on start (`[wandb] run started: https://wandb.ai/...`) — share that link to let
others watch your validator's rounds live.

`scripts/run_auto_validator.sh` resolves this *before* starting the pm2-managed validator: if
`WANDB_API_KEY` isn't already set and neither `--wandb.off` nor `--wandb.offline` was passed,
it prompts for the key in the foreground (there's no terminal to answer a prompt once the
process is backgrounded under pm2). Pass `--wandb-non-interactive` to skip the prompt for
scripted/CI use and let wandb attempt the anonymous fallback instead.

By default, runs log to the `glyph-research-org/text-compression` team project. To log to
your own wandb project/entity instead, set `WANDB_API_KEY` in the environment (or `.env`) and
pass:

```bash
--wandb.project <your-project> --wandb.entity <your-wandb-entity>
```

Other flags:

- `--wandb.off` — disable wandb entirely (no import, no network, byte-identical behavior
  to a build without this feature).
- `--wandb.name` — override the run name. Defaults to this coldkey's on-chain identity name
  (`btcli wallet set-identity`), or its hotkey ss58 if no identity is set, so multiple
  validators sharing the project are distinguishable at a glance.
- `--wandb.offline` — log locally only (writes under `./wandb/`, no network), useful for
  CI or air-gapped testing.
- `--wandb.notes "..."` — free-text note attached to the run.
- `--wandb.restart_interval <hours>` — finish and reopen the run on this cadence so a
  long-lived validator doesn't accumulate one unbounded run (default 24h; 0 disables).

## Notes

- **Version safety**: the validator fail-closes if `core.__version_key__` ≠ the
  on-chain `weights_version`. Bump both together on breaking changes.
- **Commit-reveal** must be enabled on the subnet for the anti-copy burn schedule to bite.
- **Continuous by default** (issue #79): `glyph-validator` and `glyph-weight-setter` loop
  continuously by default -- pass `--once` to run a single round and exit instead (for
  testing/CI; a real validator should not pass this). `--loop` is a deprecated no-op kept only
  so an existing invocation that already passes it doesn't break.
- **set_weights rate limiting**: below the subnet's weights-rate-limit window, `set_weights`
  logs `"set_weights: skipped, rate-limited (N blocks remaining)"` instead of attempting and
  reporting a contentless failure (the bittensor SDK returns no error/message in that case by
  construction).
- **Offline check** (no chain, no Docker/GPU, no HuggingFace access needed):
  `glyph-validator --offline-demo --runner local ...` uses the bundled `samples/corpus`
  sample by default; pass `--corpus-dir` to point at a different local directory instead.
