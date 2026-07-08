# Glyph Subnet

[![CI](https://github.com/glyph-research/glyph-subnet/actions/workflows/ci.yml/badge.svg)](https://github.com/glyph-research/glyph-subnet/actions/workflows/ci.yml)

Glyph is a lossless **neural text-compression** benchmark subnet — a perpetual,
decentralized Hutter Prize. Miners commit one permanent codec per hotkey. Validators
sample fresh, never-before-seen text, run each codec's compress→decompress round-trip on
local Docker on an RTX 4090 (Chutes (SN64) serverless GPU is available as an optional
secondary path — see [docs/VALIDATING.md](docs/VALIDATING.md)), and set weights with a
king-of-the-hill policy:

- current winner: `70%` · previous winner: `30%`
- plus a `25%` **temporal burn** (one unpredictable tempo per 4-tempo window → UID 0) that
  makes copy-cat validation strictly losing. **Currently disabled** network-wide
  (`core.constants.BURN_ENABLED = False`, issue #43) — see
  [docs/reign-and-burn.md](docs/reign-and-burn.md) for the current state and re-enable procedure.

Score = compression ratio (compressed ÷ raw, lower is better) with a hard **bit-exact
round-trip** gate. A challenger takes the crown only by beating the incumbent by `ε`
(default `5%`); ties go to the earliest commit; losers are excluded forever (one shot).

See [`docs/`](docs) for guides.

## Repository layout

```
src/
  core/        shared: constants, commitments, artifact, state, weights, burn_schedule, dotenv
  chain/       chain adapter + commitment reader        (glyph-chain-reader)
  validation/  codec artifact precheck + checks
  eval/        runners (local + Chutes), chute_app, evaluator, scoring, streams, corpus,
               live_corpus (per-round beacon-seeded HF corpus, issue #71), deploy
  weight_setter/     temporal-burn weights                    (glyph-weight-setter)
  reign_worker/      king-of-the-hill round                   (glyph-reign-worker)
  validator/   all-in-one orchestrator + offline demo   (glyph-validator)
miner/               commit | check | publish | register      (glyph-miner)
scripts/             install, genesis king, deploy chute, auto-update (run_auto_validator.sh)
reference_codec/     minimal zstd codec (artifact contract example)
samples/             bundled corpus + demo codec for the offline demo
docs/  tests/
```

## Install

```bash
./scripts/install_deps.sh           # venv + package + pm2
# or: pip install -e ".[dev]"
cp .env.example .env                # miners: set HF_TOKEN; validators: only if using --runner chutes
pytest -q
```

## Miner

```bash
glyph-miner check   --local-path ./reference_codec        # self-benchmark locally
glyph-miner publish --path ./my-codec --repo you/your-codec
glyph-miner register --netuid 117 --wallet-name w --hotkey-name h
glyph-miner commit  --netuid 117 --wallet-name w --hotkey-name h --model-repo you/your-codec
```

See [docs/MINING.md](docs/MINING.md). Commitments are permanent per hotkey.

## Validator

**Default eval path is local Docker on an RTX 4090** — every validator running GPU codecs must
use identical hardware, or compress/decompress throughput isn't comparable across validators
(same-system determinism). This is a network-wide requirement, not a suggestion: a
validator without Docker + `nvidia-container-toolkit` + a matching GPU **fails closed by
design** (`DockerRunner` checks the GPU model via `nvidia-smi` and refuses to run on anything
else). See [docs/VALIDATING.md](docs/VALIDATING.md) for the full requirement and CPU-only opt-out.

Every validator builds its own copy of the evaluation corpus live from HuggingFace
(FineWeb + Pile + enwik9), keyed by the round's on-chain beacon — no owner-run oracle
process, no shared corpus file to host or keep in sync (issue #71); see
[docs/VALIDATING.md](docs/VALIDATING.md#corpus) for the determinism guarantee.

```bash
docker build -f docker/glyph-runner-default.Dockerfile -t glyph-runner-default:latest .   # zstandard-enabled base image
# auto-updating validator under PM2 (edit wallet/netuid) -- --runner docker + --docker-gpu are
# both the default, so they don't need to be passed explicitly
./scripts/run_auto_validator.sh --netuid 117 \
  --wallet-name w --hotkey-name h --docker-image glyph-runner-default:latest \
  --state-dir ./state
```

Or dispatch to the deployed Chutes (SN64) eval chutes instead (subject to Chutes' own SKU/
availability):

```bash
cp .env.example .env                                       # CHUTES_API_KEY
./scripts/deploy_runner_chute.sh                           # deploy the compress + decompress chutes (once)
./scripts/run_auto_validator.sh --netuid 117 \
  --wallet-name w --hotkey-name h --runner chutes --state-dir ./state
```

Offline M0 demo (no chain, no Chutes) — exercises eval → king-of-the-hill → weights:

```bash
glyph-validator --offline-demo --corpus-dir samples/corpus \
  --eval-source demo --eval-streams 4 --eval-stream-bytes 2000 \
  --eval-benchmark-source "" --eval-benchmark-streams 0 \
  --floor-bps 1 --baseline-level 3 \
  --local-codec weak=./samples/demo_codec_l6 --local-codec strong=./reference_codec
```

See [docs/VALIDATING.md](docs/VALIDATING.md) and [docs/reign-and-burn.md](docs/reign-and-burn.md).

## Version safety

The validator fail-closes when `core.__version_key__` ≠ the subnet's on-chain
`weights_version`. With commit-reveal enabled, `set_weights` auto-routes through
commit/reveal. The PM2 auto-updater (`scripts/run_auto_validator.sh`) tracks
`glyph-research/glyph-subnet`.

## Tests

```bash
pytest -q
```
