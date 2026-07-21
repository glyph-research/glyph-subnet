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
  makes copy-cat validation strictly losing. **Currently enabled** network-wide
  (`core.constants.BURN_ENABLED = True`, issue #88) — see
  [docs/reign-and-burn.md](docs/reign-and-burn.md) for the current state and how to disable.
- **Miner Conviction** (issues #141/#156): a winner slot must keep its cumulative alpha
  earnings **chain-locked** to its hotkey (`btcli lock add --netuid 117 ...`) — the free
  (unlocked-allowed) amount is `max(10% of earned, 1000 α)`. A slot below its required
  lock earns nothing that tempo (its share burns, never reallocated to the other winner)
  and resumes automatically at the next weight-setting once enough is locked. Measured in
  alpha on both sides; the crown itself is never affected. From block `8,740,000`
  (`CONVICTION_LOCK_CHECK_START_BLOCK`) plain staked-but-unlocked alpha no longer counts:
  locked mass is chain-enforced unstakeable, so exit follows the lock's decay schedule
  (or a deliberate perpetual→decaying switch), never a cliff. Either lock mode satisfies
  the gate — perpetual is the low-maintenance choice; a decaying lock gates again as it
  decays below the line until re-locked. Until that block, raw staked alpha is the gated
  quantity (v1 rule).

Score = compression ratio (compressed ÷ raw, lower is better) with a hard **bit-exact
round-trip** gate. A challenger takes the crown only by beating the incumbent by `ε`
(default `5%`). When several challengers land in the same round they run as a **sequential
gauntlet in commit order** (issue #136): the earliest commit challenges first, and each
later challenger must beat the *current* — possibly just-crowned — winner by the full `ε`,
so a later-committed marginal tweak of someone else's round can't steal it; identical
commit blocks tie-break by hotkey; losers are excluded forever (one shot).

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
cp .env.example .env                # miners: set HF_TOKEN (write); validators: recommended too (read-only, avoids anonymous HF CDN 403s) + BLOCKMACHINE_API_KEY (Standard plan, fast conviction backfill) -- see docs/VALIDATING.md
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
(FineWeb-Edu + Pile, 2x/1x scored mix, plus two benchmark-only display windows: enwik9 and
a per-round live-data snapshot of recently-changed Wikipedia text, fetched in the
between-rounds window — text no committed model can have memorized, issue #139),
keyed by the round's on-chain beacon and a seed-derived dataset shard so the reachable
sampling range is the whole dataset, not a fixed slice (issue #112) — no owner-run oracle
process, no shared corpus file to host or keep in sync (issue #71); see
[docs/VALIDATING.md](docs/VALIDATING.md#corpus) for the determinism guarantee.

**Recommended: a [blockmachine](https://rpc.blockmachine.io) RPC API key — choose the
Standard plan.** The Miner Conviction ledger (issue #141) backfills one historical
metagraph per tempo from an archive source; with `BLOCKMACHINE_API_KEY` set in `.env`
(issue #151, the install script asks for it) a fresh backfill takes ~2–3 minutes instead
of 40+ on the public archive node. Optional and purely a local speed preference — without
a key, or on any key failure, the validator automatically uses the public archive node.

`./scripts/install_deps.sh` already builds the `glyph-runner-default:latest` image (zstandard-enabled),
so no separate `docker build` step is needed here:

```bash
# auto-updating validator under PM2 (edit wallet/netuid) -- --runner docker, --docker-gpu,
# --docker-image, and --state-dir are all defaults, so none of them need to be passed explicitly
./scripts/run_auto_validator.sh --netuid 117 --wallet-name w --hotkey-name h
```

Or dispatch to the deployed Chutes (SN64) eval chutes instead (subject to Chutes' own SKU/
availability):

```bash
cp .env.example .env                                       # CHUTES_API_KEY
./scripts/deploy_runner_chute.sh                           # deploy the compress + decompress chutes (once)
./scripts/run_auto_validator.sh --netuid 117 --wallet-name w --hotkey-name h --runner chutes
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

The validator never scores or sets weights while `core.__version_key__` ≠ the subnet's
on-chain `weights_version` — a mismatch (expected briefly during a version-bump release,
until the owner updates the on-chain hyperparameter) logs a warning and idles, retrying
each round, rather than crashing the process (issue #120). With commit-reveal enabled,
`set_weights` auto-routes through commit/reveal. The PM2 auto-updater
(`scripts/run_auto_validator.sh`) tracks `glyph-research/glyph-subnet`.

## Tests

```bash
pytest -q
```
