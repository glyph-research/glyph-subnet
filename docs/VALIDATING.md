# Validating on Glyph

A validator evaluates committed codecs on fresh, beacon-seeded streams and sets weights
with the king-of-the-hill + temporal-burn policy. GPU work is bursted to Chutes (SN64);
the always-on parts are cheap CPU.

## Setup

```bash
./scripts/install_deps.sh
cp .env.example .env            # set CHUTES_API_KEY=cpk_...  (chutes keys create --admin)
pytest -q
```

## Deploy the evaluation chute (once)

```bash
./scripts/deploy_runner_chute.sh
# note the chute URL; pass it via --chute-url or GLYPH_CHUTE_URL
```

If you deploy under a Chutes account other than the default `glyph`, set
`GLYPH_CHUTE_USERNAME=<account>` (it builds the default chute URL and is the build/deploy
username). All validators must point at the *same* deployed chute — set `GLYPH_CHUTE_URL` to
override the URL directly. These are deployment-specific; consensus-critical launch values
(e.g. the burn-window anchor) are committed in `src/core/constants.py` and identical
network-wide, never per-operator env.

### Validate the live invocation contract

The chute is invoked as `POST {base}/run_stream` with `Authorization: Basic <cpk_...>`; the
request body is `chute_app.RunStreamRequest` and the reply is a JSON dump of
`StreamResultModel`. That binding is pinned offline by `tests/test_chute_contract.py` (runs in
CI, no GPU). After a deploy, confirm it end-to-end on the live instance with the smoke helper,
which drives the real `ChutesRunner` for both stream shapes:

```bash
CHUTES_API_KEY=cpk_... python scripts/smoke_chute.py \
  --repo you/glyph-ref-codec --rev main \
  --chute-url https://<acct>-glyph-runner.chutes.ai \
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

For production Chutes runs, also publish the same corpus as one contiguous blob (chunk order ==
sorted manifest order) and pass its URL via `--corpus-url`. The deployed runner then range-fetches
each stream itself instead of the validator inlining the 256 MiB sample. Without `--corpus-url`
(or with `--runner local`), streams are inlined as before — fine for tests and smoke runs.

## Run

All-in-one — `glyph-validator` is a console entry point; wrap it in PM2 (edit wallet/netuid):

```bash
pm2 start glyph-validator --name glyph-validator -- \
  --netuid 117 --wallet-name validator --hotkey-name default --runner chutes \
  --corpus-url https://<host>/corpus.bin --state-dir ./state
```

Or split into services (each is a console entry point — same `pm2 start <script> -- <args>` form):

```bash
pm2 start glyph-reign-worker  --name glyph-reign-worker  -- --netuid 117 --wallet-name validator --hotkey-name default --runner chutes --corpus-url https://<host>/corpus.bin   # evaluate + update crown
pm2 start glyph-weight-setter --name glyph-weight-setter -- --netuid 117 --wallet-name validator --hotkey-name default                                          # temporal-burn weights every tempo
pm2 start glyph-oracle        --name glyph-oracle        -- --out-dir ./corpus --target-bytes 268435456                                                         # daily fresh corpus
```

Auto-updating validator (tracks `glyph-research/glyph-subnet`):

```bash
./scripts/setup_hooks.sh
./scripts/run_auto_validator.sh --network finney --netuid 117 \
  --wallet-name validator --hotkey-name default --runner chutes \
  --corpus-url https://<host>/corpus.bin --state-dir ./state
```

## Notes

- **Version safety**: the validator fail-closes if `core.__version_key__` ≠ the
  on-chain `weights_version`. Bump both together on breaking changes.
- **Commit-reveal** must be enabled on the subnet for the anti-copy burn schedule to bite.
- **Offline check** (no chain/Chutes): `glyph-validator --offline-demo --runner local ...`
  uses the mixed corpus by default; pass `--corpus-dir samples/corpus` for the tiny
  bundled sample.
