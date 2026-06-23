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

## Provide a corpus

Run the data oracle (or point at any corpus directory):

```bash
glyph-oracle --out-dir ./corpus --target-bytes 268435456   # 256 MiB of fresh text
```

## Run

All-in-one — `glyph-validator` is a console entry point; wrap it in PM2 (edit wallet/netuid):

```bash
pm2 start glyph-validator --name glyph-validator -- \
  --netuid 117 --wallet-name validator --hotkey-name default --runner chutes \
  --corpus-dir ./corpus --state-dir ./state
```

Or split into services (each is a console entry point — same `pm2 start <script> -- <args>` form):

```bash
pm2 start glyph-reign-worker  --name glyph-reign-worker  -- --netuid 117 --wallet-name validator --hotkey-name default --runner chutes --corpus-dir ./corpus   # evaluate + update crown
pm2 start glyph-weight-setter --name glyph-weight-setter -- --netuid 117 --wallet-name validator --hotkey-name default                                          # temporal-burn weights every tempo
pm2 start glyph-oracle        --name glyph-oracle        -- --out-dir ./corpus --target-bytes 268435456                                                         # daily fresh corpus
```

Auto-updating validator (tracks `glyph-research/glyph-subnet`):

```bash
./scripts/setup_hooks.sh
./scripts/run_auto_validator.sh --network finney --netuid 117 \
  --wallet-name validator --hotkey-name default --runner chutes --corpus-dir ./corpus --state-dir ./state
```

## Notes

- **Version safety**: the validator fail-closes if `core.__version_key__` ≠ the
  on-chain `weights_version`. Bump both together on breaking changes.
- **Commit-reveal** must be enabled on the subnet for the anti-copy burn schedule to bite.
- **Offline check** (no chain/Chutes): `glyph-validator --offline-demo --corpus-dir samples/corpus --runner local ...`.
