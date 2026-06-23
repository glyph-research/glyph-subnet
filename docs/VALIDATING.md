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

All-in-one (recommended to start):

```bash
pm2 start pm2/ecosystem.validator.config.js     # edit wallet/netuid first
```

Or split into services:

```bash
pm2 start pm2/ecosystem.reign-worker.config.js   # evaluate + update crown
pm2 start pm2/ecosystem.weight-setter.config.js  # temporal-burn weights every tempo
pm2 start pm2/ecosystem.oracle.config.js         # daily fresh corpus
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
