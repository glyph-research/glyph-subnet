# Glyph architecture

Glyph is organized as a set of small service packages under `src/`, a top-level `miner/`
package, and PM2 ecosystems that run the services as independent processes.

## Packages (`src/`)

| Package | Role |
|---|---|
| `core` | Shared primitives: constants, commitments, artifact contract + hashing, validator state, rolling-winner weights, temporal burn schedule, `.env` loader. Also holds `__version__` / `__version_key__`. |
| `chain` | Bittensor chain adapter (`chain.py`) and a commitment reader (`reader.py` → `glyph-chain-reader`). |
| `validation` | Codec artifact precheck (manifest/entrypoints/size, hashing, duplicate-hash disqualification). |
| `eval` | Evaluation service: codec runners (`LocalSubprocessRunner`, `ChutesRunner`), the deployed `glyph-runner` chute (`chute_app.py`), paired evaluator, scoring + gates, beacon stream sampling, corpus providers, chute deploy CLI. |
| `oracle` | Fresh-data oracle: scrapes attested-timestamp text into a corpus + manifest (`glyph-oracle`). |
| `weight_setter` | Temporal-burn weight decision (`decide_weights`) + standalone setter service (`glyph-weight-setter`). |
| `reign_worker` | King-of-the-hill round: paired evaluation + crown update + one-shot exclusion (`glyph-reign-worker`). |
| `validator` | All-in-one orchestrator composing reign + weights, plus the offline M0 demo (`glyph-validator`). |
| `miner/` | `commit`, `check`, `publish`, `register` subcommands behind the `glyph-miner` dispatcher. |

## Data flow (one validator cycle)

```
chain commitments ──▶ validation.precheck ──▶ state.commitments
fresh corpus (oracle) ──▶ beacon-seeded streams ─┐
                                                        ▼
reign_worker.run_round: paired eval on eval runner (Chutes) ──▶ scores ──▶ crown
                                                        │
state.winner_history + last_round_outputs ─────────────┘
                                                        ▼
weight_setter.decide_weights (70/30 ± temporal burn) ──▶ chain.set_weights (commit-reveal)
```

## Deployment

Run the all-in-one validator (`pm2 start pm2/ecosystem.validator.config.js`) or split it
into `glyph-reign-worker` + `glyph-weight-setter` + `glyph-oracle` for separation of
concerns. The `glyph-runner` chute is deployed once to Chutes (SN64) and shared by all
validators. See [VALIDATING.md](VALIDATING.md).
