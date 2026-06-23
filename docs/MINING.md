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
- no outbound network, cloud storage, shell download/upload helpers, or hidden fetches.
  Validators statically review artifact source for common exfiltration paths and the
  production Chutes runner executes entrypoints with network isolation and scrubbed secrets.

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
