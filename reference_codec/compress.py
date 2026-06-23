"""Reference Glyph codec (compress half): a deterministic zstd -19 wrapper.

This is the baseline-beating reference artifact and the M0 harness fixture. A real miner
replaces this with a neural arithmetic coder; the entrypoint contract is identical.
"""

import argparse

import zstandard as zstd

# Max zstd level (ultra). A stub that beats the zstd -19 baseline floor; a real miner
# replaces this with a neural arithmetic coder that beats it by much more.
LEVEL = 22


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "rb") as handle:
        data = handle.read()
    # write_content_size=True (default) embeds the original size so decompress is exact.
    compressor = zstd.ZstdCompressor(level=LEVEL)
    blob = compressor.compress(data)
    with open(args.output, "wb") as handle:
        handle.write(blob)


if __name__ == "__main__":
    main()
