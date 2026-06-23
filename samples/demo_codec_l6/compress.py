"""Demo-only weaker codec: zstd level 6. Beats a low baseline but loses to the level-22 reference."""

import argparse

import zstandard as zstd

LEVEL = 6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.input, "rb") as handle:
        data = handle.read()
    with open(args.output, "wb") as handle:
        handle.write(zstd.ZstdCompressor(level=LEVEL).compress(data))


if __name__ == "__main__":
    main()
