"""Reference Glyph codec (decompress half): zstd inverse of compress.py."""

import argparse

import zstandard as zstd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "rb") as handle:
        blob = handle.read()
    decompressor = zstd.ZstdDecompressor()
    data = decompressor.decompress(blob)
    with open(args.output, "wb") as handle:
        handle.write(data)


if __name__ == "__main__":
    main()
