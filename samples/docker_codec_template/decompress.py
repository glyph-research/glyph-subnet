"""Template decompress entrypoint -- swap the zstd call for your real codec's decompression."""

import argparse

import zstandard as zstd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.input, "rb") as handle:
        blob = handle.read()
    data = zstd.ZstdDecompressor().decompress(blob)
    with open(args.output, "wb") as handle:
        handle.write(data)


if __name__ == "__main__":
    main()
