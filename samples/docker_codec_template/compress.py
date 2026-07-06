"""Template compress entrypoint -- swap the zstd call for your real codec's compression."""

import argparse

import zstandard as zstd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.input, "rb") as handle:
        data = handle.read()
    with open(args.output, "wb") as handle:
        handle.write(zstd.ZstdCompressor(level=19).compress(data))


if __name__ == "__main__":
    main()
