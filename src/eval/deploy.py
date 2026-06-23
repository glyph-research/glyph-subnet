"""Build and deploy the glyph eval chutes (DESIGN §6, build step 11).

Thin wrapper around the ``chutes`` CLI so operators stand up the evaluation endpoints
reproducibly. Compress and decompress are SEPARATE chutes (separate containers) so a codec
cannot stash the raw input during compress and read it back during decompress (#14):
``eval.chute_app:compressor_chute`` and ``eval.chute_app:decompressor_chute``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

CHUTE_REFS = ("eval.chute_app:compressor_chute", "eval.chute_app:decompressor_chute")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and deploy the glyph eval chutes")
    parser.add_argument("--build", action="store_true", help="Run `chutes build`")
    parser.add_argument("--deploy", action="store_true", help="Run `chutes deploy`")
    parser.add_argument("--public", action="store_true", help="Build/publish a public image")
    parser.add_argument("--accept-fee", action="store_true", help="Accept the deploy fee non-interactively")
    return parser


def _run(argv: list[str]) -> int:
    print("+", " ".join(argv))
    return subprocess.run(argv).returncode


def main() -> None:
    args = build_parser().parse_args()

    if not (args.build or args.deploy):
        print("Glyph eval chute references (deploy both as separate chutes):")
        for ref in CHUTE_REFS:
            print("  build: chutes build", ref, "--public --wait")
            print("  deploy: chutes deploy", ref, "--accept-fee")
        print("Re-run with --build and/or --deploy to execute these for you.")
        return

    rc = 0
    for ref in CHUTE_REFS:
        if args.build:
            build_cmd = ["chutes", "build", ref, "--wait"]
            if args.public:
                build_cmd.append("--public")
            rc = _run(build_cmd)
            if rc != 0:
                sys.exit(rc)
        if args.deploy:
            deploy_cmd = ["chutes", "deploy", ref]
            if args.accept_fee:
                deploy_cmd.append("--accept-fee")
            rc = _run(deploy_cmd)
            if rc != 0:
                sys.exit(rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()
