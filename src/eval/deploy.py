"""Build and deploy the glyph-runner chute (DESIGN §6, build step 11).

Thin wrapper around the ``chutes`` CLI so operators stand up the evaluation endpoint
reproducibly. The chute itself is defined in ``eval.chute_app:chute``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

CHUTE_REF = "eval.chute_app:chute"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and deploy the glyph-runner chute")
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
        print("Glyph runner chute reference:", CHUTE_REF)
        print("Build:  chutes build", CHUTE_REF, "--public --wait")
        print("Deploy: chutes deploy", CHUTE_REF, "--accept-fee")
        print("Re-run with --build and/or --deploy to execute these for you.")
        return

    rc = 0
    if args.build:
        build_cmd = ["chutes", "build", CHUTE_REF, "--wait"]
        if args.public:
            build_cmd.append("--public")
        rc = _run(build_cmd)
        if rc != 0:
            sys.exit(rc)
    if args.deploy:
        deploy_cmd = ["chutes", "deploy", CHUTE_REF]
        if args.accept_fee:
            deploy_cmd.append("--accept-fee")
        rc = _run(deploy_cmd)
    sys.exit(rc)


if __name__ == "__main__":
    main()
