"""Build and deploy the glyph eval chutes (DESIGN §6, build step 11).

Stands up the SEPARATE compressor and decompressor evaluation chutes (separate containers so a
codec cannot stash the raw input during compress and read it back during decompress, #14).

chutes >=0.6 loads a chute by a FLAT ``module:chute`` ref and reads ``<cwd>/<module>.py`` (it
rejects dotted refs like ``eval.chute_app:...``). The eval chute lives at ``src/eval/chute_app.py``
and uses package imports, so this builds a small context dir -- ``chute_app.py`` at the root plus
``src/`` (for its ``from eval.../core...`` imports and the image's ``add("src", ...)``) -- and runs
the chutes CLI there with the flat ref ``chute_app:compressor_chute`` / ``:decompressor_chute``.

Both chutes share one image (``build_image``), so ``--build`` builds it once; ``--deploy`` deploys
each chute. Deploys are public + TEE (the only mode Chutes currently allows for these), which is
also what lets validators invoke them cross-account.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]  # .../src
CHUTE_MODULE = "chute_app"
CHUTE_NAMES = ("compressor_chute", "decompressor_chute")
CHUTE_REFS = tuple(f"{CHUTE_MODULE}:{name}" for name in CHUTE_NAMES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and deploy the glyph eval chutes")
    parser.add_argument("--build", action="store_true", help="Run `chutes build` (once; shared image)")
    parser.add_argument("--deploy", action="store_true", help="Run `chutes deploy` for each chute")
    parser.add_argument("--public", action="store_true", help="Public build/deploy (TEE; default for Chutes)")
    parser.add_argument("--accept-fee", action="store_true", help="Accept the deploy fee non-interactively")
    return parser


def _build_context() -> str:
    """A temp dir chutes can load the flat ``chute_app`` ref from: chute_app.py + src/."""

    ctx = tempfile.mkdtemp(prefix="glyph-eval-deploy-")
    shutil.copytree(SRC_DIR, os.path.join(ctx, "src"))
    shutil.copy(SRC_DIR / "eval" / "chute_app.py", os.path.join(ctx, "chute_app.py"))
    return ctx


def _run(argv: list[str], cwd: str, env: dict[str, str]) -> int:
    print("+", " ".join(argv), f"(cwd={cwd})")
    return subprocess.run(argv, cwd=cwd, env=env).returncode


def main() -> None:
    args = build_parser().parse_args()

    if not (args.build or args.deploy):
        print("Glyph eval chutes (separate compressor + decompressor):")
        for ref in CHUTE_REFS:
            print("  ", ref)
        print("Re-run with --build and/or --deploy [--public --accept-fee] to execute these for you.")
        return

    ctx = _build_context()
    env = {**os.environ, "PYTHONPATH": "src"}  # so chute_app.py's from eval/core imports resolve

    rc = 0
    if args.build:
        # Both chutes share one image -> build it once via the first ref.
        build_cmd = ["chutes", "build", CHUTE_REFS[0], "--wait"]
        if args.public:
            build_cmd.append("--public")
        rc = _run(build_cmd, ctx, env)
        if rc != 0:
            sys.exit(rc)
    if args.deploy:
        for ref in CHUTE_REFS:
            deploy_cmd = ["chutes", "deploy", ref]
            if args.public:
                deploy_cmd.append("--public")
            if args.accept_fee:
                deploy_cmd.append("--accept-fee")
            rc = _run(deploy_cmd, ctx, env)
            if rc != 0:
                sys.exit(rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()
