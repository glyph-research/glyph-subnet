"""Shared bt.logging setup for the validator/weight_setter/reign_worker services (issue #80).

``bittensor.utils.btlogging.logging`` is a process-wide singleton whose default state
suppresses ``.info()`` -- only ``.warning()``/``.error()`` are visible until something raises
the level. That default would silently drop most of what these services log (round state,
version checks, etc.), which is the opposite of what replacing ``print()`` with this is for.
``configure_logging`` makes INFO the baseline (matching prior ``print()`` visibility) unless
the operator asked for more via ``--logging.debug``/``--logging.trace``.
"""

from __future__ import annotations

import argparse

from bittensor.utils.btlogging import logging as bt_logging


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    bt_logging.add_args(parser)


def configure_logging(args: argparse.Namespace) -> None:
    if getattr(args, "logging.trace", False):
        bt_logging.set_trace()
    elif getattr(args, "logging.debug", False):
        bt_logging.set_debug()
    else:
        bt_logging.set_info()
