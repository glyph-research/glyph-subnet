"""issue #80: bt.logging's own default state suppresses .info() (only warning/error show),
which would silently drop most console output migrated off print() -- configure_logging must
make INFO the baseline unless the operator explicitly asked for more verbosity."""

from bittensor.utils.btlogging import logging as bt_logging

from core.log_config import add_logging_args, configure_logging


def _args(**overrides):
    defaults = {"logging.debug": False, "logging.trace": False}
    return type("Args", (), {**defaults, **overrides})()


def test_configure_logging_defaults_to_info_when_no_flags_passed():
    bt_logging.set_warning()  # start from a state where .info() would be suppressed
    configure_logging(_args())
    assert bt_logging.current_state.id == "Info"


def test_configure_logging_respects_debug_flag():
    bt_logging.set_warning()
    configure_logging(_args(**{"logging.debug": True}))
    assert bt_logging.current_state.id == "Debug"


def test_configure_logging_respects_trace_flag():
    bt_logging.set_warning()
    configure_logging(_args(**{"logging.trace": True}))
    assert bt_logging.current_state.id == "Trace"


def test_add_logging_args_registers_standard_bittensor_flags():
    import argparse

    parser = argparse.ArgumentParser()
    add_logging_args(parser)
    args = parser.parse_args(["--logging.debug"])
    assert getattr(args, "logging.debug") is True
