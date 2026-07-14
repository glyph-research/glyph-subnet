"""issue #116: the validator's private seed salt refreshes every round instead of being
generated once and persisted for the validator's lifetime -- a long-lived on-disk secret
would expose every future round's sampling if it ever leaked, and derive_seed's private
component should be as fresh as its public (block hash) one. The explicit --salt-file
override keeps the original read-and-reuse behavior for tests/manual reproducibility."""

from validator.service import _load_salt


def test_default_path_generates_a_fresh_salt_every_call(tmp_path):
    # The core regression guard: same state_dir, no --salt-file -> two different salts.
    a = _load_salt(tmp_path, None)
    b = _load_salt(tmp_path, None)
    assert a != b
    assert len(a) == 64 and len(b) == 64  # secrets.token_hex(32)


def test_default_path_still_writes_the_salt_file_for_observability(tmp_path):
    salt = _load_salt(tmp_path, None)
    path = tmp_path / "validator_salt.txt"
    assert path.read_text().strip() == salt
    # ...and the file tracks the newest salt on each subsequent call.
    newer = _load_salt(tmp_path, None)
    assert path.read_text().strip() == newer != salt


def test_default_path_ignores_a_preexisting_salt_file(tmp_path):
    # A leftover lifetime-salt file from before this change must not be read back and
    # resurrected -- that would silently reintroduce the fixed-forever behavior on upgrade.
    legacy = tmp_path / "validator_salt.txt"
    legacy.write_text("legacy-lifetime-salt")
    assert _load_salt(tmp_path, None) != "legacy-lifetime-salt"


def test_explicit_salt_file_is_reused_unchanged_across_calls(tmp_path):
    explicit = tmp_path / "fixed_salt.txt"
    a = _load_salt(tmp_path, str(explicit))
    b = _load_salt(tmp_path, str(explicit))
    assert a == b
    assert explicit.read_text().strip() == a


def test_explicit_salt_file_reads_an_existing_value(tmp_path):
    explicit = tmp_path / "fixed_salt.txt"
    explicit.write_text("known-reproducible-salt\n")
    assert _load_salt(tmp_path, str(explicit)) == "known-reproducible-salt"
