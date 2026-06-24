import pytest

from core.commitments import (
    CodecCommitment,
    commitment_digest,
    parse_commit_phase_by_hotkey,
    parse_commit_phase_digest,
    parse_commitment,
    parse_commitments_by_hotkey,
    serialize_commit_phase,
    serialize_commitment,
    serialize_reveal_phase,
)


def test_commitment_digest_is_deterministic_and_input_sensitive():
    d = commitment_digest("ns/repo", "abc123", "deadbeef")
    assert d == commitment_digest("ns/repo", "abc123", "deadbeef")
    assert d != commitment_digest("ns/repo", "abc123", "feedbeef")  # salt matters
    assert d != commitment_digest("ns/other", "abc123", "deadbeef")  # repo matters
    assert d != commitment_digest("ns/repo", "xyz999", "deadbeef")  # rev matters


def test_commit_phase_hides_repo_until_reveal():
    salt = "0011223344556677"
    digest = commitment_digest("secret/repo", "rev12345", salt)
    commit_value = serialize_commit_phase(digest)
    # The commit-phase value leaks nothing about the repo/rev.
    assert "secret/repo" not in commit_value
    assert "rev12345" not in commit_value
    assert parse_commit_phase_digest(commit_value) == digest
    # The reveal opens to exactly the committed digest.
    reveal_value = serialize_reveal_phase("secret/repo", "rev12345", salt)
    commitment, parsed_salt = parse_commitment(reveal_value)
    assert (commitment.repo, commitment.rev, parsed_salt) == ("secret/repo", "rev12345", salt)
    assert commitment_digest(commitment.repo, commitment.rev, parsed_salt) == digest


def test_parse_commitment_handles_legacy_and_rejects_commit_phase():
    legacy, salt = parse_commitment(serialize_commitment(CodecCommitment(repo="a/b", rev="abc123")))
    assert (legacy.repo, legacy.rev, salt) == ("a/b", "abc123", None)
    with pytest.raises(ValueError):
        parse_commitment(serialize_commit_phase("deadbeef"))


def test_parse_commit_phase_digest_returns_none_for_revealed():
    assert parse_commit_phase_digest(serialize_reveal_phase("a/b", "abc123", "s")) is None
    assert parse_commit_phase_digest("g1|a/b|abc123") is None


def test_parse_by_hotkey_splits_phases_and_populates_digest():
    salt = "abcdef0123456789"
    raw = {
        "hk_reveal": serialize_reveal_phase("ns/codec", "rev123456", salt),
        "hk_commit": serialize_commit_phase(commitment_digest("ns/codec", "rev123456", salt)),
        "hk_legacy": "g1|other/codec|def456",
        "hk_empty": "",
    }
    # Commit-phase entries are not returned as revealed commitments...
    revealed = {p.hotkey: p for p in parse_commitments_by_hotkey(raw)}
    assert set(revealed) == {"hk_reveal", "hk_legacy"}
    # ...but are surfaced by the commit-phase parser.
    assert set(parse_commit_phase_by_hotkey(raw)) == {"hk_commit"}

    # A revealed commitment carries the digest binding it to its commit phase.
    assert revealed["hk_reveal"].digest == commitment_digest("ns/codec", "rev123456", salt)
    assert revealed["hk_reveal"].digest == parse_commit_phase_by_hotkey(raw)["hk_commit"]
    # Legacy commitments have no commit-phase binding.
    assert revealed["hk_legacy"].digest is None


def test_repo_and_rev_reject_pipe_to_keep_wire_format_unambiguous():
    with pytest.raises(ValueError):
        CodecCommitment(repo="a|b/c", rev="abc123")
    with pytest.raises(ValueError):
        CodecCommitment(repo="a/b", rev="ab|c123")
