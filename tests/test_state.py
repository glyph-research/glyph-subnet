"""Validator state persistence: the #180 improvement backfill and #187's crash-safety.

issue #180 -- the real improvement of a pre-#177 winner is recoverable from state we already
hold (each entry dethroned the next one down the retained history), so it is recomputed at
load time rather than defaulted. Only the oldest retained entry, whose opponent is no longer
retained, falls back to DEFAULT_WIN_MARGIN.

issue #187 -- saving must be atomic and loading must degrade rather than crash: a truncating
write left a 0-byte state file after an auto-update restart and the validator could not
start at all.
"""

import json

import core.state
from core.constants import DEFAULT_WIN_MARGIN, STATE_BACKUP_COUNT
from core.state import ValidatorState, _backup_paths, load_state, save_state
from core.weights import WinnerEntry, compute_weights, winner_share

# --- issue #180: backfill improvement for pre-#177 winners -------------------------------


def _write_state(path, entries, stamped=()):
    """Persist a winner_history whose entries omit `improvement` unless listed in
    `stamped` (index -> value), mimicking JSON written before #177."""

    history = []
    for index, (hotkey, ratio) in enumerate(entries):
        entry = {
            "hotkey": hotkey, "repo": f"{hotkey}/c", "revision": "rev123456",
            "ratio": ratio, "commit_block": 10,
        }
        if index in dict(stamped):
            entry["improvement"] = dict(stamped)[index]
        history.append(entry)
    path.write_text(json.dumps({"winner_history": history}))


def test_backfill_recovers_the_real_improvement_from_adjacent_ratios(tmp_path):
    # The live-state acceptance case from issue #180: uid 124 dethroned uid 122 by 5.56%,
    # which the 1%-default migration would have understated into a 40% share.
    path = tmp_path / "state.json"
    _write_state(path, [("uid124", 0.07199660936991374), ("uid122", 0.0762338638305664)])

    history = load_state(path).winner_history
    assert abs(history[0].improvement - 0.0555833) < 1e-6
    # 25 + 15*5.558 = 108% -> capped at the pot: the champion takes everything.
    assert winner_share(history[0], is_top_payee=True) > 1.0
    weights = compute_weights(
        ["burn-sink", "uid124", "uid122"], history, is_burn_tempo=False, burn_uid=0
    )
    assert weights == [0.0, 1.0, 0.0]


def test_the_oldest_retained_entry_keeps_the_margin_fallback(tmp_path):
    # It dethroned a winner we no longer retain, so there is nothing to compute from --
    # the one place DEFAULT_WIN_MARGIN legitimately applies.
    path = tmp_path / "state.json"
    _write_state(path, [("a", 0.40), ("b", 0.50)])
    assert load_state(path).winner_history[-1].improvement == DEFAULT_WIN_MARGIN


def test_a_stamped_improvement_is_never_recomputed(tmp_path):
    # A genuine 1% dethrone is legitimate and must survive the migration untouched, even
    # though the adjacent ratios would compute something else entirely.
    path = tmp_path / "state.json"
    _write_state(path, [("a", 0.40), ("b", 0.50)], stamped={0: 0.01})

    history = load_state(path).winner_history
    assert history[0].improvement == 0.01  # not the 20% the ratios would imply


def test_backfill_is_deterministic_and_survives_a_round_trip(tmp_path):
    # Every validator holds the same ratios, so the backfill lands identically; and once
    # written back the values are stamped, so a second load changes nothing.
    path = tmp_path / "state.json"
    _write_state(path, [("a", 0.40), ("b", 0.50), ("c", 0.80)])

    first = load_state(path)
    assert [round(e.improvement, 6) for e in first.winner_history] == [0.2, 0.375, DEFAULT_WIN_MARGIN]

    save_state(path, first)
    assert load_state(path).winner_history == first.winner_history


def test_a_worse_or_equal_ratio_backfills_to_zero_never_negative(tmp_path):
    # Defensive: history should never contain a winner worse than the one below it, but a
    # negative improvement would flip the sign of its share, so it clamps at 0.
    path = tmp_path / "state.json"
    _write_state(path, [("a", 0.60), ("b", 0.50)])
    assert load_state(path).winner_history[0].improvement == 0.0


# --- issue #187: atomic saves and crash-safe loads ---------------------------------------


def _state(hotkey="a", ratio=0.40, improvement=0.05):
    return ValidatorState(
        winner_history=[
            WinnerEntry(
                hotkey=hotkey, repo=f"{hotkey}/c", revision="rev1",
                ratio=ratio, commit_block=10, improvement=improvement,
            )
        ]
    )


def _capture_logs(monkeypatch):
    """Collect what an operator would actually see, so a silent recovery fails the test."""

    logged = []
    monkeypatch.setattr(core.state, "_log", lambda msg, level="warning": logged.append((level, msg)))
    return logged


def test_the_normal_save_load_round_trip_is_unchanged(tmp_path):
    path = tmp_path / "validator_state.json"
    state = _state()
    save_state(path, state)

    assert load_state(path).winner_history == state.winner_history
    # The temp file is consumed by the atomic replace, never left lying around.
    assert not (tmp_path / "validator_state.json.tmp").exists()


def test_a_save_that_dies_partway_leaves_the_previous_state_intact(tmp_path, monkeypatch):
    # The heart of #187: the old write_text truncated first, so a process death here left a
    # 0-byte file. Writing a sibling temp and replacing means the original is untouched.
    path = tmp_path / "validator_state.json"
    save_state(path, _state(hotkey="old", ratio=0.40))
    before = path.read_text()

    def die(*args, **kwargs):
        raise OSError("process killed mid-write")

    monkeypatch.setattr(core.state.os, "fsync", die)
    try:
        save_state(path, _state(hotkey="new", ratio=0.30))
    except OSError:
        pass

    assert path.read_text() == before  # not truncated, not empty
    assert load_state(path).winner_history[0].hotkey == "old"


def test_a_zero_byte_state_file_starts_instead_of_crashing(tmp_path, monkeypatch):
    # The exact production symptom: an empty file used to raise ValidationError out of
    # load_state and crash-loop the validator every round.
    logged = _capture_logs(monkeypatch)
    path = tmp_path / "validator_state.json"
    path.write_text("")

    assert load_state(path) == ValidatorState()
    assert logged and any(level == "error" for level, _ in logged)


def test_a_truncated_or_garbage_state_file_also_falls_back(tmp_path, monkeypatch):
    _capture_logs(monkeypatch)
    for damaged in ('{"winner_history": [{"hotkey": "a", "rep', "not json at all", "   "):
        path = tmp_path / "validator_state.json"
        path.write_text(damaged)
        assert load_state(path) == ValidatorState()


def test_recovery_prefers_the_newest_usable_backup_over_an_empty_state(tmp_path, monkeypatch):
    logged = _capture_logs(monkeypatch)
    path = tmp_path / "validator_state.json"
    save_state(path, _state(hotkey="older", ratio=0.50))
    save_state(path, _state(hotkey="newest", ratio=0.30))
    path.write_text("")  # the incident

    recovered = load_state(path)
    assert recovered.winner_history[0].hotkey == "newest"
    assert any("backup" in msg for _, msg in logged)


def test_a_corrupt_newest_backup_falls_through_to_an_older_one(tmp_path, monkeypatch):
    _capture_logs(monkeypatch)
    path = tmp_path / "validator_state.json"
    save_state(path, _state(hotkey="older", ratio=0.50))
    save_state(path, _state(hotkey="newest", ratio=0.30))
    _backup_paths(path)[0].write_text("")  # newest backup is damaged too
    path.write_text("")

    assert load_state(path).winner_history[0].hotkey == "older"


def test_saving_keeps_only_the_newest_n_backups(tmp_path):
    path = tmp_path / "validator_state.json"
    for index in range(STATE_BACKUP_COUNT + 3):
        save_state(path, _state(hotkey=f"h{index}", ratio=0.40))

    backups = _backup_paths(path)
    assert len(backups) == STATE_BACKUP_COUNT
    # Newest first, and the newest backup matches what is currently on disk.
    assert json.loads(backups[0].read_text()) == json.loads(path.read_text())


def test_a_backup_failure_never_fails_the_save(tmp_path, monkeypatch):
    # Backups are a convenience; the save that already succeeded must stand regardless.
    _capture_logs(monkeypatch)
    path = tmp_path / "validator_state.json"

    def die(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(core.state, "_rotate_backups", lambda p, payload: die())
    try:
        save_state(path, _state())
    except OSError:
        pass
    assert load_state(path).winner_history[0].hotkey == "a"


def test_backfill_still_runs_on_a_recovered_backup(tmp_path, monkeypatch):
    # Recovery must not bypass #180: a pre-#177 backup still gets its improvement recomputed.
    _capture_logs(monkeypatch)
    path = tmp_path / "validator_state.json"
    save_state(path, _state())
    backup = _backup_paths(path)[0]
    backup.write_text(json.dumps({"winner_history": [
        {"hotkey": "uid124", "repo": "u/c", "revision": "r", "ratio": 0.07199660936991374,
         "commit_block": 10},
        {"hotkey": "uid122", "repo": "u/c", "revision": "r", "ratio": 0.0762338638305664,
         "commit_block": 10},
    ]}))
    path.write_text("")

    assert abs(load_state(path).winner_history[0].improvement - 0.0555833) < 1e-6
