import pytest

from core.round_corpus import (
    SOURCE_SPEC,
    ChunkLocator,
    derive_round_corpus,
    resolve_position,
    round_seed,
    select_index,
)


def test_round_corpus_shape_matches_pinned_recipe():
    locators = derive_round_corpus("0xbeacon")
    # 3 FineWeb + 3 Pile + 2 enwik9 = 8 chunks, in source order.
    assert [loc.source for loc in locators] == (
        ["fineweb"] * 3 + ["pile"] * 3 + ["enwik9"] * 2
    )
    assert all(isinstance(loc, ChunkLocator) for loc in locators)
    # Each carries its pinned immutable revision.
    revs = {s.name: s.revision for s in SOURCE_SPEC}
    assert all(loc.revision == revs[loc.source] for loc in locators)


def test_same_beacon_is_byte_identical_across_validators():
    # Two independent derivations (two validators) for the same beacon must match exactly --
    # this is the consensus guarantee, and proves there is no hidden per-validator input.
    a = derive_round_corpus("0xdeadbeef")
    b = derive_round_corpus("0xdeadbeef")
    assert a == b


def test_consecutive_beacons_give_different_corpus():
    a = derive_round_corpus("0xblock_1000")
    b = derive_round_corpus("0xblock_1001")
    assert a != b
    # Every chunk's seed differs between the two beacons.
    assert all(x.seed_hex != y.seed_hex for x, y in zip(a, b))


def test_round_seed_is_deterministic_and_input_sensitive():
    s = round_seed("0xb", "fineweb", 0)
    assert s == round_seed("0xb", "fineweb", 0)
    assert s != round_seed("0xb", "fineweb", 1)  # chunk index matters
    assert s != round_seed("0xb", "pile", 0)  # source matters
    assert s != round_seed("0xc", "fineweb", 0)  # beacon matters


def test_select_index_in_range_and_deterministic():
    seed = round_seed("0xb", "fineweb", 0)
    assert select_index(seed, 7) == select_index(seed, 7)
    assert 0 <= select_index(seed, 7) < 7
    with pytest.raises(ValueError):
        select_index(seed, 0)


def test_resolve_position_deterministic_and_bounded():
    seed = round_seed("0xb", "pile", 2)
    f, rg, off = resolve_position(seed, num_files=20, num_row_groups=100)
    assert (f, rg, off) == resolve_position(seed, 20, 100)
    assert 0 <= f < 20 and 0 <= rg < 100
    assert isinstance(off, int)
    with pytest.raises(ValueError):
        resolve_position(seed, 0, 100)


def test_selection_spreads_across_files_over_many_beacons():
    # Sanity: different beacons land on a variety of files (not all the fixed first file).
    picks = {
        resolve_position(round_seed(f"0xblk{i}", "fineweb", 0), num_files=50, num_row_groups=10)[0]
        for i in range(200)
    }
    assert len(picks) > 10  # well-spread, not collapsed to one file
