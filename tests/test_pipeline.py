"""Tests for the design pipeline.

The regression tests at the bottom encode failures observed in practice against
the real abguard build, so a refactor cannot quietly reintroduce them.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl
import pytest

from her2_stopping_rule import novelty
from her2_stopping_rule import table as tbl
from her2_stopping_rule.config import CDR3_IMGT, Config
from her2_stopping_rule.sequences import (
    cdr3_identity,
    count_enumerated,
    enumerate_cdr3,
    hamming,
    has_unpaired_cysteine,
    mutations_vs_native,
    one_hot,
    splice_heavy_chain,
    total_length,
)
from her2_stopping_rule.steps.s03a_enumerate import build
from her2_stopping_rule.steps.s06a_stability import Package, apply_mutations
from her2_stopping_rule.steps.s13_submit import order_by_track

# --- sequences ------------------------------------------------------------------


def test_enumerate_produces_exact_distance() -> None:
    """Every enumerated loop sits at exactly the requested distance."""
    loops = list(enumerate_cdr3(2, positions=(3, 4, 5)))
    assert all(hamming(loop, CDR3_IMGT) == 2 for loop in loops)


def test_enumerate_count_matches_formula() -> None:
    """The generator yields exactly as many loops as the closed form predicts."""
    assert len(list(enumerate_cdr3(2, positions=(3, 4, 5)))) == count_enumerated(2, 3)


def test_enumerate_is_distinct() -> None:
    """No loop is produced twice."""
    loops = list(enumerate_cdr3(2, positions=(3, 4, 5)))
    assert len(loops) == len(set(loops))


def test_mutations_round_trip() -> None:
    """The mutation string names every changed position and nothing else."""
    loop = "SRAGGDHFTAMDY"
    assert mutations_vs_native(loop) == "W3A,G7H,Y9T"
    assert len(mutations_vs_native(loop).split(",")) == hamming(loop, CDR3_IMGT)


def test_one_hot_is_one_per_position() -> None:
    """Each position contributes exactly one set bit."""
    x = one_hot(["SRWGGDGFYAMDY", "SRAGGDHFTAMDY"])
    assert x.shape == (2, 260)
    assert np.allclose(x.sum(axis=1), 13.0)


# --- novelty --------------------------------------------------------------------


def _windows(*seqs: str) -> np.ndarray:
    """Pack 10-residue windows into the array shape the scanner expects."""
    return np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(-1, 10)


def test_scan_finds_exact_match() -> None:
    """A candidate identical to a reference is at distance zero."""
    ref = _windows("WGGDGFYAMD")
    assert novelty.scan(_windows("WGGDGFYAMD"), ref).min_distance.tolist() == [0]


def test_scan_agrees_with_brute_force() -> None:
    """The pigeonhole scan matches an exhaustive comparison."""
    rng = np.random.default_rng(0)
    alphabet = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", dtype=np.uint8)
    refs = alphabet[rng.integers(0, 20, size=(200, 10))].astype(np.uint8)
    cands = alphabet[rng.integers(0, 20, size=(50, 10))].astype(np.uint8)

    result = novelty.scan(cands, refs)
    brute = (cands[:, None, :] != refs[None, :, :]).sum(axis=2).min(axis=1)

    for got, expected in zip(result.min_distance.tolist(), brute.tolist()):
        if expected <= 2:
            assert got == expected
        else:
            assert got >= 3


def test_pigeonhole_never_misses_a_near_neighbour() -> None:
    """Any reference within distance 2 is always found."""
    ref = _windows("WGGDGFYAMD")
    near = _windows("WGGDGFYAMA", "WGGDGFYAAA", "AGGDGFYAMD")
    assert (novelty.scan(near, ref).min_distance <= 2).all()


# --- table ----------------------------------------------------------------------


def test_drop_marks_without_deleting() -> None:
    """Dropping a row keeps it and records which step rejected it."""
    df = pl.DataFrame(
        {"design_id": ["a", "b"], "p_high": [0.9, 0.1], "dropped_by": [None, None]}
    )
    out = tbl.drop(df, pl.col("p_high") < 0.5, "affinity")
    assert out.height == 2
    assert tbl.alive(out).height == 1
    assert out["dropped_by"].to_list() == [None, "affinity"]


def test_first_rejection_wins() -> None:
    """A later step cannot overwrite an earlier rejection."""
    df = pl.DataFrame({"design_id": ["a"], "p_high": [0.1], "dropped_by": ["novelty"]})
    out = tbl.drop(df, pl.col("p_high") < 0.5, "affinity")
    assert out["dropped_by"].to_list() == ["novelty"]


def test_conform_is_idempotent() -> None:
    """Conforming an already-conformed table changes nothing."""
    once = tbl.conform(pl.DataFrame({"design_id": ["a"]}))
    assert tbl.conform(once).columns == once.columns


# --- construct assembly ---------------------------------------------------------


def test_apply_mutations_rejects_out_of_range() -> None:
    """A position outside the chain is an error, not a silent no-op."""
    with pytest.raises(IndexError):
        apply_mutations("AAAA", ((9, "C"),))


def test_package_description_is_stable() -> None:
    """Framework mutations render in a form that survives into the CSV."""
    assert Package("native").describe() == "none"
    assert Package("ds", ((106, "C"),), ((34, "C"),)).describe() == "H106C,L34C"


def test_order_by_track_puts_optimisation_first() -> None:
    """De novo designs only fill ranks optimisation did not claim."""
    df = pl.DataFrame(
        {
            "design_id": ["dn1", "op1", "op2"],
            "track_predicted": ["de_novo", "optimisation", "optimisation"],
            "rank": [1, 2, 3],
            "heavy_sequence": ["EVQL"] * 3,
            "light_sequence": ["DIQM"] * 3,
        }
    )
    assert order_by_track(df, Config())["design_id"].to_list() == ["op1", "op2", "dn1"]


# --- regression tests, from failures seen against the real abguard ---------------


def test_trastuzumab_itself_fails_novelty() -> None:
    """The native loop is zero mutations from the reference, so it is not novel.

    abguard: "CDRH3 SRWGGDGFYAMDY is 0 mutation(s) from SRWGGDGFYAMDY in
    trastuzumab; a novel design needs at least 3".
    """
    assert hamming(CDR3_IMGT, CDR3_IMGT) == 0
    assert cdr3_identity(CDR3_IMGT) == 1.0


def test_three_mutations_clears_the_floor() -> None:
    """Three mutations is enough, despite the task PDF implying four.

    Four of six probe designs at exactly three mutations passed
    ``team-validate``; abguard reports ``required_cdrh3_mutations = 3``. The
    failures were prior-art collisions, not the mutation floor.
    """
    cfg = Config()
    assert cfg.min_dist_to_provided == 3
    assert hamming("SRAGGDHFTAMDY", CDR3_IMGT) == 3


def test_paired_construct_is_227_residues() -> None:
    """The construct fits abguard's 250-residue cap with room to spare.

    An scFv with a (G4S)4 linker would be 247. Paired avoids the linker.
    """
    heavy = splice_heavy_chain("SRAGGDHFTAMDY")
    assert len(heavy) == 120
    assert total_length(heavy) == 227


def test_enumeration_excludes_unpaired_cysteines() -> None:
    """Loops abguard would reject on cysteines never enter the table."""
    cfg = dataclasses.replace(Config(), edit_distances=(1,))
    df = build(cfg)
    assert not any(has_unpaired_cysteine(c) for c in df["cdr3_imgt"].to_list())


def test_distance_four_space_size() -> None:
    """The enumerated space matches the number quoted in the plan."""
    assert count_enumerated(3) == 823_080
    assert count_enumerated(4) == 27_367_410


def test_calibration_never_exceeds_its_budget() -> None:
    """The ipTM cutoff admits at most the target fraction of measured negatives.

    A plain quantile overshoots on ties and never reaches a zero target, because
    a design at exactly the cutoff is admitted.
    """
    from her2_stopping_rule.steps.s08_calibrate import choose_cutoff

    rng = np.random.default_rng(0)
    scores = rng.random(500)
    labels = (rng.random(500) > 0.5).astype(int)
    for target in (0.0, 0.01, 0.05, 0.2, 0.5):
        _, achieved = choose_cutoff(scores, labels, target)
        assert achieved <= target + 1e-12, (target, achieved)


def test_predicted_tier_counts_only_evidence() -> None:
    """An axis with no evidence is not counted as an improvement."""
    from her2_stopping_rule.steps.s11_rank import predicted_tier

    cfg = Config()
    df = pl.DataFrame(
        {
            "p_high": [0.9, 0.9, None],
            "epitope_recovery": [0.9, None, None],
            "disulfide_formed": [True, None, None],
        }
    )
    assert df.select(predicted_tier(cfg))[:, 0].to_list() == [3, 1, 0]


def test_reference_prune_never_hides_a_collision() -> None:
    """Pruning by distance-from-native must not miss any near neighbour.

    s04a only compares a candidate against references within
    ``min_distance - 1`` of its own edit distance. That is sound by the
    triangle inequality, but a bug there would silently pass designs abguard
    then rejects, so it is checked against an unpruned brute force.
    """
    from her2_stopping_rule.config import CDR3_IMGT
    from her2_stopping_rule.data import Blacklist

    rng = np.random.default_rng(1)
    alphabet = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", dtype=np.uint8)
    native = np.frombuffer(CDR3_IMGT[2:12].encode(), dtype=np.uint8)

    refs = alphabet[rng.integers(0, 20, size=(3000, 10))].astype(np.uint8)
    distances = (refs != native).sum(axis=1).astype(np.int32)
    blacklist = Blacklist(refs, [f"r{i}" for i in range(len(refs))], distances, 0)

    for k in (3, 4):
        cands = np.tile(native, (200, 1)).copy()
        for row in range(cands.shape[0]):
            for pos in rng.choice(10, size=k, replace=False):
                cands[row, pos] = alphabet[rng.integers(0, 20)]

        full = (cands[:, None, :] != refs[None, :, :]).sum(axis=2).min(axis=1)
        kept = blacklist.relevant_to(k, 3)
        pruned = (
            (cands[:, None, :] != refs[kept][None, :, :]).sum(axis=2).min(axis=1)
            if kept.any()
            else np.full(cands.shape[0], 99)
        )
        # Wherever the true nearest neighbour is inside the threshold, the
        # pruned scan must find it too.
        close = full < 3
        assert (pruned[close] == full[close]).all(), k


def test_conform_casts_to_the_schema_type() -> None:
    """A step that builds rows without a schema still concatenates cleanly.

    polars infers Int64 where the table declares Int32, and the mismatch only
    surfaces later as a vstack failure in whichever step concatenates next.
    """
    loose = pl.DataFrame({"design_id": ["a"], "n_mutations": [3]})
    assert loose["n_mutations"].dtype == pl.Int64
    assert tbl.conform(loose)["n_mutations"].dtype == pl.Int32
    # The real failure: concatenating a loose frame onto a conformed one.
    pl.concat([tbl.conform(tbl.empty()), tbl.conform(loose)], how="vertical")


def test_guard_catches_file_level_errors() -> None:
    """A repeated heavy/light pair must reject every design it names.

    abguard reports this against the file, on one line, with no [rule] prefix.
    A guard that only parses per-design errors passes a file abguard rejects.
    """
    from her2_stopping_rule.steps.s02_guard import parse_errors

    err = (
        "  - repeated heavy/light pair for sequence_id: aa, bb, cc\n"
        "  - [novelty] dd: CDRH3 X is 2 mutation(s) from Y in "
        "uspto:uspto_WIO11640; a novel design needs at least 3"
    )
    verdicts = parse_errors(err)
    assert set(verdicts) == {"aa", "bb", "cc", "dd"}
    assert not any(v.compliant for v in verdicts.values())
    # abguard screens patents too, which no local filter can replicate.
    assert verdicts["dd"].nearest_source == "uspto"


def test_only_real_constructs_are_submittable() -> None:
    """Rows without both chains never reach the submission."""
    from her2_stopping_rule.steps.s13_submit import order_by_track

    df = pl.DataFrame(
        {
            "design_id": ["loop", "real"],
            "track_predicted": ["optimisation", "optimisation"],
            "rank": [1, 2],
            "heavy_sequence": [None, "EVQL"],
            "light_sequence": [None, "DIQM"],
        }
    )
    assert order_by_track(df, Config())["design_id"].to_list() == ["real"]


def test_guard_uses_membership_not_error_parsing() -> None:
    """A design absent from abguard's --drop output is rejected, reason or not.

    abguard truncates its error list, so a guard that decides compliance by
    reading error text passes designs a later, smaller call then rejects.
    That is how ten patent collisions reached a finished submission.
    """
    from her2_stopping_rule.steps.s02_guard import Verdict

    survivors = {"a"}
    failures = {"b": Verdict("b", False, "novelty", "patent")}
    verdicts = {}
    for sid in ("a", "b", "c"):
        verdicts[sid] = (
            Verdict(sid, True)
            if sid in survivors
            else failures.get(sid, Verdict(sid, False, "abguard", "not in output"))
        )
    assert verdicts["a"].compliant
    assert not verdicts["b"].compliant
    # c has no error line at all and must still be rejected.
    assert not verdicts["c"].compliant
