from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from next_steps.chatgpt_2026_08_23.tiny_overfit import (
    DEFAULT_SOURCE,
    EXPECTED_SOURCE_HASHES,
    audit_tiny_overfit_bundle,
    audit_tiny_overfit_source,
    build_tiny_overfit_bundle,
    compare_checkpoints,
    compare_memorization_details,
    load_audited_cell,
)
from next_steps.chatgpt_2026_08_23.train_tiny_overfit import prepare_run, tiny_overfit_spec
from wordle_lab.common import read_json, read_jsonl, sha256_file, write_jsonl


GENERAL_SHA256 = "911767a1feda7ab3320fb97ef6870e305447811f2359b038b50a5aea7168fd3b"
SINGLETON_SHA256 = "f2c4a92299a0bdf69e3c6fe034e2fced572903d138763c39991138a2a0c33266"
MANIFEST_SHA256 = "b01b108efd913edb99ffb7c116f7feb6b6404817044336926e0484c63012ac99"


def test_tiny_bundle_has_two_disjoint_exact_32_state_cells(tmp_path: Path):
    manifest = build_tiny_overfit_bundle(DEFAULT_SOURCE, tmp_path, force=True)
    general = read_jsonl(tmp_path / "general_32.jsonl")
    singletons = read_jsonl(tmp_path / "singleton_32.jsonl")
    assert len(general) == len(singletons) == 32
    assert len({row["pair_id"] for row in general}) == 16
    assert all(row["posterior_size"] == 1 for row in singletons)
    assert len({row["target_word"] for row in singletons}) == 32
    assert not {row["state_id"] for row in general} & {row["state_id"] for row in singletons}
    assert manifest["locked_test_access"] is False
    assert manifest["source_rows_sha256"] == manifest["source_declared_rows_sha256"]


def test_contrast_pairs_change_feedback_and_required_action(tmp_path: Path):
    build_tiny_overfit_bundle(DEFAULT_SOURCE, tmp_path, force=True)
    rows = read_jsonl(tmp_path / "general_32.jsonl")
    pairs = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], []).append(row)
    for pair in pairs.values():
        assert len(pair) == 2
        left, right = pair
        assert left["history"][0]["guess"] == right["history"][0]["guess"]
        assert left["history"][0]["feedback"] != right["history"][0]["feedback"]
        assert left["target_word"] != right["target_word"]
        assert left["feedback_hamming_distance"] == right["feedback_hamming_distance"]


def test_tiny_manifest_hashes_the_emitted_rows(tmp_path: Path):
    manifest = build_tiny_overfit_bundle(DEFAULT_SOURCE, tmp_path, force=True)
    assert manifest["cells"]["general_32"]["sha256"] == sha256_file(tmp_path / "general_32.jsonl")
    assert manifest["cells"]["singleton_32"]["sha256"] == sha256_file(tmp_path / "singleton_32.jsonl")
    assert manifest["cells"]["general_32"]["sha256"] == GENERAL_SHA256
    assert manifest["cells"]["singleton_32"]["sha256"] == SINGLETON_SHA256
    assert sha256_file(tmp_path / "manifest.json") == MANIFEST_SHA256


def test_source_and_bundle_audit_recompute_all_training_provenance(tmp_path: Path):
    source_audit = audit_tiny_overfit_source(DEFAULT_SOURCE)
    assert source_audit["hashes"] == EXPECTED_SOURCE_HASHES
    assert source_audit["training_rows"] == 512
    assert source_audit["feedback_rows_recomputed"] == 512
    assert source_audit["feedback_tiles_recomputed"] == 777
    assert source_audit["posterior_rows_recomputed"] == 512
    assert source_audit["target_legality_rows_recomputed"] == 512
    assert source_audit["train_dev_overlap"] == 0

    build_tiny_overfit_bundle(DEFAULT_SOURCE, tmp_path, force=True)
    audit = audit_tiny_overfit_bundle(DEFAULT_SOURCE, tmp_path)
    assert audit["status"] == "passed"
    assert audit["general_singleton_overlap"] == 0
    assert audit["selected_feedback_rows_recomputed"] == 64
    assert audit["selected_target_legality_rows_recomputed"] == 64
    assert audit["singleton_posteriors_recomputed"] == 32
    assert audit["locked_test_access"] is False


def test_source_hash_drift_and_emitted_feedback_tampering_fail_closed(tmp_path: Path):
    copied_source = tmp_path / "source"
    shutil.copytree(DEFAULT_SOURCE, copied_source)
    manifest_path = copied_source / "manifest.json"
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source hash mismatch for manifest.json"):
        audit_tiny_overfit_source(copied_source)

    bundle = tmp_path / "bundle"
    build_tiny_overfit_bundle(DEFAULT_SOURCE, bundle, force=True)
    rows = read_jsonl(bundle / "general_32.jsonl")
    rows[0]["history"][0]["feedback"] = "GGGGG"
    write_jsonl(bundle / "general_32.jsonl", rows)
    with pytest.raises(RuntimeError, match="feedback does not recompute"):
        audit_tiny_overfit_bundle(DEFAULT_SOURCE, bundle)


def test_singleton_recomputation_and_cell_disjointness_fail_closed(tmp_path: Path):
    bundle = tmp_path / "singleton-tamper"
    build_tiny_overfit_bundle(DEFAULT_SOURCE, bundle, force=True)
    rows = read_jsonl(bundle / "singleton_32.jsonl")
    rows[0]["target_word"] = "ZZZZZ"
    rows[0]["completion"] = [{"role": "assistant", "content": "Final answer: ZZZZZ"}]
    write_jsonl(bundle / "singleton_32.jsonl", rows)
    with pytest.raises(RuntimeError, match="target is not a legal"):
        audit_tiny_overfit_bundle(DEFAULT_SOURCE, bundle)

    bundle = tmp_path / "overlap-tamper"
    build_tiny_overfit_bundle(DEFAULT_SOURCE, bundle, force=True)
    general = read_jsonl(bundle / "general_32.jsonl")
    singletons = read_jsonl(bundle / "singleton_32.jsonl")
    singletons[0]["state_id"] = general[0]["state_id"]
    write_jsonl(bundle / "singleton_32.jsonl", singletons)
    with pytest.raises(RuntimeError, match="not disjoint"):
        audit_tiny_overfit_bundle(DEFAULT_SOURCE, bundle)


def _detail(row: dict, *, rank: int, probability: float, exact: bool) -> dict:
    return {
        "example_id": row["example_id"],
        "state_id": row["state_id"],
        "task": row["task"],
        "pair_id": row.get("pair_id"),
        "target_word": row["target_word"],
        "target_rank": rank,
        "target_probability": probability,
        "exact_match": exact,
    }


def test_paired_comparison_reports_rank_probability_and_exact_deltas(tmp_path: Path):
    build_tiny_overfit_bundle(DEFAULT_SOURCE, tmp_path, force=True)
    rows, universe, audit = load_audited_cell(tmp_path / "general_32.jsonl")
    assert len(universe) == 128
    assert audit["universe_sha256"] == EXPECTED_SOURCE_HASHES["universe.json"]
    base = [_detail(row, rank=4, probability=0.1, exact=False) for row in rows]
    adapter = [_detail(row, rank=1, probability=0.6, exact=True) for row in rows]
    joined, summary = compare_memorization_details(rows, base, adapter)
    assert all(row["target_rank_delta"] == -3 for row in joined)
    assert all(row["target_rank_improvement"] == 3 for row in joined)
    assert all(row["target_probability_delta"] == pytest.approx(0.5) for row in joined)
    assert all(row["exact_match_delta"] == 1 for row in joined)
    assert summary["deltas"]["mean_target_rank"] == -3
    assert summary["deltas"]["mean_target_rank_improvement"] == 3
    assert summary["deltas"]["mean_target_probability"] == pytest.approx(0.5)
    assert summary["deltas"]["natural_exact_accuracy"] == 1.0
    assert summary["state_changes"]["exact_gained"] == 32

    base_path = write_jsonl(tmp_path / "base-items.jsonl", base)
    adapter_path = write_jsonl(tmp_path / "adapter-items.jsonl", adapter)
    output_dir = tmp_path / "comparison"
    emitted = compare_checkpoints(
        tmp_path / "general_32.jsonl",
        output_dir,
        base_items_path=base_path,
        adapter_items_path=adapter_path,
    )
    assert emitted["mode"] == "consumed_precomputed_items"
    assert read_json(output_dir / "pre_post_summary.json")["deltas"]["mean_target_rank"] == -3
    assert len(read_jsonl(output_dir / "pre_post_items.jsonl")) == 32


def test_paired_comparison_rejects_nonidentical_state_sets(tmp_path: Path):
    build_tiny_overfit_bundle(DEFAULT_SOURCE, tmp_path, force=True)
    rows = read_jsonl(tmp_path / "singleton_32.jsonl")
    base = [_detail(row, rank=2, probability=0.2, exact=False) for row in rows]
    adapter = [_detail(row, rank=1, probability=0.8, exact=True) for row in rows[:-1]]
    with pytest.raises(RuntimeError, match="cover exactly"):
        compare_memorization_details(rows, base, adapter)


def test_training_spec_is_audited_and_deterministic_run_refuses_overwrite(tmp_path: Path):
    bundle = tmp_path / "bundle"
    build_tiny_overfit_bundle(DEFAULT_SOURCE, bundle, force=True)
    rows_path = bundle / "general_32.jsonl"
    spec = tiny_overfit_spec("general", rows_path, steps=1)
    assert spec["data"]["source_manifest_sha256"] == EXPECTED_SOURCE_HASHES["manifest.json"]
    assert spec["data"]["universe_sha256"] == EXPECTED_SOURCE_HASHES["universe.json"]
    run_dir = prepare_run(spec, rows_path, output_root=tmp_path / "runs")
    assert (run_dir / "spec.json").is_file()
    assert (run_dir / "dataset_manifest.json").is_file()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        prepare_run(spec, rows_path, output_root=tmp_path / "runs")
