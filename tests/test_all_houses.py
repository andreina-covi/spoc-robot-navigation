import pytest

from online_evaluation.online_evaluation_types_and_utils import (
    normalized_eval_sample_to_task_spec,
    room_visit_house_sample,
)
from online_evaluation.online_evaluator import OnlineEvaluatorManager
from scripts.placement_trials import houses_from_samples
from tasks.multi_task_eval_sampler import SEEDED_START_ROTATIONS_DEG, seeded_start_candidates


def _grid():
    return [{"x": x * 0.25, "y": 0.9, "z": z * 0.25} for x in range(8) for z in range(8)]


def test_seeded_start_candidates_are_deterministic_and_order_free():
    grid = _grid()
    first = list(seeded_start_candidates(grid, 436))
    assert list(seeded_start_candidates(list(reversed(grid)), 436)) == first
    # every position at every heading, each pair once
    assert len(first) == len(grid) * len(SEEDED_START_ROTATIONS_DEG)
    assert len({(p["x"], p["z"], h) for p, h in first}) == len(first)
    assert {h for _, h in first} == set(SEEDED_START_ROTATIONS_DEG)
    firsts = {str(next(seeded_start_candidates(grid, i))) for i in range(20)}
    assert len(firsts) > 1  # different houses start in different places


def test_seeded_start_candidates_need_positions():
    with pytest.raises(RuntimeError):
        next(seeded_start_candidates([], 3))


def test_house_sample_becomes_room_visit_task_spec():
    sample = room_visit_house_sample(436, 4, "test")
    spec = normalized_eval_sample_to_task_spec(sample)
    assert spec["task_type"] == "RoomVisit"
    assert spec["house_index"] == 436
    assert spec["house_split"] == "test"
    assert spec["num_rooms_in_house"] == 4
    assert spec["eval_info"]["seeded_start"] is True
    assert spec["eval_info"]["task_path"].split("/")[-4:] == [
        "all_houses", "SimpleExploreHouse", "test", "000436"
    ]
    assert spec["natural_language_spec"] == (
        "go to all 4 rooms in the house. "
        "indicate when you have seen a new room and when you are done"
    )


def test_placement_houses_from_house_samples():
    samples = [room_visit_house_sample(i, 2, "test") for i in (5, 6, 7)]
    assert houses_from_samples(samples) == {"test": [5, 6, 7]}


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(eval_subset="test", list_of_tasks=["RoomVisit"]),
        dict(eval_subset="minival", list_of_tasks=["RoomVisit"], all_houses=True),
        dict(eval_subset="val", list_of_tasks=["ObjectNavType"], all_houses=True),
    ],
)
def test_manager_rejects_bad_subset_or_task(tmp_path, kwargs):
    with pytest.raises(ValueError):
        OnlineEvaluatorManager(outdir=str(tmp_path), **kwargs)


def _episode(root, name, split, index, placement=False, finished=True):
    import json

    ann = root / name / "annotations"
    ann.mkdir(parents=True)
    (ann / f"navigation-house_{index:06d}.csv").write_text("timestep\n")
    if finished:
        meta = {"split": split, "house_index": str(index)}
        (ann / f"episode_meta-house_{index:06d}.json").write_text(json.dumps(meta))
    if placement:
        (ann / f"placement_candidates-house_{index:06d}-{split}-radius_1.2.csv").write_text("x\n")
    return ann


def test_finished_episodes_and_partial_folders(tmp_path):
    from utils.export_progress import (
        finished_episodes,
        has_placement_trials,
        set_aside_partial_episodes,
    )

    done = _episode(tmp_path, "house_000001", "val", 1, placement=True)
    walked = _episode(tmp_path, "house_000002", "val", 2)
    _episode(tmp_path, "house_000003", "val", 3, finished=False)
    _episode(tmp_path, "house_000030", "val", 30, finished=False)

    found = finished_episodes(tmp_path)
    assert found == {("val", 1): done, ("val", 2): walked}
    assert has_placement_trials(done) and not has_placement_trials(walked)

    moved = set_aside_partial_episodes(tmp_path, 3)
    assert [p.name for p in moved] == [".partial-house_000003"]
    assert (tmp_path / "house_000030").is_dir()  # another house is untouched
    assert set_aside_partial_episodes(tmp_path, 1) == []  # finished folders stay


def test_load_houses_reads_once_in_given_order(tmp_path):
    import gzip
    import json

    from scripts.placement_trials import load_houses

    with gzip.open(tmp_path / "val.jsonl.gz", "wt") as handle:
        for i in range(10):
            handle.write(json.dumps({"id": i}) + "\n")
    assert [h["id"] for _, h in load_houses("val", [7, 2, 5], tmp_path)] == [7, 2, 5]
    with pytest.raises(IndexError):
        load_houses("val", [3, 12], tmp_path)
