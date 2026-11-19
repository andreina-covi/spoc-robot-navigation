"""What the collector exports per detection.

The collector applies no visibility threshold: every non-structural detection goes to
the navigation CSV with its visibility metrics, and cm-benchmark filters later.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from collector import Collector
from utils.constants.object_constants import (
    has_named_object_id,
    is_exportable_object,
    is_structural_object,
)


@pytest.fixture
def collector(tmp_path, monkeypatch):
    monkeypatch.setattr("collector.OBJAVERSE_NAVIGATION_PATH", str(tmp_path))
    return Collector(scene_name="house_000001")


class TestObjectIdFilters:
    def test_wall_and_floor_are_structural(self):
        assert is_structural_object(obj_type="Wall")
        assert is_structural_object(obj_type="Floor")
        assert is_structural_object(object_id="Wall|+00.00|+00.00|+00.00")
        assert is_structural_object(object_id="Floor|0|0")

    def test_cup_is_not_structural(self):
        assert not is_structural_object(obj_type="Cup", object_id="Cup|1|2")

    def test_named_object_ids(self):
        assert has_named_object_id("Cup|2|10")
        assert has_named_object_id("door|1|2")
        assert has_named_object_id("ObjaPouch|2|1")
        assert not has_named_object_id("2|4")
        assert not has_named_object_id("4")
        assert not has_named_object_id("")
        assert not has_named_object_id(None)

    def test_exportable_keeps_numeric_ids_and_drops_structural(self):
        assert is_exportable_object(obj_type="Cup", object_id="Cup|1|2")
        assert is_exportable_object(obj_type="Painting", object_id="2|4")
        assert is_exportable_object(obj_type="Television", object_id="6|5")
        assert not is_exportable_object(obj_type="Wall", object_id="Wall|0|0")
        assert not is_exportable_object(obj_type="Floor", object_id="6|5")


class TestFilterExportDetections:
    def test_keeps_tiny_and_numeric_ids_and_drops_structural(self, collector):
        detections = {
            "Cup|1|2": [10, 10, 11, 11],  # 1 px box: no size threshold
            "6|5": [0, 0, 50, 50],
            "Wall|0|0": [0, 0, 100, 100],
            "Floor|0|0": [0, 0, 100, 100],
        }
        assert set(collector.filter_export_detections(detections)) == {"Cup|1|2", "6|5"}

    def test_empty(self, collector):
        assert collector.filter_export_detections({}) == {}
        assert collector.filter_export_detections(None) == {}


class TestVisiblePixels:
    COLOR = (10, 20, 30)

    def _event(self, mask):
        seg = np.zeros(mask.shape + (3,), dtype=np.uint8)
        seg[mask] = self.COLOR
        return SimpleNamespace(instance_segmentation_frame=seg)

    def test_counts_mask_pixels_inside_box(self, collector):
        mask = np.zeros((20, 30), dtype=bool)
        mask[2:6, 3:8] = True  # 20 pixels
        mask[15:18, 25:28] = True  # outside the box below
        event = self._event(mask)
        assert collector.get_visible_pixels_from_bbox(event, [0, 0, 10, 10], self.COLOR) == 20

    def test_mark_is_the_most_interior_mask_pixel(self, collector):
        mask = np.zeros((20, 30), dtype=bool)
        mask[4:11, 6:13] = True  # 7x7 square, centre (row 7, col 9)
        event = self._event(mask)
        pixels, mark_x, mark_y = collector.get_visible_pixels_from_bbox(
            event, [0, 0, 30, 20], self.COLOR, with_mark=True
        )
        assert pixels == 49
        assert (mark_x, mark_y) == (9, 7)

    def test_no_mask_pixels_has_no_mark(self, collector):
        event = self._event(np.zeros((20, 30), dtype=bool))
        assert collector.get_visible_pixels_from_bbox(
            event, [0, 0, 30, 20], self.COLOR, with_mark=True
        ) == (0, None, None)


def test_navigation_csv_has_visibility_metrics_and_no_filter_flag(collector):
    cols = collector.get_dict_navigation()
    for col in (
        "visible-pixels",
        "bbox-area",
        "min-side",
        "occupancy-ratio",
        "mark-x",
        "mark-y",
        "in-frame-fraction",
        "expected-bbox-area",
        "scripted-glance",
        "step-source",
    ):
        assert col in cols
    assert "passes-visibility" not in cols
