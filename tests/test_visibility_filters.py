"""Unit tests for Collector visibility filters and object-id export rules."""

from types import SimpleNamespace

import numpy as np
import pytest

from collector import Collector
from utils.constants.object_constants import (
    has_named_object_id,
    is_exportable_object,
    is_structural_object,
)
from utils.constants.stretch_initialization_utils import (
    INTEL_CAMERA_HEIGHT,
    INTEL_CAMERA_WIDTH,
)


@pytest.fixture
def collector(tmp_path, monkeypatch):
    monkeypatch.setattr("collector.OBJAVERSE_NAVIGATION_PATH", str(tmp_path))
    return Collector(scene_name="house_000001", visibility_mode="fast")


@pytest.fixture
def strict_collector(tmp_path, monkeypatch):
    monkeypatch.setattr("collector.OBJAVERSE_NAVIGATION_PATH", str(tmp_path))
    return Collector(scene_name="house_000001", visibility_mode="strict")


def _bbox(cmin, rmin, cmax, rmax):
    return [cmin, rmin, cmax, rmax]


def _mock_controller(detections=None, frame_hw=(INTEL_CAMERA_HEIGHT, INTEL_CAMERA_WIDTH), masks=None):
    h, w = frame_hw
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    event = SimpleNamespace(
        frame=frame,
        instance_detections2D=detections or {},
        instance_masks=masks or {},
        metadata={
            "objects": [],
            "agent": {
                "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "cameraHorizon": 0.0,
            },
            "cameraPosition": {"x": 0.0, "y": 1.4, "z": 0.0},
        },
    )
    return SimpleNamespace(last_event=event)


class TestBBoxHelpers:
    def test_bbox_width_height_area(self):
        w, h, area = Collector.bbox_width_height_area(_bbox(10, 20, 50, 80))
        assert w == 40
        assert h == 60
        assert area == 2400

    def test_bbox_invalid_returns_zeros(self):
        assert Collector.bbox_width_height_area(None) == (0.0, 0.0, 0.0)
        assert Collector.bbox_width_height_area([1, 2]) == (0.0, 0.0, 0.0)

    def test_mask_pixel_count(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[0:2, 0:3] = True
        assert Collector.get_mask_pixel_count("Cup|1", {"Cup|1": mask}) == 6
        assert Collector.get_mask_pixel_count("missing", {"Cup|1": mask}) is None
        assert Collector.get_mask_pixel_count("Cup|1", None) is None


class TestMinMaskPixelsConstant:
    def test_min_mask_pixels_is_frame_times_fraction(self):
        expected = INTEL_CAMERA_WIDTH * INTEL_CAMERA_HEIGHT * Collector.DEFAULT_MIN_FRAME_FRACTION
        assert Collector.FRAME_SIZE_PX == INTEL_CAMERA_WIDTH * INTEL_CAMERA_HEIGHT
        assert abs(Collector.MIN_MASK_PIXELS - expected) < 1e-9

    def test_collector_inherits_min_mask_pixels(self, collector):
        assert abs(collector.min_mask_pixels - Collector.MIN_MASK_PIXELS) < 1e-9


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

    def test_exportable_requires_named_non_structural(self):
        assert is_exportable_object(obj_type="Cup", object_id="Cup|1|2")
        assert not is_exportable_object(obj_type="Painting", object_id="2|4")
        assert not is_exportable_object(obj_type="Wall", object_id="Wall|0|0")


class TestVisibilityMetrics:
    def test_metrics_from_bbox_only(self, collector):
        metrics = collector.compute_visibility_metrics(_bbox(0, 0, 20, 20))
        assert metrics["bbox_area"] == 400
        assert metrics["visible_area_px"] == 400
        assert metrics["used_mask"] is False
        assert metrics["unoccluded_ratio"] is None
        assert metrics["visible_frac"] == pytest.approx(400 / Collector.FRAME_SIZE_PX)

    def test_metrics_prefer_mask_pixels(self, collector):
        metrics = collector.compute_visibility_metrics(
            _bbox(0, 0, 40, 40), mask_pixels=250, silhouette_pixels=500
        )
        assert metrics["visible_area_px"] == 250
        assert metrics["used_mask"] is True
        assert metrics["unoccluded_ratio"] == pytest.approx(0.5)

    def test_unoccluded_ratio_capped_at_one(self, collector):
        metrics = collector.compute_visibility_metrics(
            _bbox(0, 0, 50, 50), mask_pixels=900, silhouette_pixels=500
        )
        assert metrics["unoccluded_ratio"] == 1.0


class TestPassesVisibilityFiltersFast:
    def test_accepts_large_square_bbox(self, collector):
        metrics = collector.compute_visibility_metrics(_bbox(0, 0, 30, 30))
        assert collector.passes_visibility_filters(metrics) is True

    def test_rejects_tiny_bbox(self, collector):
        metrics = collector.compute_visibility_metrics(_bbox(0, 0, 5, 5))
        assert collector.passes_visibility_filters(metrics) is False

    def test_rejects_thin_sliver(self, collector):
        metrics = collector.compute_visibility_metrics(_bbox(0, 0, 200, 5))
        assert metrics["bbox_area"] >= collector.min_mask_pixels
        assert collector.passes_visibility_filters(metrics) is False

    def test_rejects_below_frame_fraction(self, collector):
        metrics = collector.compute_visibility_metrics(
            _bbox(0, 0, 20, 12), frame_hw=(10000, 10000)
        )
        assert collector.passes_visibility_filters(metrics) is False


class TestPassesVisibilityFiltersStrict:
    def test_rejects_low_unoccluded_ratio(self, strict_collector):
        metrics = strict_collector.compute_visibility_metrics(
            _bbox(0, 0, 40, 40),
            mask_pixels=500,
            silhouette_pixels=2000,
        )
        assert metrics["unoccluded_ratio"] == pytest.approx(0.25)
        assert strict_collector.passes_visibility_filters(metrics) is False

    def test_accepts_sufficient_unoccluded_ratio(self, strict_collector):
        metrics = strict_collector.compute_visibility_metrics(
            _bbox(0, 0, 40, 40),
            mask_pixels=900,
            silhouette_pixels=1500,
        )
        assert strict_collector.passes_visibility_filters(metrics) is True

    def test_strict_without_silhouette_falls_back_to_area(self, strict_collector):
        metrics = strict_collector.compute_visibility_metrics(_bbox(0, 0, 40, 40))
        assert metrics["unoccluded_ratio"] is None
        assert strict_collector.passes_visibility_filters(metrics) is True


class TestFilterExportDetections:
    def test_export_keeps_tiny_named_objects(self, collector):
        detections = {
            "Cup|tiny": _bbox(0, 0, 3, 3),
            "Wall|+00.00|+00.00|+00.00": _bbox(0, 0, 100, 100),
            "2|4": _bbox(0, 0, 50, 50),
            "Mug|1": _bbox(10, 10, 50, 50),
        }
        kept = collector.filter_export_detections(detections)
        assert set(kept.keys()) == {"Cup|tiny", "Mug|1"}


class TestFilterRecognizableDetections:
    def test_filters_tiny_structural_and_numeric(self, collector):
        detections = {
            "Cup|1": _bbox(0, 0, 40, 40),
            "Cup|tiny": _bbox(0, 0, 3, 3),
            "Wall|+00.00|+00.00|+00.00": _bbox(0, 0, 100, 100),
            "2|4": _bbox(0, 0, 40, 40),
            "Sliver|1": _bbox(0, 0, 300, 4),
        }
        controller = _mock_controller(detections=detections)
        kept = collector.filter_recognizable_detections(detections, controller)
        assert set(kept.keys()) == {"Cup|1"}

    def test_empty_detections(self, collector):
        controller = _mock_controller(detections={})
        assert collector.filter_recognizable_detections({}, controller) == {}
        assert collector.filter_recognizable_detections(None, controller) == {}

    def test_fast_mode_does_not_need_masks(self, collector):
        detections = {"Mug|1": _bbox(10, 10, 50, 50)}
        controller = _mock_controller(detections=detections, masks={})
        kept = collector.filter_recognizable_detections(detections, controller)
        assert "Mug|1" in kept


class TestShownObjectFraction:
    def _obj_meta(self, size_xyz, distance):
        return {
            "objectId": "Table|1",
            "objectType": "DiningTable",
            "distance": distance,
            "axisAlignedBoundingBox": {
                "center": {"x": 0.0, "y": 0.5, "z": 2.0},
                "size": {"x": size_xyz[0], "y": size_xyz[1], "z": size_xyz[2]},
            },
        }

    def test_size_distance_estimate_scales_with_distance(self, collector):
        near = collector.estimate_full_silhouette_from_size_distance(
            self._obj_meta((1.0, 0.8, 1.5), distance=1.0)
        )
        far = collector.estimate_full_silhouette_from_size_distance(
            self._obj_meta((1.0, 0.8, 1.5), distance=4.0)
        )
        assert near is not None and far is not None
        assert near > far * 10

    def test_rejects_small_peek_of_large_object(self, collector):
        obj = self._obj_meta((2.0, 0.8, 1.2), distance=1.5)
        full = collector.estimate_full_silhouette_from_size_distance(obj)
        assert full is not None and full > 2000
        metrics = collector.compute_visibility_metrics(
            _bbox(0, 0, 25, 25), silhouette_pixels=full
        )
        assert metrics["unoccluded_ratio"] < collector.min_unoccluded_ratio
        assert collector.passes_visibility_filters(metrics) is False

    def test_accepts_mostly_visible_object(self, collector):
        obj = self._obj_meta((0.2, 0.2, 0.2), distance=2.0)
        full = collector.estimate_full_silhouette_from_size_distance(obj)
        assert full is not None
        visible = max(collector.min_mask_pixels, full * 0.85)
        side = int(np.ceil(np.sqrt(visible)))
        metrics = collector.compute_visibility_metrics(
            _bbox(0, 0, side, side), silhouette_pixels=full
        )
        assert metrics["unoccluded_ratio"] >= collector.min_unoccluded_ratio
        assert collector.passes_visibility_filters(metrics) is True

    def test_filter_uses_shown_fraction_with_metadata(self, collector):
        detections = {
            "DiningTable|1": _bbox(0, 0, 30, 30),
            "Cup|1": _bbox(0, 0, 40, 40),
        }
        objects_by_id = {
            "DiningTable|1": {
                **self._obj_meta((2.5, 0.9, 1.5), distance=1.2),
                "objectId": "DiningTable|1",
                "objectType": "DiningTable",
            },
            "Cup|1": {
                "objectId": "Cup|1",
                "objectType": "Cup",
                "distance": 1.5,
                "axisAlignedBoundingBox": {
                    "center": {"x": 0.0, "y": 1.0, "z": 1.5},
                    "size": {"x": 0.08, "y": 0.1, "z": 0.08},
                },
            },
        }
        controller = _mock_controller(detections=detections)
        kept = collector.filter_recognizable_detections(
            detections, controller, objects_by_id=objects_by_id
        )
        assert "Cup|1" in kept
        assert "DiningTable|1" not in kept


class TestNavigationCsvHasNoPassesVisibility:
    def test_dict_navigation_omits_passes_visibility(self, collector):
        cols = collector.get_dict_navigation()
        assert "passes-visibility" not in cols
        assert "visible-area-px" in cols
        assert "full-silhouette-px" in cols
