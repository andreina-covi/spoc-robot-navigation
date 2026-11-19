"""in-frame-fraction: share of an object's projected box inside the frame."""
from geometry import expected_bbox_from_3d


def _box(x0, x1, y0, y1, z0, z1):
    return [
        {"x": x, "y": y, "z": z}
        for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)
    ]


CAM = {"x": 0.0, "y": 0.0, "z": 0.0}


def test_box_fully_in_front_has_no_corner_behind():
    box = expected_bbox_from_3d(_box(-0.1, 0.1, -0.1, 0.1, 2.0, 2.2), CAM, 0.0, 0.0, 396, 224, 200)
    assert box["corners_behind"] == 0
    assert box["area"] == box["area_raw"]


def test_box_beside_the_camera_reports_corners_behind():
    # A sofa you walk past: its back half is behind the camera plane.
    box = expected_bbox_from_3d(_box(0.5, 1.5, -0.5, 0.0, -1.0, 1.0), CAM, 0.0, 0.0, 396, 224, 200)
    assert box is not None
    assert box["corners_behind"] == 4
