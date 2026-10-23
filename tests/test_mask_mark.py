import numpy as np

from collector import Collector


class _Event:
    def __init__(self, frame):
        self.instance_segmentation_frame = frame


def test_mark_is_the_interior_pixel_in_full_image_coordinates():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    color = np.array([10, 20, 30], dtype=np.uint8)
    frame[2:7, 3:8] = color
    event = _Event(frame)
    collector = Collector.__new__(Collector)

    pixels, mark_x, mark_y = collector.get_visible_pixels_from_bbox(
        event, [3, 2, 8, 7], color, with_mark=True
    )

    assert pixels == 25
    assert (mark_x, mark_y) == (5, 4)
    assert np.array_equal(frame[mark_y, mark_x], color)
