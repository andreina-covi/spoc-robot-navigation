import numpy as np

from collector import Collector


class _Event:
    def __init__(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        color = np.array([1, 2, 3], dtype=np.uint8)
        frame[1:6, 1:6] = color
        self.instance_segmentation_frame = frame
        self.instance_detections2D = {"Cup|1": [1, 1, 6, 6]}
        self.metadata = {
            "agent": {"position": {"x": 0, "y": 0, "z": 0}, "rotation": {"y": 0}, "cameraHorizon": 30},
            "cameraPosition": {"x": 0, "y": 1.5, "z": 0},
        }
        self.color = color


def test_navigation_row_keeps_agent_room_and_object_room():
    collector = Collector.__new__(Collector)
    collector.dict_colors = {"Cup|1": [1, 2, 3]}
    collector.nav_camera_mount_deg = 0
    collector.nav_camera_fov_deg = 59
    collector.FRAME_WIDTH = 8
    collector.FRAME_HEIGHT = 8
    collector.VERTICAL_FOV_DEG = 59
    event = _Event()
    obj = {
        "objectId": "Cup|1",
        "objectType": "Cup",
        "position": {"x": 1, "y": 0.8, "z": 2},
        "rotation": {"x": 0, "y": 0, "z": 0},
        "distance": 1.2,
        "roomId": "room|2",
        "roomType": "Kitchen",
        "axisAlignedBoundingBox": {
            "center": {"x": 1, "y": 0.8, "z": 2},
            "size": {"x": 0.1, "y": 0.1, "z": 0.1},
        },
    }

    rows, catalog = collector.get_object_data([obj], event, include_detections=True)

    assert catalog["Cup|1"]["room_id"] == "room|2"
    assert catalog["Cup|1"]["room_type"] == "Kitchen"
    assert rows[0][16:18] == ["room|2", "Kitchen"]
    assert rows[0][18] == 1.0  # in-frame-fraction: box fully inside the frame

    collector.dict_agent = {
        (0, "move"): {
            "objects": rows,
            "position": (0, 0, 0),
            "rotation": (0, 0, 0),
            "degrees": 0,
            "camera_horizon": 30,
            "camera_position": (0, 1.5, 0),
            "current_room": "room|1",
            "current_room_type": "Bedroom",
            "room_just_entered": False,
            "visited_rooms": "room|1",
            "seen_rooms": "",
            "action_success": True,
            "held_obj_id": None,
            "image": "images/img_0.png",
        }
    }
    navigation = collector.get_dict_navigation()
    assert navigation["current-room"] == ["room|1"]
    assert navigation["current-room-type"] == ["Bedroom"]
    assert navigation["obj-room"] == ["room|2"]
    assert navigation["obj-room-type"] == ["Kitchen"]
