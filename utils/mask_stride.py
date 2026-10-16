"""How often RoomVisit renders a nav-camera mask."""


def mask_render_stride(max_steps=None, horizon=None, frames_per_episode=None):
    """Nav masks render on every agent step."""
    return 1
