"""How often RoomVisit renders a nav-camera mask."""

MASK_FRAMES_PER_EPISODE = 50


def mask_render_stride(max_steps, horizon, frames_per_episode=MASK_FRAMES_PER_EPISODE):
    """Agent steps between nav-mask frames.

    A full horizon keeps about ``frames_per_episode`` masks
    (1000 // 50 = 20). A shorter episode keeps that same spacing.
    ``max_steps // horizon`` is 0 for a 40-step run when the horizon is
    1000, and ``max(1, 0)`` would render every step.
    """
    spacing = max(1, int(horizon) // int(frames_per_episode))
    if max_steps is None or int(max_steps) <= 0:
        return spacing
    return max(spacing, max(1, int(max_steps) // int(frames_per_episode)))
