from utils.mask_stride import mask_render_stride


def test_short_episode_keeps_full_horizon_spacing():
    # 1000 // 50 = 20. A 40-step episode uses that spacing, not every step.
    assert mask_render_stride(40, horizon=1000) == 20


def test_full_horizon_samples_about_fifty_frames():
    assert mask_render_stride(1000, horizon=1000) == 20


def test_longer_horizon_spreads_the_same_frame_budget():
    assert mask_render_stride(2000, horizon=1000) == 40


def test_unset_horizon_uses_full_spacing():
    assert mask_render_stride(-1, horizon=1000) == 20
    assert mask_render_stride(None, horizon=1000) == 20
