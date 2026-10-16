from utils.mask_stride import mask_render_stride


def test_masks_render_every_step():
    assert mask_render_stride(40, horizon=1000) == 1
    assert mask_render_stride(1000, horizon=1000) == 1
    assert mask_render_stride(2000, horizon=1000) == 1
    assert mask_render_stride(-1, horizon=1000) == 1
    assert mask_render_stride(None, horizon=1000) == 1
