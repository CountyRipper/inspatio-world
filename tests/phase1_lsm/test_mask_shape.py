import torch

from scripts.phase1_lsm.run_sharedA_hardgate_5deg import mask_video_for_vae


def test_lossless_mask_video_shape_for_vae() -> None:
    mask = torch.arange(5 * 2 * 3).reshape(5, 1, 2, 3)
    converted = mask_video_for_vae(mask)
    assert converted.shape == (1, 1, 5, 2, 3)
    assert torch.equal(converted[0, 0], mask[:, 0])
    try:
        mask_video_for_vae(torch.zeros(5, 2, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("mask without explicit singleton channel must fail")
