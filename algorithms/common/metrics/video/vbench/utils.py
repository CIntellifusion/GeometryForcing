from torch import Tensor


def mae_score(videos1: Tensor, videos2: Tensor) -> Tensor:
    """
    Compute the mean absolute error (MAE) between videos1 and videos2.
    `score = (255 - MAE) / 255`.

    Args:
        videos1: Videos of shape (B, T, C, H, W), uint8, range [0, 255].
        videos2: Videos of shape (B, T, C, H, W), uint8, range [0, 255].
    Returns:
        score: The computed MAE score of shape (B,), range [0, 1].
    """
    mae = (videos1.float() - videos2.float()).abs().mean(dim=(1, 2, 3, 4))
    return (255.0 - mae) / 255.0
