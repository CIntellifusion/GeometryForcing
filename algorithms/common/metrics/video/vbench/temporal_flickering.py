from torch import Tensor
from .dimension import Dimension
from .utils import mae_score


class TemporalFlickering(Dimension):
    """
    Temporal flickering dimension.
    """

    def forward(self, videos: Tensor) -> Tensor:
        """
        Compute the temporal flickering score, which is the average of MAE between consecutive frames.
        MAE is then normalized from [0, 255] to [0, 1].

        Args:
            videos: Videos of shape (B, T, C, H, W), uint8, range [0, 255].
        Returns:
            score: The computed temporal flickering score of shape (B,), range [0, 1].
        """
        return mae_score(videos[:, 1:], videos[:, :-1])
