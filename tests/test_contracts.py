import unittest
import numpy as np
import torch
from rolehos.runtime import preprocess, confusion, metrics
from mmseg.models.utils.portable_points import point_sample
from mmseg.models.utils.relation_state import foreground5_to_compat6


class Contracts(unittest.TestCase):
    def test_threshold_and_ties(self):
        p = torch.zeros(1, 5, 1, 3)
        p[0, 0, 0, 0] = 0.5
        p[0, 1, 0, 0] = 0.5
        p[0, 4, 0, 1] = 0.6
        # Historical readout uses strict > threshold; ties choose the first role.
        p[0, 0, 0, 2] = 0.6
        p[0, 1, 0, 2] = 0.6
        self.assertEqual(
            foreground5_to_compat6(p, threshold=0.5).tolist(), [[[0, 5, 1]]]
        )

    def test_rgb_normalization(self):
        x = preprocess(np.zeros((448, 448, 3), np.uint8))
        np.testing.assert_allclose(
            x[0, :, 0, 0].numpy(),
            -np.array([106.01075, 95.40013, 87.42854])
            / np.array([64.356636, 60.888744, 61.41911]),
            rtol=1e-6,
        )
        self.assertEqual(tuple(x.shape), (1, 3, 448, 448))

    def test_confusion_ignore_and_mini(self):
        gt = np.array([[1, 2, 4, 5, 255]], np.uint8)
        pred = np.array([[1, 2, 3, 5, 0]], np.uint8)
        c = confusion(pred, gt)
        self.assertEqual(int(c.sum()), 4)
        self.assertEqual(metrics(c, "minihoi4d")["mIoU"], 75.0)

    def test_point_sample(self):
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        coords = torch.tensor([[[0.25, 0.25], [0.75, 0.75], [0.5, 0.5]]])
        torch.testing.assert_close(
            point_sample(x, coords), torch.tensor([[[1.0, 4.0, 2.5]]])
        )


if __name__ == "__main__":
    unittest.main()
