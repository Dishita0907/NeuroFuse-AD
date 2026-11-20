import unittest
import torch
from model import NeuroFuse
from train import cohort


class ModalityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        self.model = NeuroFuse().eval()
        self.x, self.t, self.v, _, _ = cohort(2, 0, 17)

    def test_omitted_image_cannot_change_score(self):
        mask = torch.tensor([[False, True, True], [False, False, True]])
        first, weights = self.model(self.x, self.t, self.v, mask)
        second, _ = self.model(self.x+100, self.t, self.v, mask)
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.equal(weights[~mask], torch.zeros_like(weights[~mask])))
        self.assertEqual(weights[1, 2].item(), 1.)

    def test_empty_record_abstains(self):
        with self.assertRaisesRegex(ValueError, "abstain"):
            self.model(self.x, self.t, self.v, torch.zeros(2, 3, dtype=torch.bool))
