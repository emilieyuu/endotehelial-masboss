from src.abm.helpers.signalling import get_protein_recruitment, hill
from tests.abm_tests.helpers_shared import ABMHelperTestCase


class TestSignallingHelpers(ABMHelperTestCase):
    def test_hill_returns_zero_for_non_positive_stimulus(self):
        self.assertEqual(hill(0.0, K=4.0, n=2.0), 0.0)
        self.assertEqual(hill(-2.0, K=4.0, n=2.0), 0.0)

    def test_hill_returns_half_activation_at_threshold_when_n_is_one(self):
        self.assertAlmostEqual(hill(5.0, K=5.0, n=1.0), 0.5)

    def test_get_protein_recruitment_scales_hill_output_by_configured_maximum(self):
        cfg = self.make_cfg()
        recruitment = get_protein_recruitment(cfg, tau=4.0, protein="DSP")
        self.assertAlmostEqual(recruitment, 0.335)

    def test_get_protein_recruitment_returns_zero_for_knockout(self):
        cfg = self.make_cfg()
        cfg["hill_params"]["DSP"]["knocked_out"] = True
        self.assertEqual(get_protein_recruitment(cfg, tau=10.0, protein="DSP"), 0.0)
