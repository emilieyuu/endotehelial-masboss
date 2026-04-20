import numpy as np

from src.abm.helpers.mechanics import bilinear_tension, overdamped_step, relax_toward
from tests.abm_tests.helpers_shared import ABMHelperTestCase


class TestMechanicsHelpers(ABMHelperTestCase):
    def test_bilinear_tension_uses_full_stiffness_in_tension(self):
        tension = bilinear_tension(l=7.0, l0=5.0, k=3.0, kc_ratio=0.1)
        self.assertAlmostEqual(tension, 6.0)

    def test_bilinear_tension_uses_reduced_stiffness_in_compression(self):
        tension = bilinear_tension(l=3.0, l0=5.0, k=3.0, kc_ratio=0.1)
        self.assertAlmostEqual(tension, -0.6)

    def test_relax_toward_moves_fractionally_toward_target(self):
        relaxed = relax_toward(current=2.0, target=5.0, dt=0.5, tau=2.0)
        self.assertAlmostEqual(relaxed, 2.75)

    def test_overdamped_step_returns_unclamped_displacement_below_limit(self):
        displacement = overdamped_step(
            force=np.array([4.0, 0.0]),
            gamma=2.0,
            dt=0.5,
            max_displacement=5.0,
        )
        np.testing.assert_allclose(displacement, [1.0, 0.0])

    def test_overdamped_step_clamps_large_displacement_to_max_magnitude(self):
        displacement = overdamped_step(
            force=np.array([6.0, 8.0]),
            gamma=1.0,
            dt=1.0,
            max_displacement=2.0,
        )
        np.testing.assert_allclose(displacement, [1.2, 1.6])
        self.assertAlmostEqual(np.linalg.norm(displacement), 2.0)
