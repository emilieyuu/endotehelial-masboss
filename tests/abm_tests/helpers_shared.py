import copy
import unittest

from src.utils.config_utils import load_abm_sim_cfg


class ABMHelperTestCase(unittest.TestCase):
    """Shared fixture helpers for ABM helper unit tests."""

    def make_cfg(self):
        """Return a writable ABM config copy for tests."""
        return copy.deepcopy(load_abm_sim_cfg())
