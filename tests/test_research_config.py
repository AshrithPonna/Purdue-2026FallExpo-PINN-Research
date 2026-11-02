import unittest

from pinn_research.research_config import (
    INITIAL_CAPITAL,
    TEST_STEPS,
    TRAINING_OBSERVATIONS,
    default_research_configs,
)


class ResearchConfigTests(unittest.TestCase):
    def test_shared_defaults_match_research_experiment(self) -> None:
        strategy, integration = default_research_configs(
            execution_seed=123,
            execution_fee_aware=False,
        )

        self.assertEqual((TRAINING_OBSERVATIONS, TEST_STEPS), (750, 500))
        self.assertEqual(INITIAL_CAPITAL, 100_000.0)
        self.assertEqual((strategy.entry_z, strategy.exit_z), (2.0, 0.5))
        self.assertEqual(strategy.max_pairs, 1)
        self.assertEqual(integration.execution_seed, 123)
        self.assertFalse(integration.execution_fee_aware)
        self.assertEqual(integration.pinn.seed, 19)


if __name__ == "__main__":
    unittest.main()
