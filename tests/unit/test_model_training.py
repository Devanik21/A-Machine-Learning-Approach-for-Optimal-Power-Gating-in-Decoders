import unittest

class TestModelTraining(unittest.TestCase):
    def test_hyperparameter_config(self):
        """Test that hyperparameter grids are formatted correctly."""
        grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        self.assertIsInstance(grid, dict)
        self.assertIn('n_estimators', grid)

if __name__ == '__main__':
    unittest.main()
