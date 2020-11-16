import unittest

from sklearn.datasets import load_breast_cancer

from main import AutoML


class TestDatasets(unittest.TestCase):
    def test_breast_cancer(self):
        x, y = load_breast_cancer(return_X_y=True)
        auto = AutoML(x, y)
        auto.train()
        self.assertGreater(auto.best_metrics["roc_auc"], .9)

    def test_file_dataset(self):
        auto = AutoML(file="data.csv")
        auto.train()
        self.assertGreater(auto.best_metrics["roc_auc"], .9)


if __name__ == '__main__':
    unittest.main()
