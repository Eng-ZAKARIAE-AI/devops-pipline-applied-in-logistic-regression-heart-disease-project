import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from api.server import sigmoid


class TestSigmoid:
    def test_sigmoid_zero(self):
        assert sigmoid(0.0) == 0.5

    def test_sigmoid_positive(self):
        assert sigmoid(10.0) > 0.99

    def test_sigmoid_negative(self):
        assert sigmoid(-10.0) < 0.01

    def test_sigmoid_range(self):
        for z in [-100, -10, -1, 0, 1, 10, 100]:
            p = sigmoid(z)
            assert 0.0 <= p <= 1.0

    def test_sigmoid_symmetry(self):
        z = 2.0
        assert abs(sigmoid(z) + sigmoid(-z) - 1.0) < 1e-12
