import unittest
from anomaly.errors import Diff, AbsDiff, SquaredError


class TestErrors(unittest.TestCase):
    def test_diff_calc(self):
        a = 1
        b = 5
        result = Diff().calc_error(a, b)
        self.assertEqual(result, -4)

    def test_diff_reverse(self):
        result = Diff().reverse(-4)
        self.assertEqual(result, -4)

    def test_abs_diff_calc(self):
        a = 1
        b = 5
        result = AbsDiff().calc_error(a, b)
        self.assertEqual(result, 4)

    def test_abs_diff_reverse(self):
        result = Diff().reverse(4)
        self.assertEqual(result, 4)

    def test_squared_error_calc(self):
        a = 1
        b = 5
        result = SquaredError().calc_error(a, b)
        self.assertEqual(result, 16)

    def test_squared_error_reverse(self):
        result = SquaredError().reverse(16)
        self.assertEqual(result, 4)


if __name__ == '__main__':
    unittest.main()
