import unittest
import numpy as np
from taiyaki import maths


class MathsTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print('* Maths routines')
        np.random.seed(0xdeadbeef)

    def test_001_studentise(self):
        sh = (7, 4)
        x = np.random.normal(size=sh)
        x2 = maths.studentise(x)
        self.assertTrue(x2.shape == sh)
        self.assertAlmostEqual(np.mean(x2), 0.0)
        self.assertAlmostEqual(np.std(x2), 1.0)

    def test_002_studentise_over_axis0(self):
        sh = (7, 4)
        x = np.random.normal(size=sh)
        x2 = maths.studentise(x, axis=0)
        self.assertTrue(x2.shape == sh)
        self.assertTrue(np.allclose(np.mean(x2, axis=0), 0.0))
        self.assertTrue(np.allclose(np.std(x2, axis=0), 1.0))

    def test_003_studentise_over_axis1(self):
        sh = (7, 4)
        x = np.random.normal(size=sh)
        x2 = maths.studentise(x, axis=1)
        self.assertTrue(x2.shape == sh)
        self.assertTrue(np.allclose(np.mean(x2, axis=1), 0.0))
        self.assertTrue(np.allclose(np.std(x2, axis=1), 1.0))

    def test_004_med_mad(self):
        x = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.0, 0.5, 0.5, 1.0]])
        factor = 1
        loc, scale = maths.med_mad(x, factor=factor)
        self.assertTrue(np.allclose(loc, 0.5))
        self.assertTrue(np.allclose(scale, 0))

    def test_005_med_mad_over_axis0(self):
        x = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.5, 1.0, 0.5, 1.0]])
        factor = 1
        loc, scale = maths.med_mad(x, factor=factor, axis=0)
        expected_loc = [0.5, 0.5, 0.5, 1.0]
        expected_scale = [0, 0, 0, 0]
        self.assertTrue(np.allclose(loc, expected_loc))
        self.assertTrue(np.allclose(scale, expected_scale))

    def test_006_med_mad_over_axis1(self):
        x = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.0, 0.5, 0.5, 1.0]])
        factor = 1
        loc, scale = maths.med_mad(x, factor=factor, axis=1)
        expected_loc = [0.5, 0.75, 0.5]
        expected_scale = [0, 0.25, 0.25]
        self.assertTrue(np.allclose(loc, expected_loc))
        self.assertTrue(np.allclose(scale, expected_scale))

    def test_007_mad_keepdims(self):
        x = np.zeros((5, 6, 7))
        self.assertTrue(np.allclose(maths.mad(x, axis=0, keepdims=True),
                                    np.zeros((1, 6, 7))))
        self.assertTrue(np.allclose(maths.mad(x, axis=1, keepdims=True),
                                    np.zeros((5, 1, 7))))
        self.assertTrue(np.allclose(maths.mad(x, axis=2, keepdims=True),
                                    np.zeros((5, 6, 1))))


if __name__ == '__main__':
    unittest.main()
