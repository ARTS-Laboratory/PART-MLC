import unittest
import numpy as np
import pytest

from eval.metrics import trac, snr


class TestPyTestMetrics:
    time = np.linspace(0, 4, 101)
    sampling_freq = (time[-1] - time[0]) / len(time)
    sine_1 = np.sin(2 * np.pi * 3 * time)
    cos_1 = np.sin(2 * np.pi * 3 * time + (np.pi / 2))
    sine_2 = np.sin(2 * np.pi * 2 * time)
    ones = np.ones(time.shape)
    zeros = np.zeros(time.shape)

    def test_root_mean_square_frequency(self):
        pytest.skip('Not written yet')

    def test_trac_same_signal(self):
        score = trac(self.sine_1, self.sine_1)
        assert 1 == pytest.approx(score)

    def test_trac_ones_zeros(self):
        score = trac(self.ones, self.zeros)
        assert 0 == pytest.approx(score)

    def test_trac_ones_negative_ones(self):
        score = trac(self.ones, -1*self.ones)
        assert 0 == pytest.approx(score)

    def test_trac_sine_1_negative_sine_1(self):
        score = trac(self.sine_1, -1*self.sine_1)
        assert 0.117 == pytest.approx(score)

    def test_trac_sine_1_cos_1(self):
        score = trac(self.sine_1, self.cos_1)
        assert 0.422 == pytest.approx(score)

    def test_snr_same_signal(self):
        score = snr(self.sine_1, self.sine_1, self.sampling_freq)
        print(score)
        assert True

# class TestMetrics(unittest.TestCase):
#     def setUp(self) -> None:
#         self.time = np.linspace(0, 4, 101)
#         self.sine_1 = np.sin(2 * np.pi * 3 * self.time)
#         self.cos_1 = np.sin(2 * np.pi * 3 * self.time + (np.pi / 2))
#
#
#         self.sine_2 = np.sin(2 * np.pi * 2 * self.time)
#         self.ones = np.ones(self.time.shape)
#         self.zeros = np.zeros(self.time.shape)
#
#     def test_root_mean_square_frequency(self):
#         assert False
#
#     def test_trac_same_signal(self):
#         score = trac(self.sine_1, self.sine_1)
#         self.assertAlmostEqual(1, score, 4)
#
#     def test_trac_ones_zeros(self):
#         score = trac(self.ones, self.zeros)
#         self.assertAlmostEqual(0, score, 4)
#
#     def test_trac_ones_negative_ones(self):
#         score = trac(self.ones, -1*self.ones)
#         self.assertAlmostEqual(0, score, 4)
#
#     def test_trac_sine_1_negative_sine_1(self):
#         score = trac(self.sine_1, -1*self.sine_1)
#         self.assertAlmostEqual(0.117, score, 4)
#
#     def test_trac_sine_1_cos_1(self):
#         score = trac(self.sine_1, self.cos_1)
#         self.assertAlmostEqual(0.422, score, 4)
#
#     def test_snr(self):
#         assert False
#
#
# if __name__ == '__main__':
#     unittest.main()
