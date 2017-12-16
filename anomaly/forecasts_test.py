import unittest

from anomaly.forecasts import Naive, MovingAverageWindow, SingleExponentialSmoothing2Good, DoubleExponentialSmoothing2Good, TripleExponentialSmoothing2


class TestForecasts(unittest.TestCase):

    def test_naive(self):
        result = Naive().calc_forecasts(3)
        self.assertEqual(result, [3])

    def test_moving_average(self):
        values = [1,2,3]
        for i in range(1, len(values)-1):
            result = MovingAverageWindow(3).calc_forecasts(values[i])
        self.assertEqual(result, [2])

    def test_exponential_smoothing1_1(self):
        values = [512.3, 496.2, 509.8, 551.9, 539.9, 524.9, 530.3, 540.9, 541.3, 554.2, 557.5, 549.3, 549.4, 552.9,
                  549.7, 532.1, 545.5, 553.0, 582.1, 583.1]

        # want = [512.3, 507.5, 508.2, 521.3, 526.9, 526.3, 527.5, 531.5, 534.4, 540.4, 545.5, 546.7, 547.5, 549.1, 549.3,
        #        544.1, 544.5, 547.1, 557.6, 565.2]
        want = [565.2204177005497] # not exactly 565.2, probably because of more decimal values and rounding

        e = SingleExponentialSmoothing2Good(alpha=0.3)
        for i in range(1, len(values)-1):
            e.calc_forecasts(values[i])
        result = e.calc_forecasts(values[-1])
        self.assertEqual(result, want)

    def test_exponential_smoothing1_2(self):
        values = [512.3, 496.2]

        # want = [512.3, 507.5, 508.2, 521.3, 526.9, 526.3, 527.5, 531.5, 534.4, 540.4, 545.5, 546.7, 547.5, 549.1, 549.3,
        #        544.1, 544.5, 547.1, 557.6, 565.2]
        want = [507.4699999999999] # not exactly 565.2, probably because of more decimal values and rounding

        e = SingleExponentialSmoothing2Good(alpha=0.3)
        for i in range(0, len(values)-1):
            e.calc_forecasts(values[i])
        result = e.calc_forecasts(values[-1])
        self.assertEqual(result, want)


    def test_exponential_smoothing2_1(self):
        # values from https://de.wikipedia.org/wiki/Exponentielle_Gl%C3%A4ttung#Beispiel_f.C3.BCr_den_exponentiell_gegl.C3.A4tteten_DAX
        values = [512.3, 496.2, 509.8, 551.9, 539.9, 524.9, 530.3, 540.9, 541.3, 554.2, 557.5, 549.3, 549.4, 552.9,
                  549.7, 532.1, 545.5, 553.0, 582.1, 583.1]

        # want = [512.3, 507.5, 508.2, 521.3, 526.9, 526.3, 527.5, 531.5, 534.4, 540.4, 545.5, 546.7, 547.5, 549.1, 549.3,
        #        544.1, 544.5, 547.1, 557.6, 565.2]
        want = [565.2204177005497] # not exactly 565.2, because of more decimal values and rounding

        e = SingleExponentialSmoothing2Good(alpha=0.3)
        for i in range(1, len(values)-1):
            e.calc_forecasts(values[i])
        result = e.calc_forecasts(values[-1])
        self.assertEqual(result, want)

    def test_exponential_smoothing2_2(self):
        # values from https://de.wikipedia.org/wiki/Exponentielle_Gl%C3%A4ttung#Beispiel_f.C3.BCr_den_exponentiell_gegl.C3.A4tteten_DAX
        values = [512.3, 496.2]

        # want = [512.3, 507.5, 508.2, 521.3, 526.9, 526.3, 527.5, 531.5, 534.4, 540.4, 545.5, 546.7, 547.5, 549.1, 549.3,
        #        544.1, 544.5, 547.1, 557.6, 565.2]
        want = [507.4699999999999] # not exactly 565.2, because of more decimal values and rounding

        e = SingleExponentialSmoothing2Good(alpha=0.3)
        for i in range(0, len(values)-1):
            e.calc_forecasts(values[i])
        result = e.calc_forecasts(values[-1])
        self.assertEqual(result, want)

    def test_double_exponential_smoothing2(self):
        values = [512.3, 496.2, 509.8, 551.9, 539.9, 524.9, 530.3, 540.9, 541.3, 554.2, 557.5, 549.3, 549.4, 552.9,
                  549.7, 532.1, 545.5, 553.0, 582.1, 583.1]

        # want = [512.3, 507.5, 508.2, 521.3, 526.9, 526.3, 527.5, 531.5, 534.4, 540.4, 545.5, 546.7, 547.5, 549.1, 549.3,
        #        544.1, 544.5, 547.1, 557.6, 565.2]
        want = [565.2204177005497] # not exactly 565.2, probably because of more decimal values and rounding

        e = DoubleExponentialSmoothing2Good(alpha=0.3, beta=0.0)
        for i in range(1, len(values)-1):
            e.calc_forecasts(values[i])
        result = e.calc_forecasts(values[-1])
        self.assertEqual(result, want)


    def test_double_exponential_smoothing2_2(self):
        values = [512.3, 496.2, 509.8, 551.9, 539.9, 524.9, 530.3, 540.9, 541.3, 554.2, 557.5, 549.3, 549.4, 552.9,
                  549.7, 532.1, 545.5, 553.0, 582.1, 583.1]

        # want = [512.3, 507.5, 508.2, 521.3, 526.9, 526.3, 527.5, 531.5, 534.4, 540.4, 545.5, 546.7, 547.5, 549.1, 549.3,
        #        544.1, 544.5, 547.1, 557.6, 565.2]
        want = [570.6394422451164] # not exactly 565.2, probably because of more decimal values and rounding

        e = DoubleExponentialSmoothing2Good(alpha=0.3, beta=0.1)
        for i in range(1, len(values)-1):
            e.calc_forecasts(values[i])
        result = e.calc_forecasts(values[-1])
        self.assertEqual(result, want)


    # def test_double_exponential_smoothing2_3(self):
    #     # http://www.itl.nist.gov/div898/handbook/pmc/section4/pmc434.htm
    #     # Different start values!
    #     values = [6.4,  5.6,  7.8]#,  8.8,  11,  11.6,  16.7,  15.3,  21.6,  22.4]
    #
    #     # want = [512.3, 507.5, 508.2, 521.3, 526.9, 526.3, 527.5, 531.5, 534.4, 540.4, 545.5, 546.7, 547.5, 549.1, 549.3,
    #     #        544.1, 544.5, 547.1, 557.6, 565.2]
    #     want = [570.6394422451164] # not exactly 565.2, probably because of more decimal values and rounding
    #
    #     e = DoubleExponentialSmoothing2Good(alpha=0.3623, gamma=1.0)
    #     for i in range(1, len(values)-1):
    #         e.calc_forecasts(values[i])
    #     result = e.calc_forecasts(values[-1])
    #     self.assertEqual(result, want)




    def test_triple_exponential_smoothing2(self):
        values = [512.3, 496.2, 509.8, 551.9, 539.9, 524.9, 530.3, 540.9, 541.3, 554.2, 557.5, 549.3, 549.4, 552.9,
                  549.7, 532.1, 545.5, 553.0, 582.1, 583.1]

        # want = [512.3, 507.5, 508.2, 521.3, 526.9, 526.3, 527.5, 531.5, 534.4, 540.4, 545.5, 546.7, 547.5, 549.1, 549.3,
        #        544.1, 544.5, 547.1, 557.6, 565.2]
        want = [565.2204177005497] # not exactly 565.2, probably because of more decimal values and rounding

        e = TripleExponentialSmoothing2(seasons=0, alpha=0.3, beta=0.0, gamma=0.0)
        for i in range(1, len(values)-1):
            e.calc_forecasts(values[i])
        result = e.calc_forecasts(values[-1])
        self.assertEqual(result, want)



    def test_triple_exponential_smoothing2_2(self):
        values = [512.3, 496.2, 509.8, 551.9, 539.9, 524.9, 530.3, 540.9, 541.3, 554.2, 557.5, 549.3, 549.4, 552.9,
                  549.7, 532.1, 545.5, 553.0, 582.1, 583.1]

        # want = [512.3, 507.5, 508.2, 521.3, 526.9, 526.3, 527.5, 531.5, 534.4, 540.4, 545.5, 546.7, 547.5, 549.1, 549.3,
        #        544.1, 544.5, 547.1, 557.6, 565.2]
        want = [570.6394422451164] # not exactly 565.2, probably because of more decimal values and rounding

        e = TripleExponentialSmoothing2(seasons=0, alpha=0.3, beta=0.0, gamma=0.1)
        for i in range(1, len(values)-1):
            e.calc_forecasts(values[i])
        result = e.calc_forecasts(values[-1])
        self.assertEqual(result, want)



if __name__ == '__main__':
    unittest.main()
