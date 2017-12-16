import math


class BaseErrorFunction(object):
    def calc_error(self, a, b):
        raise NotImplementedError()

    def reverse(self, value):
        raise NotImplementedError()


# Todo: Border error with using window average or exponential smoothing,
#       depends on input order of forecast value and real value.
class Diff(BaseErrorFunction):
    def calc_error(self, a, b):
        return a - b

    def reverse(self, value):
        return value


class AbsDiff(BaseErrorFunction):
    def calc_error(self, a, b):
        # 35492.48990837435
        return abs(a - b)

    def reverse(self, value):
        return value


class AbsDiffMulti(BaseErrorFunction):
    # No difference to AbsDiff when using Sigma Threshold
    def __init__(self, multiplicator):
        self.multiplicator = multiplicator

    def calc_error(self, a, b):
        return abs(a - b) * self.multiplicator

    def reverse(self, value):
        return value / self.multiplicator


class SquaredError(BaseErrorFunction):
    def calc_error(self, a, b):
        return (a - b) * (a - b)

    def reverse(self, value):
        return math.sqrt(abs(value))


class MeanAbsError(BaseErrorFunction):
    def __init__(self, window_size):
        self.window_size = window_size
        self.abs_errors = []

    def calc_error(self, a, b):
        abs_err = abs(a-b)
        self.abs_errors.append(abs_err)
        while len(self.abs_errors) > self.window_size:
            self.abs_errors = self.abs_errors[1:]
        sum = 0
        for err in self.abs_errors:
            sum += err
        mean = sum / len(self.abs_errors)
        return mean

    def reverse(self, value):
        return value


class MeanSquaredError(BaseErrorFunction):
    def __init__(self, window_size):
        self.window_size = window_size
        self.squared_errors = []

    def calc_error(self, a, b):
        se = (a - b) * (a - b)
        self.squared_errors.append(se)
        while len(self.squared_errors) > self.window_size:
            self.squared_errors = self.squared_errors[1:]
        sum = 0
        for err in self.squared_errors:
            sum += err
        mean = sum / len(self.squared_errors)
        return mean

    def reverse(self, value):
        return math.sqrt(abs(value))



class LastAbsMeanError(BaseErrorFunction):
    def __init__(self, window_size):
        self.window_size = window_size
        self.abs_errors = []

    def calc_error(self, a, b):
        abs_err = abs(a-b)

        if len(self.abs_errors) > 0:
            sum = 0
            for err in self.abs_errors:
                sum += err
            mean = sum / len(self.abs_errors)
        else:
            mean = abs_err

        self.abs_errors.append(abs_err)
        while len(self.abs_errors) > self.window_size:
            self.abs_errors = self.abs_errors[1:]

        return mean

    def reverse(self, value):
        return value



class RootMeanSquaredError(BaseErrorFunction):
    def __init__(self, window_size):
        self.squared_errors = []
        self.window_size = window_size

    def calc_error(self, a, b):
        se = (a - b) * (a - b)
        sum = 0

        self.squared_errors.append(se)
        while len(self.squared_errors) > self.window_size:
            self.squared_errors = self.squared_errors[1:]

        for se in self.squared_errors:
            sum += se
        rmse = math.sqrt(sum / len(self.squared_errors))



        return rmse

    # Not able to reverse a RMSE to the real value
    def reverse(self, value):
        return value


class LastRootMeanSquaredError(BaseErrorFunction):
    def __init__(self, window_size):
        self.squared_errors = []
        self.window_size = window_size

    def calc_error(self, a, b):
        se = (a - b) * (a - b)

        if len(self.squared_errors) > 0:
            sum = 0.0
            for past_se in self.squared_errors:
                sum += past_se
            result = math.sqrt(sum / len(self.squared_errors))
        else:
            result = 0

        self.squared_errors.append(se)
        while len(self.squared_errors) > self.window_size:
            self.squared_errors = self.squared_errors[1:]

        return result

    # Not able to reverse a RMSE to the real value
    def reverse(self, value):
        return value
