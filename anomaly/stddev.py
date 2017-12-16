import math

# Approximations only
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
class Stddev(object):
    def __init__(self):
        # needed for calulation
        self.n = 0 #  ATTENTION: Buffer Overflow?
        self.mean = 0.0
        self.M2 = 0.0
        # not needed for calculation
        self.stddev = 0.0

    def train(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

        if self.n < 2:
            variance = 0
        else:
            variance = self.M2 / (self.n - 1)

        self.stddev = math.sqrt(variance)
        return self.stddev

    # The probability of the particular value
    def sigma_of_val(self, value):
        if self.stddev == 0:
            return float('nan')
        return abs(value - self.mean) / self.stddev

    def sigma_and_train(self, value):
        inside_probability_content = self.sigma_of_val(value)
        self.train(value)
        return inside_probability_content

    def train_and_sigma(self, value):
        self.train(value)
        inside_probability_content = self.sigma_of_val(value)
        return inside_probability_content

    # e.g. to get the max. value of a specific sigma threshold
    def val_of_sigma(self, sigma):
        return self.mean + sigma*self.stddev


class ExponentialSmoothingStddev(object):
    def __init__(self, alpha):
        # needed for calulation
        self.n = 0 #  ATTENTION: Buffer Overflow?
        self.mean = 0.0
        self.M2 = 0.0 # For normal, this value _almost only_ increases
        # not needed for calculation
        self.stddev = 0.0
        #
        self.alpha = alpha
        self.last_smoothed_m2 = 0.0

    def train(self, x):
        self.n += 1

        # For normal, this value _almost only_ increases
        # Only Difference to "default" Sigma.
        # Unfortunatly, it keeps shrinking, even if the stddev is not increasing anymore
        self.M2
        if self.n == 1:
            self.last_smoothed_m2 = self.M2
        else:
            self.last_smoothed_m2 = self.alpha * self.M2 + (1-self.alpha) * self.last_smoothed_m2
        ###

        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.last_smoothed_m2 += delta * delta2

        if self.n < 2:
            variance = 0
        else:
            variance = self.last_smoothed_m2 / (self.n - 1)

        self.stddev = math.sqrt(variance)
        return self.stddev

    def sigma_of_val(self, value):
        if self.stddev == 0:
            return float('nan')
        return abs(value - self.mean) / self.stddev

    def sigma_and_train(self, value):
        inside_probability_content = self.sigma_of_val(value)
        self.train(value)
        return inside_probability_content

    def train_and_sigma(self, value):
        self.train(value)
        inside_probability_content = self.sigma_of_val(value)
        return inside_probability_content

    # e.g. to get the max. value of a specific sigma threshold
    def val_of_sigma(self, sigma):
        return self.mean + sigma*self.stddev
