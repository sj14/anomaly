import stddev
import math


class BaseThresholdFunction(object):
    def calc_threshold(self, value):
        raise NotImplementedError()

    def borders(self, value, error_fn):
        raise NotImplementedError()

    def afterwards(self):
        raise NotImplementedError()



class SimpleThreshold(BaseThresholdFunction):
    def __init__(self, threshold):
        self.threshold = threshold

    def calc_threshold(self, value):
        if value > self.threshold:
            return True
        else:
            return False

    def borders(self, value, error_fn):
        if error_fn is None:
            upper_border = self.threshold
            lower_border = self.threshold
        else:
            upper_border = value + error_fn.reverse(self.threshold)
            lower_border = value - error_fn.reverse(self.threshold)
        return upper_border, lower_border

    def afterwards(self):
        return



class ExponentialSmoothingSigma(BaseThresholdFunction):
    def __init__(self, sigma, alpha):
        self.threshold = sigma
        self.standard_deviation = stddev.ExponentialSmoothingStddev(alpha=alpha)
        self.value_to_sigma = None

    def calc_threshold(self, value):
        self.value_to_sigma = value
        if self.standard_deviation.sigma_of_val(value) > self.threshold:
            return True
        else:
            return False

    def borders(self, value, error_fn):
        upper_border = self.standard_deviation.mean + self.standard_deviation.stddev*self.threshold # value + sigma_border
        lower_border = self.standard_deviation.mean - self.standard_deviation.stddev*self.threshold
        if error_fn is not None:
            upper_border = value + error_fn.reverse(upper_border)
            lower_border = value - abs(error_fn.reverse(-self.standard_deviation.mean-self.standard_deviation.stddev*self.threshold))
        return upper_border, lower_border

    def afterwards(self):
        self.standard_deviation.train(self.value_to_sigma)


class Sigma(BaseThresholdFunction):
    def __init__(self, threshold):
        self.threshold = threshold
        self.standard_deviation = stddev.Stddev()
        self.value_to_sigma = None

    def calc_threshold(self, value):
        self.value_to_sigma = value
        if self.standard_deviation.sigma_of_val(value) > self.threshold:
            return True
        else:
            return False

    def borders(self, real_or_forecast, error_fn):
        # Threshold based probability of raw values
        lower_border = self.standard_deviation.mean - self.standard_deviation.stddev * self.threshold
        upper_border = self.standard_deviation.mean + self.standard_deviation.stddev * self.threshold
        if error_fn is not None:
            # Threshold based on probability of error values
            lower_border = real_or_forecast - abs(error_fn.reverse(upper_border)) # YES! it has to be upper_border, because otherwise it is -stddev.mean - stddev.stddev, which is the same as upper_border!
            upper_border = real_or_forecast + error_fn.reverse(upper_border)

        return upper_border, lower_border

    def afterwards(self):
        self.standard_deviation.train(self.value_to_sigma)


# https://medium.com/@iliasfl/data-science-tricks-simple-anomaly-detection-for-metrics-with-a-weekly-pattern-2e236970d77
# class ExponentialMovingStddev(BaseThresholdFunction):
#
#     def __init__(self, weight):
#         self.weight = weight
#
#     def calc(self, value):
#         return math.sqrt(self.weight * value*value + (1-self.weight)* (value)*(value))
#         #EMS < - sqrt(w * EMS ^ 2 + (1 - w) * (x - EMA) ^ 2)
#
#     def borders(self, value, error_fn):
#         upper_border = math.sqrt(abs(value)) # value + sigma_border
#         lower_border = -math.sqrt(abs(value))
#         if error_fn is not None:
#             upper_border = math.sqrt(error_fn.reverse(abs(value)))
#             lower_border = -math.sqrt(error_fn.reverse(abs(value)))
#         return upper_border, lower_border
#
#     def afterwards(self):
#         return


class Probabilistic(BaseThresholdFunction):
    def __init__(self, threshold):
        self.threshold = threshold
        #self.stddev = stddev.Stddev()
        self.value_to_sigma = None
        self.P = None
        self.counter = 0
        self.alpha_original = 0.9
        self.s1 = None
        self.s2 = None
        self.sigma = None

    def calc_threshold(self, value):
        self.counter += 1
        #beta = 0.5
        is_anomaly = False
        #alpha = 1 - 1 / self.counter

        if self.counter == 1:
            self.s1 = value
            self.s2 = value*value
            #self.sigma = math.sqrt(self.s2 - (self.s1*self.s1))

        #if self.sigma != 0:
        #Z = (value - self.s1) / self.sigma
        #Z = self.stddev.standard_score(value)
        #Z = value - self.
        self.P = (1 / math.sqrt(2 * math.pi)) * math.exp(-((Z ** 2) / 2))

        if self.P < self.threshold:
            print("P: ", self.P)
            #alpha = 1 - 1/self.counter
            is_anomaly = True
        #else:
            #alpha = (1 - beta * self.P)*self.alpha_original

        #self.s1 = alpha * self.s1 + (1-alpha) * value
        #self.s2 = alpha * self.s2 + (1-alpha) * (value*value)

        #self.sigma = math.sqrt(self.s2 - (self.s1*self.s1))
        #self.value_to_sigma = math.sqrt(self.s2 - (self.s1*self.s1))
        #self.value_to_sigma = math.sqrt(value)
        self.value_to_sigma = value

        return is_anomaly


    def borders(self, value, error_fn):
        upper_border = self.stddev.mean + self.P + self.stddev.stddev*self.threshold # value + sigma_border
        lower_border = self.stddev.mean - self.P - self.stddev.stddev * self.threshold
        #upper_border = value + self.sigma # value + sigma_border
        #lower_border = value - self.sigma
        if error_fn is not None:
            upper_border = value + error_fn.reverse(upper_border)
            lower_border = value - abs(error_fn.reverse(-self.stddev.mean - self.P - self.stddev.stddev*self.threshold))
        return upper_border, lower_border

    def afterwards(self):
        #return
        self.stddev.train(self.value_to_sigma)


class ExternelProbabilistic(BaseThresholdFunction):
    def __init__(self, ProbabilisticForecasting_fn):
        self.forecast_fn = ProbabilisticForecasting_fn

    def calc_threshold(self, value):
        return self.forecast_fn.is_anomaly

    def borders(self, value, error_fn):
        upper_border = value + self.forecast_fn.stddev.mean + self.forecast_fn.stddev.stddev  # value + sigma_border
        lower_border = value - self.forecast_fn.stddev.mean - self.forecast_fn.stddev.stddev
        #upper_border = value + self.sigma # value + sigma_border
        #lower_border = value - self.sigma
        #if error_fn is not None:
        #    upper_border = value + error_fn.reverse(upper_border)
        #    lower_border = value - abs(error_fn.reverse(-self.stddev.mean - self.P - self.stddev.stddev*self.threshold))
        return upper_border, lower_border

    def afterwards(self):
        return