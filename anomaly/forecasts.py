import math
import stddev

class BaseForecastFunction(object):
    def __init__(self, window_size=0, horizon=1):
        self.window = []
        self.window_size = window_size
        self.horizon = horizon

    def calc_forecasts(self, value):
        raise NotImplementedError()

    def add_to_window(self, value):
        self.window.append(value)
        while len(self.window) > self.window_size:
            # drop first element from list
            self.drop_first_from_window()

    def drop_first_from_window(self):
        self.window = self.window[1:]

    def drop_last_from_window(self):
        self.window = self.window[0:-1]



class BaseForecastAutoWindowFunction(BaseForecastFunction):
    def __init__(self, window_size, horizon=1):
        super(BaseForecastAutoWindowFunction, self).__init__(window_size, horizon)
        self.horizon = horizon
        self.window_buffer = []
        self.window_size_buffer = window_size + 10
        self.window_size_max = self.window_size_buffer

    def calc_forecasts(self, value):
        raise NotImplementedError()

    def calc_without_adjust_window(self):
        raise NotImplementedError()

    def manage_window(self, value):
        self.window_buffer.append(value)

        if len(self.window) < 1:
            self.window.append(value)
            return

        cur_lowest_error = None
        cur_best_window_size = 1
        cur_cont_increased_error = 0

        for i in range(1, len(self.window_buffer), 1):
            self.window = self.window_buffer[len(self.window_buffer)-i-1:len(self.window_buffer)-1] # only last i entries of the window
            calc = self.calc_without_adjust_window()
            error = abs(calc-value)

            if cur_lowest_error is None:
                cur_lowest_error = error
                cur_best_window_size = i
            elif error < cur_lowest_error:
                cur_cont_increased_error = 0
                cur_lowest_error = error
                cur_best_window_size = i
                if (cur_best_window_size + 1) * 2 > self.window_size_max:
                    # Current window size is half as big as the currently max. allowed window size.
                    # Increasing max. allowed window size.
                    self.window_size_max = round((cur_best_window_size + 1) * 1.33)
            else:
                cur_cont_increased_error += 1
                if cur_cont_increased_error == 5:
                    break

        #print("Best Window Size is: ", cur_best_window_size)

        self.window = self.window_buffer[len(self.window_buffer)-cur_best_window_size-1:len(self.window_buffer)-1]
        #print("window after: ", self.window)


        while len(self.window_buffer) > 10 and len(self.window_buffer) > self.window_size_max:
            # drop first element from list
            self.drop_first_from_window()

    def drop_last_from_window(self):
        self.window = self.window[:-1]

    def drop_first_from_window(self):
        self.window_buffer = self.window_buffer[1:]


class Naive(BaseForecastFunction):
    def __init__(self, horizon=1):
        super(Naive, self).__init__(horizon=horizon)

    def calc_forecasts(self, value):
        forecasts = []
        for h in range(1, self.horizon+1):
            forecasts.append(value)
        return forecasts


class SeasonalNaive(BaseForecastFunction):
    def __init__(self, seasons, drift=False, horizon=1):
        super(SeasonalNaive, self).__init__(window_size=seasons, horizon=horizon)
        self.seasons = seasons
        self.use_drift = drift

    def calc_forecasts(self, value):
        forecasts = []

        for h in range(1, self.horizon+1):
            if len(self.window) < self.seasons:
                forecasts.append(value)
            else:
                last_season_value = self.window[-self.seasons-1 + h]
                last_season_next_value = self.window[-self.seasons + h]

                drift_value = 0
                # only a simple drift between the current point and the last point of the season
                if self.use_drift is True and len(self.window) >= self.seasons and h == 1:
                    drift_value = value - last_season_value
                    # drift_relative = ((100/last_season_next_value) * value)/100
                f = last_season_next_value + drift_value
                forecasts.append(f)

        self.add_to_window(value)
        return forecasts


class MovingAverageOnline(BaseForecastFunction):
    def __init__(self, horizon=1):
        super(MovingAverageOnline, self).__init__(window_size=1, horizon=horizon)
        self.counter = 0.0
        self.average = 0.0

    def calc_forecasts(self, value):
        self.counter += 1
        # current moving avarage, which is used for one-step forecasting
        self.average += (value - self.average)/self.counter
        return self.average

        # n-step forecasting moving average does not make sense and will aloways add the same number (the average)
        # divided by counter +1, which will result in the same average
        n_forecasts = [self.average]
        n_forecasts_average = self.average
        n_forecasts_counter = self.counter
        for i in range(1, self.horizon):
            n_forecasts_counter += 1
            n_forecasts_average += (n_forecasts[-i] - n_forecasts_average) / n_forecasts_counter
            n_forecasts.append(n_forecasts_average)
        return n_forecasts


class MovingAverageWindow(BaseForecastFunction):
    def __init__(self, window_size, horizon=1):
        super(MovingAverageWindow, self).__init__(window_size=window_size, horizon=horizon)

    def calc_forecasts(self, value):

        if len(self.window) < 1:
            self.add_to_window(value)
            return value

        average = sum(self.window) / len(self.window)

        forecasts = []
        forecasts.append(average)
        for i in range(1, self.horizon):
            forecasts.append(average)

        self.add_to_window(value)

        return forecasts



class MovingAverageAutoWindow(BaseForecastAutoWindowFunction):
    def __init__(self, horizon=1):
        super(MovingAverageAutoWindow, self).__init__(window_size=1, horizon=horizon)

    def calc_forecasts(self, value):
        self.manage_window(value)

        if len(self.window) < 1:
            return [value]

        average = sum(self.window) / len(self.window)
        n_forecasts = []
        n_forecasts.append(average)
        for i in range(0, self.horizon):
            n_forecasts.append(average)
        return n_forecasts


    def calc_without_adjust_window(self):
        return sum(self.window) / len(self.window)


# Level
class SingleExponentialSmoothing2Good(BaseForecastFunction):
    def __init__(self, alpha, horizon=1):
        super(SingleExponentialSmoothing2Good, self).__init__()
        self.alpha = alpha
        self.horizon = horizon
        self.last_level = None
        self.counter = 0

    def calc_forecasts(self, value):
        self.counter += 1

        if self.last_level is None:
            self.last_level = value
        level = self.alpha * value + (1 - self.alpha) * self.last_level
        self.last_level = level

        result = level
        return [result]


# Level + Trend
class DoubleExponentialSmoothing2Good(BaseForecastFunction):
    def __init__(self, alpha, beta, horizon=1):
        super(DoubleExponentialSmoothing2Good, self).__init__(window_size=1)
        self.alpha = alpha
        self.beta = beta
        self.horizon = horizon

        self.last_level = None
        self.last_additive_trend = 0.0
        self.counter = 0

    def calc_forecasts(self, value):
        self.add_to_window(value)
        self.counter += 1

        if self.last_level is None:
            self.last_level = value

        level = self.alpha * value + (1 - self.alpha) * (self.last_level + self.last_additive_trend)
        additive_trend = self.beta * (level - self.last_level) + (1 - self.beta) * self.last_additive_trend

        self.last_level = level
        self.last_additive_trend = additive_trend


        result = level + self.horizon * additive_trend
        return [result]


# Level + Trend + Season
class TripleExponentialSmoothing2(BaseForecastFunction):
    def __init__(self, seasons, alpha, beta, gamma, horizon=1, additive_seasonality=True):
        super(TripleExponentialSmoothing2, self).__init__(window_size=1)
        self.alpha = alpha  # overall smoothing
        self.beta = beta  # trend smoothing
        self.gamma = gamma    # season smoothing
        self.seasons = seasons
        self.horizon = horizon

        self.last_level = None
        self.last_trend = 0.0
        self.seasonality_list = []
        self.counter = 0
        self.additive_seasonality = additive_seasonality

    def calc_forecasts(self, value):
        self.add_to_window(value)
        self.counter += 1

        if self.seasons == 0:
            self.seasons = 1

        results = []
        for h in range(1, self.horizon+1):

            if len(self.seasonality_list) > self.counter + h % self.seasons:
                last_seasonality = self.seasonality_list[self.counter + h % self.seasons]
            else:
                if self.additive_seasonality is True:
                    last_seasonality = 0
                else:
                    last_seasonality = 1

            if self.last_level is None:
                self.last_level = value

            if self.additive_seasonality is True:
                level = self.alpha * (value - last_seasonality) + (1 - self.alpha) * (self.last_level + self.last_trend)
            else:
                # multiplicative seasonality
                level = self.alpha * (value / last_seasonality) + (1 - self.alpha) * (self.last_level + self.last_trend)

            trend = self.beta * (level - self.last_level) + (1 - self.beta) * self.last_trend

            if self.additive_seasonality is True:
                seasonality = self.gamma * (value - level) + (1 - self.gamma) * last_seasonality
            else:
                # multiplicative seasonality
                seasonality = self.gamma * (value / level) + (1 - self.gamma) * last_seasonality


            if len(self.seasonality_list) > self.counter % self.seasons:
                self.seasonality_list[self.counter % self.seasons] = seasonality
            else:
                self.seasonality_list.append(seasonality)


            seasonal_index = (self.counter-1) % self.seasons
            if len(self.seasonality_list) > seasonal_index:
                if self.additive_seasonality is True:
                    result = (level + h*trend) + self.seasonality_list[seasonal_index]
                else:
                    # multiplicative seasonality
                    result = (level + h*trend) * self.seasonality_list[seasonal_index]
            else:
                result = level + h*trend

            results.append(result)

            self.last_level = level
            self.last_trend = trend
        return results[-self.horizon:]



class TripleExponentialSmoothingWithBorder(BaseForecastFunction):
    def __init__(self, seasons, alpha, beta, gamma, horizon=1, additive_seasonality=True):
        super(TripleExponentialSmoothingWithBorder, self).__init__(window_size=1)
        self.alpha = alpha  # overall smoothing
        self.gamma = gamma  # trend smoothing
        self.beta = beta  # season smoothing
        self.seasons = seasons
        self.horizon = horizon

        self.last_level = None
        self.last_linear_trend = 0.0
        self.seasonal_trend_list = []
        self.counter = 0
        self.additive_seasonality = additive_seasonality

        self.seasonal_deviation_list = []
        self.seasonal_index = -1
        self.last_forecast_result = 0
        self.lower_border = 0
        self.upper_border = 0


    def calc_forecasts(self, value):
        self.add_to_window(value)
        self.counter += 1

        if self.seasons == 0:
            self.seasons = 1

        results = []
        for h in range(1, self.horizon+1):
            self.seasonal_index = (self.counter-1) % self.seasons

            if len(self.seasonal_trend_list) > self.seasonal_index:
                last_seasonal_trend = self.seasonal_trend_list[self.seasonal_index]
            else:
                if self.additive_seasonality is True:
                    last_seasonal_trend = 0
                else:
                    last_seasonal_trend = 1

            if self.last_level is None:
                self.last_level = value

            if self.additive_seasonality is True:
                level = self.alpha * (value - last_seasonal_trend) + (1 - self.alpha) * (self.last_level + self.last_linear_trend)
            else:
                # multiplicative seasonality
                level = self.alpha * (value / last_seasonal_trend) + (1 - self.alpha) * (self.last_level + self.last_linear_trend)

            linear_trend = self.beta * (level - self.last_level) + (1 - self.beta) * self.last_linear_trend

            if self.additive_seasonality is True:
                seasonal_trend = self.gamma * (value - level) + (1 - self.gamma) * last_seasonal_trend
            else:
                # multiplicative seasonality
                seasonal_trend = self.gamma * (value / level) + (1 - self.gamma) * last_seasonal_trend


            if len(self.seasonal_trend_list) > self.seasonal_index+1:
                if self.additive_seasonality is True:
                    result = (level + h*linear_trend) + self.seasonal_trend_list[self.seasonal_index+1]
                else:
                    # multiplicative seasonality
                    result = (level + h*linear_trend) * self.seasonal_trend_list[self.seasonal_index+1]
            else:
                    result = (level + h*linear_trend)


            if len(self.seasonal_trend_list) > self.seasonal_index:
                self.seasonal_trend_list[self.seasonal_index] = seasonal_trend
            else:
                self.seasonal_trend_list.append(seasonal_trend)


            results.append(result)

            # Borders Start
            err = abs(result - value)

            if len(self.seasonal_deviation_list) > self.seasonal_index:
                last_seasonal_deviation = self.seasonal_deviation_list[self.seasonal_index]
            else:
                last_seasonal_deviation = err

            deviation = self.gamma * err + (1 - self.gamma) * last_seasonal_deviation

            self.lower_border = result - 2.0 * last_seasonal_deviation
            self.upper_border = result + 2.0 * last_seasonal_deviation

            if len(self.seasonal_deviation_list) > self.seasonal_index:
                self.seasonal_deviation_list[self.seasonal_index] = deviation
            else:
                self.seasonal_deviation_list.append(deviation)
            # Borders Stop

            self.last_forecast_result = result
            self.last_level = level
            self.last_linear_trend = linear_trend
        return results[-self.horizon:]


    # Threshold
    def calc_threshold(self, value_or_error):
        if value_or_error > self.upper_border:
            self.is_anomaly = True
        else:
            self.is_anomaly = False
        return self.is_anomaly

    # Borders
    def borders(self, value_or_forecasted, error_fn):
        return self.upper_border, self.lower_border


    def afterwards(self):
        return




# Level + Trend + Season
class TripleExponentialSmoothingWithHorizon(BaseForecastFunction):
    def __init__(self, seasons, alpha, beta, gamma, horizon=1, additive_seasonality=True):
        super(TripleExponentialSmoothingWithHorizon, self).__init__(window_size=1)
        self.alpha = alpha  # overall smoothing
        self.gamma = gamma    # trend smoothing
        self.beta = beta  # season smoothing
        self.seasons = seasons
        self.horizon = horizon

        self.counter = 0
        self.previous_level = 0
        self.previous_trend = 0
        self.seasonal_trends_list = []


    def calc_forecasts(self, value):
        self.add_to_window(value)
        self.counter += 1

        if self.seasons == 0:
            self.seasons = 1

        if len(self.seasonal_trends_list) > (self.counter) % self.seasons:
            last_season_seasonal_trend = self.seasonal_trends_list[(self.counter-1) % self.seasons]
        else:
            last_season_seasonal_trend = 0

        level = self.alpha * (value - last_season_seasonal_trend) + (1 - self.alpha) * (self.previous_level + self.previous_trend)
        trend = self.beta  * (level - self.previous_level)        + (1 - self.beta)  *  self.previous_trend
        seasonal_trend = self.gamma*(value - self.previous_level - self.previous_trend)

        self.previous_level = level
        self.previous_trend = trend

        if len(self.seasonal_trends_list) > (self.counter) % self.seasons:
            self.seasonal_trends_list[(self.counter-1) % self.seasons] = seasonal_trend
        else:
            self.seasonal_trends_list.append(seasonal_trend)

        forecasts = []
        for h in range(1, self.horizon+1):
            if len(self.seasonal_trends_list) > (self.counter-1+h) % self.seasons:
                last_seasonal_trend_of_next_value = self.seasonal_trends_list[(self.counter-1+h) % self.seasons]
            else:
                last_seasonal_trend_of_next_value = 0

            forecast = level + h * trend + last_seasonal_trend_of_next_value
            forecasts.append(forecast)

            if self.counter < self.seasons*2:
                break

        return forecasts



class SeasonalTripleExponentialSmoothing(BaseForecastFunction):
    def __init__(self, seasons_primary, seasons_secondary, forecast_alpha, forecast_beta, forecast_gamma, additive_trend=True):
        super(SeasonalTripleExponentialSmoothing, self).__init__(window_size=1)
        self.seasons_primary = seasons_primary
        self.seasons_secondary = seasons_secondary
        self.forecast_alpha = forecast_alpha  # overall smoothing
        self.forecast_beta = forecast_beta  # trend smoothing
        self.forecast_gamma = forecast_gamma # secondary season

        self.counter = 0
        self.smoothed_below_season_size = 0
        self.seasonal_levels = []
        self.seasonal_trends = []
        self.seasonal_secondary = []
        self.additive_trend = additive_trend

    def calc_forecasts(self, value):
        self.counter += 1
        self.add_to_window(value)

        if self.seasons_primary == 0:
            self.seasons_primary = 1

        if self.counter <= self.seasons_primary:
            self.smoothed_below_season_size = self.forecast_alpha * value + (1 - self.forecast_alpha) * self.smoothed_below_season_size
            result = self.smoothed_below_season_size

            self.seasonal_levels.append(self.smoothed_below_season_size)

            if self.additive_trend is True:
                self.seasonal_trends.append(0.0)
            else:
                self.seasonal_trends.append(1.0)

            # Calculate and store new second seasonality
            if len(self.seasonal_secondary) < self.seasons_secondary:
                self.seasonal_secondary.append(1)

            ### Stop here and return result
            return result


        # Get Last Level
        season_last_level = self.seasonal_levels[self.counter % self.seasons_primary]

        # Get Last Trend
        season_last_trend = self.seasonal_trends[self.counter % self.seasons_primary]

        # Get last second seasonality
        if len(self.seasonal_secondary) < self.seasons_secondary:
            season_last_secondary = 1
        else:
            season_last_secondary = self.seasonal_secondary[self.counter % self.seasons_secondary]

        # Calculate Result based on last level and trend
        if self.additive_trend is True:
            result = (season_last_level + season_last_trend) * season_last_secondary
        else:
            result = (season_last_level * season_last_trend) * season_last_secondary


        # Calculate new Level
        level = self.forecast_alpha * (value / season_last_secondary) + (1 - self.forecast_alpha) * (season_last_level + season_last_trend)

        # Store new Level
        self.seasonal_levels[self.counter % self.seasons_primary] = level


        # Calculate and store new second seasonality
        if len(self.seasonal_secondary) < self.seasons_secondary:
            season_secondary = self.forecast_gamma * (value / level) + (1 - self.forecast_gamma)
            self.seasonal_secondary.append(season_secondary)
        else:
            season_secondary = self.forecast_gamma * (value / level) + (1 - self.forecast_gamma) * season_last_secondary
            self.seasonal_secondary[self.counter % self.seasons_secondary] = season_secondary

        # Calculate new tred
        if self.additive_trend is True:
            trend = self.forecast_beta * (level - season_last_level) + (1 - self.forecast_beta) * season_last_trend
        else:
            # Multiplicative trend
            trend = self.forecast_beta * (level / season_last_level) + (1 - self.forecast_beta) * season_last_trend

        # Store new Trend
        self.seasonal_trends[self.counter % self.seasons_primary] = trend

        return [result]

    # Threshold
    def calc_threshold(self, value_or_error):
        return self.is_anomaly


    def borders(self, value_or_forecasted, error_fn):
        return self.upper_border, self.upper_border

    def afterwards(self):
        return


class SeasonalProbabilisticTripleExponentialSmoothing(BaseForecastFunction):
    def __init__(self, seasons_primary, seasons_secondary, forecast_alpha, forecast_beta, forecast_gamma, horizon=1,
                 additive_trend=True):
        super(SeasonalProbabilisticTripleExponentialSmoothing, self).__init__(window_size=1)
        self.seasons_primary = seasons_primary
        self.seasons_secondary = seasons_secondary
        self.forecast_alpha = forecast_alpha  # overall smoothing
        self.forecast_beta = forecast_beta  # trend smoothing
        self.forecast_gamma = forecast_gamma  # secondary season

        self.counter = 0
        self.smoothed_below_season_size = 0
        self.seasonal_levels = []
        self.seasonal_trends = []
        self.seasonal_secondary = []
        self.additive_trend = additive_trend

        # Probabilistic
        self.z_score = 0.0
        self.probability = 0.0
        self.prediction = 0
        self.standard_deviation = 0

    def calc_forecasts(self, value):
        self.counter += 1
        self.add_to_window(value)

        if self.seasons_primary == 0:
            self.seasons_primary = 1

        # calculate the probability of this value
        self.calc_probabilistic(value)


        if self.counter <= self.seasons_primary:
            self.smoothed_below_season_size = self.forecast_alpha * value + (1 - self.forecast_alpha) * self.smoothed_below_season_size
            result = self.smoothed_below_season_size

            self.seasonal_levels.append(self.smoothed_below_season_size)

            if self.additive_trend is True:
                self.seasonal_trends.append(0.0)
            else:
                self.seasonal_trends.append(1.0)

            # Calculate and store new second seasonality
            if len(self.seasonal_secondary) < self.seasons_secondary:
                self.seasonal_secondary.append(1)

            ### Stop here and return result
            self.last_result = result
            return result

        # Get Last Level
        season_last_level = self.seasonal_levels[self.counter % self.seasons_primary]

        # Get Last Trend
        season_last_trend = self.seasonal_trends[self.counter % self.seasons_primary]

        # Get last second seasonality
        if len(self.seasonal_secondary) < self.seasons_secondary:
            season_last_secondary = 1
        else:
            season_last_secondary = self.seasonal_secondary[self.counter % self.seasons_secondary]

        # Calculate Result based on last level and trend
        if self.additive_trend is True:
            result = (season_last_level + season_last_trend) * season_last_secondary
        else:
            result = (season_last_level * season_last_trend) * season_last_secondary

        # Calculate new Level
        level = self.forecast_alpha * (value / season_last_secondary) + (1 - self.forecast_alpha * self.probability) * (
        season_last_level + season_last_trend)

        # Store new Level
        self.seasonal_levels[self.counter % self.seasons_primary] = level

        # Calculate and store new second seasonality
        if len(self.seasonal_secondary) < self.seasons_secondary:
            season_secondary = self.forecast_gamma * (value / level) + (1 - self.forecast_gamma)
            self.seasonal_secondary.append(season_secondary)
        else:
            season_secondary = self.forecast_gamma * (value / level) + (1 - self.forecast_gamma) * season_last_secondary
            self.seasonal_secondary[self.counter % self.seasons_secondary] = season_secondary

        # Calculate new tred
        if self.additive_trend is True:
            trend = self.forecast_beta * (level - season_last_level) + (1 - self.forecast_beta) * season_last_trend
        else:
            # Multiplicative trend
            trend = self.forecast_beta * (level / season_last_level) + (1 - self.forecast_beta) * season_last_trend

        # Store new Trend
        self.seasonal_trends[self.counter % self.seasons_primary] = trend

        self.last_result = result
        return [result]
        #return [self.prediction]

    # Threshold
    def calc_threshold(self, value_or_error):
        # only using difference, not based on error function!
        if self.z_score > 3.0:
            return True
        return False

    def borders(self, value_or_forecasted, error_fn):
        return 0, 0

    def afterwards(self):
        return

    # Probabilistic
    def calc_probabilistic(self, real_value):
        if self.standard_deviation != 0:
            self.z_score = (real_value - self.prediction) / self.standard_deviation
        self.probability = 1 / math.sqrt(2*math.pi) * math.exp(-(self.z_score*self.z_score/2))

        if self.counter == 1:
            self.s1 = real_value
            self.s2 = real_value*real_value
        else:
            self.s1 = self.last_result * self.s1 + (1 - self.last_result) * real_value
            self.s2 = self.last_result * self.s2 + (1 - self.last_result) * real_value * real_value
        self.prediction = self.s1
        self.standard_deviation = math.sqrt(abs(self.s2 - self.s1*self.s1))


class SeasonalDoubleExponentialSmoothing(BaseForecastFunction):
    def __init__(self, seasons, forecast_alpha, forecast_beta, horizon=1, additive_trend=True):
        super(SeasonalDoubleExponentialSmoothing, self).__init__(window_size=1)
        self.seasons = seasons
        self.forecast_alpha = forecast_alpha  # overall smoothing
        self.forecast_beta = forecast_beta  # trend smoothing

        self.counter = 0
        self.smoothed_below_season_size = 0
        self.seasonal_levels = []
        self.seasonal_trends = []
        self.additive_trend = additive_trend

    def calc_forecasts(self, value):
        self.counter += 1
        self.add_to_window(value)

        if self.seasons == 0:
            self.seasons = 1

        if self.counter <= self.seasons:
            self.smoothed_below_season_size = self.forecast_alpha * value + (
                                                                            1 - self.forecast_alpha) * self.smoothed_below_season_size
            result = self.smoothed_below_season_size

            self.seasonal_levels.append(self.smoothed_below_season_size)

            if self.additive_trend is True:
                self.seasonal_trends.append(0.0)
            else:
                self.seasonal_trends.append(1.0)
            return result

        # Get Last Level
        season_last_level = self.seasonal_levels[self.counter % self.seasons]

        # Get Last Trend
        season_last_trend = self.seasonal_trends[self.counter % self.seasons]

        # Calculate Result based on last level and trend
        if self.additive_trend is True:
            result = season_last_level + season_last_trend
        else:
            result = season_last_level * season_last_trend

        # Calculate new Level
        level = self.forecast_alpha * value + (1 - self.forecast_alpha) * (season_last_level + season_last_trend)

        # Store new Level
        self.seasonal_levels[self.counter % self.seasons] = level

        if self.additive_trend is True:
            trend = self.forecast_beta * (level - season_last_level) + (1 - self.forecast_beta) * season_last_trend
        else:
            # Multiplicative trend
            trend = self.forecast_beta * (level / season_last_level) + (1 - self.forecast_beta) * season_last_trend

        # Store new Trend
        self.seasonal_trends[self.counter % self.seasons] = trend

        return [result]

    # Threshold
    def calc_threshold(self, value_or_error):

        return self.is_anomaly

    def borders(self, value_or_forecasted, error_fn):

        return self.upper_border, self.upper_border

    def afterwards(self):
        return



class NeuralNetwork(BaseForecastFunction):
    def __init__(self, horizon=1):
        super(NeuralNetwork, self).__init__(window_size=1, horizon=horizon)
        import nnet
        self.nnet_instance = nnet.NNet()
        self.nnet_instance.compile_model()

    def calc_forecasts(self, value):
        results = []
        results.append(self.nnet_instance.train_and_predict(value))
        for h in range(1, self.horizon):
            results.append(self.nnet_instance.predict(results[-1]))

        return results[-self.horizon:]

