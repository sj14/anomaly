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
    def __init__(self):
        super(MovingAverageOnline, self).__init__(window_size=1, horizon=horizon)
        self.counter = 0.0
        self.average = 0.0

    def calc_forecasts(self, value):
        self.counter += 1
        # current moving avarage, which is used for one-step forecasting
        self.average += (value - self.average)/self.counter
        return self.average


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
class SingleExponentialSmoothing(BaseForecastFunction):
    def __init__(self, alpha):
        super(SingleExponentialSmoothing, self).__init__()
        self.alpha = alpha
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
class DoubleExponentialSmoothing(BaseForecastFunction):
    def __init__(self, alpha, beta):
        super(DoubleExponentialSmoothing, self).__init__(window_size=1)
        self.alpha = alpha
        self.beta = beta

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
class TripleExponentialSmoothing(BaseForecastFunction):
    def __init__(self, seasons, alpha, beta, gamma, horizon=1, additive_seasonality=True):
        super(TripleExponentialSmoothing, self).__init__(window_size=1)
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

