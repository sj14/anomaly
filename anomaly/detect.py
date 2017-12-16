import timeit

from enum import Enum


class DetectTypes(Enum):
    missing_label = 0
    false_positive = 1
    false_negative = 2
    true_positive = 3
    true_negative = 4


class Analysis(object):
    def __init__(self):
        #self._window = []
        self._sigma_threshold = 0
        self.forecasted_cur = []
        self.forecasted_last = 0 # only needed for correct plotting and calculating forecast absolute error
        self.upper_border = 0.0
        self.lower_border = 0.0
        self.detected_type = DetectTypes.missing_label
        # Stats
        self.stat_start_time = None
        self.stat_values_counter = 0.0
        self.stat_missing_labels = 0.0
        self.stat_missing_label_rate = 100.0
        self.stat_true_positives = 0.0
        self.stat_true_positive_rate = None
        self.stat_true_negatives = 0.0
        self.stat_false_positive_rate = None # aka sensitivity, recall, hit-rate
        self.stat_false_negatives = 0.0
        self.stat_true_negative_rate = None # aka specificity
        self.stat_false_positives = 0.0
        self.stat_false_negative_rate = None
        self.stat_positive_predictive_value = None # aka PPV, Precision
        self.stat_negative_predictive_value = None # aka NPV
        self.stat_f1_score = None
        self.stat_accuracy = None
        self.stat_forecast_sum_absolute_difference = 0
        self.stat_forecast_sum_squared_error = 0
        # Only for batch
        self.batch_anomalies = []
        self.batch_forecasts = []
        self.batch_detected_types = []
        self.batch_upper_borders = []
        self.batch_lower_borders = []

    def reset(self):
        self.__init__()

    def detect_continuously(self, real_value, threshold_fn=None, forecast_fn=None, error_fn=None, label=None, robust=False, weighting=False):
        self.stat_values_counter += 1

        if len(self.forecasted_cur) > 0:
            if type(self.forecasted_cur[0]) is list:
                self.forecasted_cur[0] = self.forecasted_cur[0][0]

            error = self.forecasted_cur[0] - real_value
            self.stat_forecast_sum_absolute_difference += abs(error)
            self.stat_forecast_sum_squared_error += error*error
        else:
            self.forecasted_cur.append(real_value)


        if error_fn is not None:
            # Order of paramters is important for at least Diff() with moving average or exponential smoothing
            value_to_threshold = error_fn.calc_error(real_value, self.forecasted_cur[0]) # for window average
            #value_to_threshold = error_fn.calc(self.forecasted_cur, value) # for exponential smoothing
        else:
            value_to_threshold = real_value

        is_anomaly = False
        if threshold_fn is not None:
            # The actual detection part
            is_anomaly = threshold_fn.calc_threshold(value_to_threshold)

            # Detection Types, e.g. False-Positive or False-Negative
            self.detection_type(is_anomaly, label)

            # Borders
            if forecast_fn is not None:
                self.upper_border, self.lower_border = threshold_fn.borders(self.forecasted_cur[0], error_fn)
            else:
                self.upper_border, self.lower_border = threshold_fn.borders(real_value, error_fn)

            # Check if robust is enabled
            # and train threshold function
            # and drop value from window
            if is_anomaly == False or robust == False: # or len(self._window) < window_size:
                threshold_fn.afterwards()
            if is_anomaly == True and robust == True:
                forecast_fn.drop_last_from_window()

        if forecast_fn is not None:
            # save forecast of this point as last forecast
            self.forecasted_last = self.forecasted_cur[0]

            # drop current forecasted point
            # we have used it and there is no need for it anymore.
            self.forecasted_cur = self.forecasted_cur[1:]

            # if forecast weighting is true, do a new forecast calculation
            if weighting is True:
                forecast_new = forecast_fn.calc_forecasts(real_value)
                # Calculate the median weights of all forecasts done for a particular point
                for i, val in enumerate(forecast_new):
                    if len(self.forecasted_cur)-1 < i:
                        continue

                    # Calculate the moving averages for all forecasts of this particular point
                    if self.stat_values_counter > forecast_fn.horizon:
                        forecast_new[i] = self.forecasted_cur[i] + (forecast_new[i] - self.forecasted_cur[i]) / forecast_fn.horizon
                    else:
                        forecast_new[i] = self.forecasted_cur[i] + (forecast_new[i] - self.forecasted_cur[i]) / self.stat_values_counter
                self.forecasted_cur = forecast_new
            # else, wait as long as all forecasts "have been used" (no cumulative weighting of forecasts)
            else:
                # add value to window, maybe it is needed for the next calculation
                # even when weighting is false, and the calculation is only run every "horizons"
                forecast_fn.window.append(real_value)
                if len(self.forecasted_cur) == 0:
                    # drop last value, because it gets automatically added in the calc() function, too.
                    # otherwise the value is duplicated and messes up the window
                    forecast_fn.drop_last_from_window()
                    self.forecasted_cur.append(forecast_fn.calc_forecasts(real_value))

        return is_anomaly

    # Allows to calculate false positive/negative rate etc. on the batch
    def detect_batch(self, values, threshold_fn=None, forecast_fn=None, error_fn=None, labels=None, robust=False, weighting=False):
        for index, value in enumerate(values):
            if labels is not None and len(labels) > index:
                label = labels[index]
            else:
                label = None

            anomaly = self.detect_continuously(forecast_fn=forecast_fn, error_fn=error_fn, threshold_fn=threshold_fn,
                                               real_value=value, label=label, robust=robust, weighting=weighting,
                                               #window_size=window_size
                                               )
            self.batch_anomalies.append(anomaly)
            self.batch_forecasts.append(self.forecasted_last)
            self.batch_detected_types.append(self.detected_type)
            self.batch_upper_borders.append(self.upper_border)
            self.batch_lower_borders.append(self.lower_border)

    def calculate_stats(self, stdout=False):
            if self.stat_values_counter > 0 or self.stat_missing_labels > 0:
                self.stat_missing_label_rate = 100 / self.stat_values_counter * self.stat_missing_labels
            if self.stat_true_positives > 0 or self.stat_false_negatives > 0:
                self.stat_true_positive_rate = self.stat_true_positives / (self.stat_true_positives + self.stat_false_negatives) # aka sensitivity, recall, hit-rate
                self.stat_false_negative_rate = self.stat_false_negatives / (self.stat_true_positives + self.stat_false_negatives)
            if self.stat_true_negatives > 0 or self.stat_false_positives > 0:
                self.stat_true_negative_rate = self.stat_true_negatives / (self.stat_true_negatives + self.stat_false_positives) # aka specificity
                self.stat_false_positive_rate = self.stat_false_positives / (self.stat_true_negatives + self.stat_false_positives)
            if self.stat_true_positives > 0 or self.stat_false_positives > 0:
                self.stat_positive_predictive_value = self.stat_true_positives / (self.stat_true_positives + self.stat_false_positives) # aka PPV, Precision
                self.stat_negative_predictive_value = self.stat_true_negatives / (self.stat_true_negatives + self.stat_false_negatives) # aka NPV
            if self.stat_negative_predictive_value is not None and self.stat_true_positive_rate is not None:
                self.stat_f1_score = 2 * (self.stat_positive_predictive_value * self.stat_true_positive_rate / (self.stat_negative_predictive_value + self.stat_true_positive_rate))
            if self.stat_true_positives > 0 or self.stat_true_negatives > 0 or self.stat_false_positives > 0 or self.stat_false_negatives > 0:
                self.stat_accuracy = (self.stat_true_positives + self.stat_true_negatives) / (self.stat_true_positives + self.stat_true_negatives + self.stat_false_positives + self.stat_false_negatives)

            #aic = self.stat_values * math.log(self.stat_forecast_sum_absolute_difference / self.stat_values) + 2 * number_parameters
            #bic = self.stat_values * math.log(self.stat_forecast_sum_absolute_difference / self.stat_values) + number_parameters * math.log(self.stat_values)

            if stdout is True:
                print("missing labels rate:", self.stat_missing_label_rate)
                print("true positive rate:", self.stat_true_positive_rate)  # aka sensitivity
                print("false negative rate:", self.stat_false_negative_rate)
                print("true negative rate:", self.stat_true_negative_rate)  # aka specificity
                print("false positive rate:", self.stat_false_positive_rate)
                print("positive predictive value:", self.stat_positive_predictive_value)
                print("negative predictive value:", self.stat_negative_predictive_value)
                print("F1 Score:", self.stat_f1_score)
                print("Accuracy:", self.stat_accuracy)
                print("Forecast Sum of Absolute Difference (SAD): ", self.stat_forecast_sum_absolute_difference)
                print("Forecast Sum of Squared Errors (SSE): ", self.stat_forecast_sum_squared_error)

    def detection_type(self, anomaly_detected, label):
        # The detection type based on the given label
        if label == None:
            # Missing Label
            #print("Missing Label")
            self.detected_type = DetectTypes.missing_label
            self.stat_missing_labels += 1
        elif anomaly_detected == True and label == True:
            # True-Positive
            #print("True-Positive")
            self.detected_type = DetectTypes.true_positive
            self.stat_true_positives += 1
        elif anomaly_detected == True and label == False:
            # False-Positive
            #print("False-Positive")
            self.detected_type = DetectTypes.false_positive
            self.stat_false_positives += 1
        elif anomaly_detected == False and label == True:
            # False-Negative
            #print("False-Negative")
            self.detected_type = DetectTypes.false_negative
            self.stat_false_negatives += 1
        elif anomaly_detected == False and label == False:
            # True-Negative
            # print("True-Negative")
            self.detected_type = DetectTypes.true_negative
            self.stat_true_negatives += 1

    def start_timer(self):
        self.stat_start_time = timeit.default_timer()

    def reset_timer(self):
        self.stat_start_time = None

    def duration(self):
        return timeit.default_timer() - self.stat_start_time




