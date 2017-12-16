import matplotlib.pyplot as plt
import numpy as np
from anomaly import detect, forecasts, errors, thresholds, datagen


def main():
    # Generate the data
    values, anomaly_labels = datagen.cyclic_bump()

    # Run the anomaly detection algorithm
    analysis = detect_batch(values=values, labels=anomaly_labels)

    # Plot the results
    plot_analysis(values, anomaly_labels, analysis, plot_name="cyclic_bump")


def detect_batch(values, labels):
    seasonal_triple_exponential_smoothing = forecasts.SeasonalTripleExponentialSmoothing(
        seasons_primary=158,
        seasons_secondary=1106,
        forecast_alpha=0.2,
        forecast_beta=0.0,
        forecast_gamma=0.2,
        additive_trend=True)

    analysis = detect.Analysis()
    analysis.reset()
    analysis.start_timer()

    analysis.detect_batch(values=values,
                          error_fn=errors.RootMeanSquaredError(2),
                          labels=labels,
                          forecast_fn=seasonal_triple_exponential_smoothing,
                          threshold_fn=thresholds.Sigma(3.0),
                          )

    duration = analysis.duration()
    print("Duration:", duration)
    analysis.calculate_stats(stdout=True)
    return analysis


def plot_analysis(values, labels, analysis, plot_name):
    plot = plt.figure()
    plot.canvas.set_window_title(plot_name)
    plt.xlabel("Time")
    plt.ylabel("Value")

    # Draw "raw" values
    plt.plot(values, color="grey", zorder=5, alpha=0.7)

    # Draw upper and lower borders
    plt.plot(analysis.batch_upper_borders, format('-'), color="blue", zorder=10, alpha=0.2)
    plt.plot(analysis.batch_lower_borders, format('-'), color="blue", zorder=10, alpha=0.2)

    # Draw forecasts
    plt.plot(analysis.batch_forecasts, format('-'), color="blue", zorder=6, alpha=0.6)

    # Draw anomalous region
    plt.fill_between(range(len(values)), plt.ylim()[0] * 0.95, plt.ylim()[1] * 0.95, where=np.array(labels) == 1.0, color='red', alpha=0.2, zorder=3)

    # Draw True and False Positives
    for index, anomaly_detected in enumerate(analysis.batch_anomalies):
        detected_type = None
        value = None
        if len(analysis.batch_detected_types) > index:
            detected_type = analysis.batch_detected_types[index]
        if len(values) > index:
            value = values[index]
        if anomaly_detected and detected_type is detect.DetectTypes.true_positive:
            plt.plot(index, value, format('o'), color="green", alpha=0.8, zorder=10)
        elif anomaly_detected and detected_type is detect.DetectTypes.false_positive:
            plt.plot(index, value, format('o'), color="red", alpha=0.8, zorder=10)

    # Save Plot
    plt.margins(0.05)
    plot.savefig("./output/pdf/" + plot_name + ".pdf")
    plt.show()


if __name__ == "__main__":
    main()
