# anomaly
A simple Python framework to compare time series anomaly detection algorithms. Special interest is put into algorithms
which work in real-time and with continuous data streams.

The framework allows to exchange different function of the detection algorithm:
- The forecast function (e.g. moving average)
- The error function (e.g. absolute error or the RMSE)
- The threshold function (e.g. simple threshold or standard deviation)

Some of the implemented algorithms are:
- moving average
- naive
- seasonal naive
- single/double/triple exponential smoothing
- an own promising combination of seasonal naive and exponential smoothing (seasonal exponential smoothing)

To easily evaluate the algorithm, the package comes with a function to simply create own test data with injected
anomalies (datagen.py).


# Example Plots
The example folder shows how to use the framework and how to plot the results.
The following three plots were created based on the seasonal triple exponential smoothing forecast and a 3-sigma threshold.
The grey line indicates the raw value, the dark blue line the forecast, and the light blue lines the allowed range of normal values.
The green dots indicate true positive anomalies and red dots indicate false positives detections.

![cyclic_bump](example/output/png/cyclic_bump.png)

![cyclic_sagged](example/output/png/cyclic_sagged.png)

![grow_with_error](example/output/png/grow_with_error.png)


# TODO
- clean up
- switch to Python 3
- do proper packaging
