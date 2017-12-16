# This Python file uses the following encoding: utf-8
# encoding=utf8
import sys
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from anomaly import detect, forecasts, errors, thresholds, datagen
import NewDetection

reload(sys)
sys.setdefaultencoding('utf8')

def plot_point_anomaly():
    values, anomaly_labels = datagen.point_anomaly()
    plot_name = "point_anomaly"
    plot = plt.figure()
    plot.canvas.set_window_title(plot_name)
    plt.xlabel("transaction number")
    plt.ylabel("money spent in €")

    plt.plot(values, color="grey", zorder=5, alpha=0.7)
    plt.margins(0.05)

    plot.savefig("./output/pdf/" + plot_name + ".pdf")
    plt.show()

def plot_contextual_anomaly():
    values, anomaly_labels = datagen.contextual_anomaly()
    plot_name = "contextual_anomaly"
    plot = plt.figure()
    plot.canvas.set_window_title(plot_name)
    plt.xlabel("transaction number")
    plt.ylabel("money spent in €")

    plt.plot(values, color="grey", zorder=5, alpha=0.7)
    plt.margins(0.05)

    plot.savefig("./output/pdf/" + plot_name + ".pdf")
    plt.show()


def plot_contextual_anomaly2():
    values, anomaly_labels = datagen.contextual_anomaly2()
    plot_name = "contextual_anomaly2"
    plot = plt.figure()
    plot.canvas.set_window_title(plot_name)
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.plot(values, color="grey", zorder=5, alpha=0.7)
    plt.margins(0.05)

    plot.savefig("./output/pdf/" + plot_name + ".pdf")
    plt.show()



def plot_collective_anomaly():
    values, anomaly_labels = datagen.collective_anomaly()
    plot_name = "collective_anomaly"
    plot = plt.figure()
    plot.canvas.set_window_title(plot_name)
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.plot(values, color="grey", zorder=5, alpha=0.7)
    plt.margins(0.05)

    plot.savefig("./output/pdf/" + plot_name + ".pdf")
    plt.show()




def plot_white_noise():
    values, anomaly_labels = datagen.white_noise()
    plot_name = "white_noise"
    plot = plt.figure()
    plot.canvas.set_window_title(plot_name)
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.plot(values, color="grey", zorder=5, alpha=0.7)
    plt.margins(0.05)

    plot.savefig("./output/pdf/" + plot_name + ".pdf")
    plt.show()

def plot_cyclic_linear_trend():
    values, anomaly_labels = datagen.linear_growth_no_noise()
    plot_name = "cyclic_linear_trend"
    plot = plt.figure()
    plot.canvas.set_window_title(plot_name)
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.plot(values, color="grey", zorder=5, alpha=0.7)
    plt.margins(0.05)

    plot.savefig("./output/pdf/" + plot_name + ".pdf")
    plt.show()


def plot_standard_distribution():
    # https://stackoverflow.com/questions/10138085/python-pylab-plot-normal-distribution
    import matplotlib.mlab as mlab
    import math

    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * variance, mu + 3 * variance, 100)
    #plt.plot(x, mlab.normpdf(x, mu, sigma))

    plot_name = "standard_normal_distribution"
    plot = plt.figure()
    plot.canvas.set_window_title(plot_name)
    plt.xlabel("z")
    plt.ylabel("f(z)")

    plt.plot(x, mlab.normpdf(x, mu, sigma), color="grey", zorder=5, alpha=0.7)
    plt.margins(0.05)

    plot.savefig("./output/pdf/" + plot_name + ".pdf")
    plt.show()


def plot_linear_threshold():
    plot_name = "linear_threshold"
    plot = plt.figure()
    plot.canvas.set_window_title(plot_name)
    plt.xlabel("x")
    plt.ylabel("f(x)")

    plt.plot( [0,1,2,3], [0,0,1,1], color="black", zorder=5, alpha=1.)
    plt.margins(0.05)

    plot.savefig("./output/pdf/" + plot_name + ".pdf")
    plt.show()


def plot_binary_threshold():
    plot_name = "binary_threshold"
    plot = plt.figure()
    plot.canvas.set_window_title(plot_name)
    plt.xlabel("x")
    plt.ylabel("f(x)")

    plt.plot( [0,1,1,2], [0,0,1,1], color="black", zorder=5, alpha=1.)
    plt.margins(0.05)

    plot.savefig("./output/pdf/" + plot_name + ".pdf")
    plt.show()

def plot_sigmoid_threshold():
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    x = np.linspace(-10, 10, 100)

    plot_name = "sigmoid_threshold"
    plot = plt.figure()
    plot.canvas.set_window_title(plot_name)
    plt.xlabel("x")
    plt.ylabel("f(x)")

    plt.plot(x, sigmoid(x), color="black", zorder=5, alpha=1.)
    plt.margins(0.05)

    plot.savefig("./output/pdf/" + plot_name + ".pdf")
    plt.show()


def plot_noise_nonoise():
    values, anomaly_labels = datagen.noise_nonoise()
    plot_name = "noise_nonoise"
    plot = plt.figure()
    plot.canvas.set_window_title(plot_name)
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.plot(values, color="grey", zorder=5, alpha=0.7)
    plt.margins(0.05)

    plot.savefig("./output/pdf/" + plot_name + ".pdf")
    plt.show()

if __name__ == "__main__":
    #plot_binary_threshold()
    #plot_linear_threshold()
    #plot_sigmoid_threshold()
    #plot_standard_distribution()
    #plot_cyclic_linear_trend()
    #plot_white_noise()
    plot_point_anomaly()
    plot_contextual_anomaly()
    plot_contextual_anomaly2()
    plot_collective_anomaly()
    #plot_noise_nonoise()
