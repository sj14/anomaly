import math

import numpy as np
import matplotlib.pyplot as plt

# Attribution: Many ideas taken from: https://github.com/martin-magakian/Anomaly-Detection-test


# strange_num = -1 adds the anomaly label to all points from strangePos to the end of the cycle
# strange_noise = -1 using the same noise as the normal "noise"
# strange_static ignores the normal cyclic value
def createCycle(amount=1, step_size=0.02, cycle_size=1.0, noise=0.0, strange_pos=0, strange_noise=-1, strange_num=0, strange_shift=0.0, strange_static=False):
    if strange_noise == -1:
        strange_noise = noise

    values = []
    labels = []

    for i in range(amount):
        points = np.arange(0.0, math.pi, step_size)
        points = points.tolist()

        for i, val in enumerate(points):
            if i > strange_pos and i <= strange_pos+strange_num or strange_num == -1:
                labels.append(True)
                noise_applied = np.random.standard_normal(size=1) * strange_noise + strange_shift

                if strange_static is True:
                    noise_applied = noise_applied - math.sin(val)
            else:
                labels.append(False)
                noise_applied = np.random.standard_normal(size=1) * noise

            sin_applied = math.sin(val)
            values.append((sin_applied + float(noise_applied)) * cycle_size)

    return values, labels




def point_anomaly():
    np.random.seed(1)

    values = []
    labels = []

    for i in range(1, 60, 1):
        if i == 50:
            values.append(1100)
            labels.append(False)
            continue

        values.append(abs(np.random.randint(935, 985, 1)))
        labels.append(False)

    np.random.seed()
    return values, labels


def contextual_anomaly():
    np.random.seed(1)

    values = []
    labels = []

    for i in range(1, 230, 1):
        if i == 50:
            values.append(1100)
            labels.append(False)
            continue

        if i == 100:
            values.append(1095)
            labels.append(False)
            continue

        if i == 150:
            values.append(1105)
            labels.append(False)
            continue

        if i == 200:
            values.append(1101)
            labels.append(False)
            continue

        if i == 210:
            values.append(1098)
            labels.append(False)
            continue

        values.append(abs(np.random.randint(935, 985, 1)))
        labels.append(False)

    np.random.seed()
    return values, labels





def contextual_anomaly2():
    np.random.seed(1)

    values = []
    labels = []

    d1 = createCycle(amount=3, noise=0.05)
    values.extend(d1[0])
    labels.extend(d1[1])

    d2 = createCycle(noise=0.05, strange_pos=80, strange_num=1, strange_noise=0.1, strange_shift=-1)
    values.extend(d2[0])
    labels.extend(d2[1])

    np.random.seed()
    return values, labels


def noise_nonoise():
    np.random.seed(1)

    values = []
    labels = []

    d1 = createCycle(amount=1, noise=0.05, strange_pos=80, strange_num=150, strange_noise=0.0,)
    values.extend(d1[0])
    labels.extend(d1[1])

    np.random.seed()
    return values, labels



def collective_anomaly():
    np.random.seed(1)

    values = []
    labels = []

    d1 = createCycle(amount=3, noise=0.05)
    values.extend(d1[0])
    labels.extend(d1[1])

    d2 = createCycle(step_size=0.04, noise=0.05)
    values.extend(d2[0])
    labels.extend(d2[1])

    d3 = createCycle(amount=1, noise=0.05)
    values.extend(d3[0])
    labels.extend(d3[1])

    np.random.seed()
    return values, labels




def cyclic_bump():
    np.random.seed(1)

    values = []
    labels = []

    d1 = createCycle(amount=5, noise=0.05)
    values.extend(d1[0])
    labels.extend(d1[1])

    d2 = createCycle(noise=0.05, strange_pos=0, strange_num=5, strange_noise=0.1, strange_shift=0.5)
    values.extend(d2[0])
    labels.extend(d2[1])

    d3 = createCycle(amount=1, noise=0.05)
    values.extend(d3[0])
    labels.extend(d3[1])

    np.random.seed()
    return values, labels

def bump_to_early():
    np.random.seed(1)

    values = []
    labels = []

    d1 = createCycle(amount=5, noise=0.05)
    values.extend(d1[0])
    labels.extend(d1[1])

    d2 = createCycle(noise=0.05, strange_pos=30, strange_num=5, strange_noise=0.01, strange_shift=0.3)
    values.extend(d2[0])
    labels.extend(d2[1])

    d3 = createCycle(amount=1, noise=0.05)
    values.extend(d3[0])
    labels.extend(d3[1])

    np.random.seed()
    return values, labels


def trend():
    np.random.seed(1)

    values = []
    labels = []

    d1 = createCycle(amount=7, noise=0.05)
    labels.extend(d1[1])

    for i, val in enumerate(d1[0]):
        value_d1 = val + i/10
        values.append(value_d1)

    np.random.seed()
    return values, labels


# TODO: NOT WORKING, return types are different than from the other functions?
def increasing_line():
    np.random.seed(1)

    values = []
    labels = []

    for i in range(1000):
        values.append(i)
        labels.append(False)

    np.random.seed()
    return values, labels



def grow_suddenly():
    np.random.seed(1)

    values = []
    labels = []

    d1 = createCycle(amount=5, noise=0.05)
    values.extend(d1[0])
    labels.extend(d1[1])

    d2 = createCycle(cycle_size=1.15, noise=0.05*1.15, strange_pos=0, strange_num=-1)
    values.extend(d2[0])
    labels.extend(d2[1])

    d3 = createCycle(amount=1, noise=0.05)
    values.extend(d3[0])
    labels.extend(d3[1])

    np.random.seed()
    return values, labels


def more_noise():
    np.random.seed(1)

    values = []
    labels = []

    d1 = createCycle(amount=5, noise=0.05)
    values.extend(d1[0])
    labels.extend(d1[1])

    d2 = createCycle(strange_pos=0, strange_num=-1, strange_noise=0.15)
    values.extend(d2[0])
    labels.extend(d2[1])

    d3 = createCycle(amount=1, noise=0.05)
    values.extend(d3[0])
    labels.extend(d3[1])

    np.random.seed()
    return values, labels


def cyclic_sagged():
    np.random.seed(1)

    values = []
    labels = []

    d1 = createCycle(amount=5, noise=0.05)
    values.extend(d1[0])
    labels.extend(d1[1])

    d2 = createCycle(noise=0.05, strange_pos=60, strange_num=40, strange_shift=0.7, strange_static=True)
    values.extend(d2[0])
    labels.extend(d2[1])

    d3 = createCycle(amount=1, noise=0.05)
    values.extend(d3[0])
    labels.extend(d3[1])

    np.random.seed()
    return values, labels


def cyclic_level_shift():
    np.random.seed(1)

    values = []
    labels = []

    d1 = createCycle(amount=5, noise=0.05)
    values.extend(d1[0])
    labels.extend(d1[1])

    d2 = createCycle(noise=0.05, strange_pos=60, strange_num=40, strange_shift=0.7, strange_static=True)
    values.extend(d2[0])
    labels.extend(d2[1])

    d3 = createCycle(amount=1, noise=0.05)
    values.extend(d3[0])
    labels.extend(d3[1])

    np.random.seed()
    return values, labels


def remove_noise():
    np.random.seed(1)

    values = []
    labels = []

    d1 = createCycle(amount=5, noise=0.05)
    values.extend(d1[0])
    labels.extend(d1[1])

    d2 = createCycle(noise=0.05, strange_pos=0, strange_num=-1, strange_noise=0)
    values.extend(d2[0])
    labels.extend(d2[1])

    d3 = createCycle(amount=5, noise=0.05)
    values.extend(d3[0])
    labels.extend(d3[1])

    np.random.seed()
    return values, labels


def small_change():
    np.random.seed(1)
    values = []
    labels = []

    d1 = createCycle(amount=5, noise=0.0)
    values.extend(d1[0])
    labels.extend(d1[1])

    d2 = createCycle(noise=0.0, strange_pos=50, strange_num=8, strange_noise=0.02)
    values.extend(d2[0])
    labels.extend(d2[1])

    d3 = createCycle(amount=5, noise=0.0)
    values.extend(d3[0])
    labels.extend(d3[1])

    np.random.seed()
    return values, labels

def stop_suddenly_zero():
    np.random.seed(1)

    values = []
    labels = []

    d1 = createCycle(amount=14, noise=0.05)
    values.extend(d1[0])
    labels.extend(d1[1])

    d2 = createCycle(noise=0.05, strange_pos=80, strange_num=80, strange_shift=0, strange_noise=0, strange_static=True)
    values.extend(d2[0])
    labels.extend(d2[1])

    d3 = createCycle(amount=14, noise=0.05)
    values.extend(d3[0])
    labels.extend(d3[1])

    np.random.seed()
    return values, labels


def stop_suddenly_one():
    np.random.seed(1)

    values = []
    labels = []

    d1 = createCycle(amount=5, noise=0.05)
    values.extend(d1[0])
    labels.extend(d1[1])

    d2 = createCycle(amount=1, noise=0.05, strange_pos=80, strange_num=80, strange_shift=1, strange_noise=0, strange_static=True)
    values.extend(d2[0])
    labels.extend(d2[1])

    d3 = createCycle(amount=5, noise=0.05)
    values.extend(d3[0])
    labels.extend(d3[1])

    np.random.seed()
    return values, labels


def white_noise_shift():
    np.random.seed(1)

    values = []
    labels = []

    d1 = createCycle(noise=0.15, strange_pos=0, strange_num=-1, strange_static=True)
    values.extend(d1[0])
    labels.extend(d1[1])

    d2 = createCycle(noise=0.15, strange_pos=0, strange_num=-1, strange_shift=5, strange_static=True)
    values.extend(d2[0])
    labels.extend(d2[1])

    d3 = createCycle(noise=0.15, strange_pos=0, strange_num=-1, strange_shift=10, strange_static=True)
    values.extend(d3[0])
    labels.extend(d3[1])

    d4 = createCycle(noise=0.15, strange_pos=0, strange_num=-1, strange_shift=5, strange_static=True)
    values.extend(d4[0])
    labels.extend(d4[1])

    d5 = createCycle(noise=0.15, strange_pos=0, strange_num=-1, strange_shift=0, strange_static=True)
    values.extend(d5[0])
    labels.extend(d5[1])

    np.random.seed()
    return values, labels


def white_noise():
    np.random.seed(1)

    values = []
    labels = []

    d1 = createCycle(amount=5, noise=0.15, strange_pos=0, strange_num=-1, strange_static=True)
    values.extend(d1[0])
    labels.extend(d1[1])

    np.random.seed()
    return values, labels



def grow_with_error():
    np.random.seed(1)

    values = []
    labels = []

    for i in range(1,6):
        d1 = createCycle(cycle_size=i, noise=0.05)
        values.extend(d1[0])
        labels.extend(d1[1])

    d6 = createCycle(cycle_size=6, noise=0.05, strange_pos=75, strange_num=1, strange_shift=-0.8)
    values.extend(d6[0])
    labels.extend(d6[1])

    d7 = createCycle(cycle_size=7, noise=0.05)
    values.extend(d7[0])
    labels.extend(d7[1])

    np.random.seed()
    return values, labels


def cyclic():
    np.random.seed(1)

    values = []
    labels = []

    d1 = createCycle(amount=7, noise=0.05)
    values.extend(d1[0])
    labels.extend(d1[1])

    np.random.seed()
    return values, labels


def cyclic_diff_length():
    np.random.seed(1)

    values = []
    labels = []

    d1 = createCycle(amount=5, step_size=0.03, noise=0.05)
    values.extend(d1[0])
    labels.extend(d1[1])

    d2 = createCycle(amount=1, step_size=0.02, noise=0.05, strange_pos=0, strange_num=-1)
    values.extend(d2[0])
    labels.extend(d2[1])

    d3 = createCycle(amount=1, step_size=0.03, noise=0.05)
    values.extend(d3[0])
    labels.extend(d3[1])

    np.random.seed()
    return values, labels


def linear_growth():
    np.random.seed(1)

    values = []
    labels = []
    for i in range(1,8):
        d1 = createCycle(cycle_size=i, noise=0.2)
        values.extend(d1[0])
        labels.extend(d1[1])

    np.random.seed()
    return values, labels


def cyclic_weeks():
    np.random.seed(1)

    values = []
    labels = []

    d1 = createCycle(amount=5, noise=0.05)
    values.extend(d1[0])
    labels.extend(d1[1])

    d2 = createCycle(amount=2, cycle_size=1.5, noise=0.05)
    values.extend(d2[0])
    labels.extend(d2[1])

    d3 = createCycle(amount=5, noise=0.05)
    values.extend(d3[0])
    labels.extend(d3[1])

    d4 = createCycle(amount=2, cycle_size=1.5, noise=0.05)
    values.extend(d4[0])
    labels.extend(d4[1])

    d5 = createCycle(amount=5, noise=0.05)
    values.extend(d5[0])
    labels.extend(d5[1])

    d6 = createCycle(amount=2, cycle_size=1.5, noise=0.05)
    values.extend(d6[0])
    labels.extend(d6[1])

    np.random.seed()
    return values, labels


def cyclic_weeks_changing():
    np.random.seed(1)

    values = []
    labels = [] # no anomalies!!!

    d1 = createCycle(amount=5, noise=0.05)
    values.extend(d1[0])
    labels.extend(d1[1])

    d2 = createCycle(amount=2, cycle_size=1.5, noise=0.05)
    values.extend(d2[0])
    labels.extend(d2[1])

    d3 = createCycle(amount=5, noise=0.05, strange_pos=60, strange_num=40, strange_shift=0.7, strange_static=True)
    values.extend(d3[0])
    labels.extend(d1[1])

    d4 = createCycle(amount=2, cycle_size=1.5, noise=0.05, strange_pos=60, strange_num=40, strange_shift=0.7, strange_static=True)
    values.extend(d4[0])
    labels.extend(d2[1])

    d5 = createCycle(amount=5, noise=0.05, strange_pos=60, strange_num=40, strange_shift=0.7, strange_static=True)
    values.extend(d5[0])
    labels.extend(d1[1])

    d6 = createCycle(amount=2, cycle_size=1.5, noise=0.05, strange_pos=60, strange_num=40, strange_shift=0.7, strange_static=True)
    values.extend(d6[0])
    labels.extend(d2[1])

    np.random.seed()
    return values, labels



def cyclic_weeks_changing_long():
    np.random.seed(1)

    values = []
    labels = [] # no anomalies!!!

    d1 = createCycle(amount=5, noise=0.05)
    values.extend(d1[0])
    labels.extend(d1[1])

    d2 = createCycle(amount=2, cycle_size=1.5, noise=0.05)
    values.extend(d2[0])
    labels.extend(d2[1])

    d3 = createCycle(amount=5, noise=0.05, strange_pos=60, strange_num=40, strange_shift=0.7, strange_static=True)
    values.extend(d3[0])
    labels.extend(d1[1])

    d4 = createCycle(amount=2, cycle_size=1.5, noise=0.05, strange_pos=60, strange_num=40, strange_shift=0.7, strange_static=True)
    values.extend(d4[0])
    labels.extend(d2[1])

    d5 = createCycle(amount=5, noise=0.05, strange_pos=60, strange_num=40, strange_shift=0.7, strange_static=True)
    values.extend(d5[0])
    labels.extend(d1[1])

    d6 = createCycle(amount=2, cycle_size=1.5, noise=0.05, strange_pos=60, strange_num=40, strange_shift=0.7, strange_static=True)
    values.extend(d6[0])
    labels.extend(d2[1])

    d7 = createCycle(amount=5, noise=0.05, strange_pos=60, strange_num=40, strange_shift=0.7, strange_static=True)
    values.extend(d7[0])
    labels.extend(d1[1])

    d8 = createCycle(amount=2, cycle_size=1.5, noise=0.05, strange_pos=60, strange_num=40, strange_shift=0.7, strange_static=True)
    values.extend(d8[0])
    labels.extend(d2[1])

    d9 = createCycle(amount=5, noise=0.05, strange_pos=60, strange_num=40, strange_shift=0.7, strange_static=True)
    values.extend(d9[0])
    labels.extend(d1[1])

    d10 = createCycle(amount=2, cycle_size=1.5, noise=0.05, strange_pos=60, strange_num=40, strange_shift=0.7, strange_static=True)
    values.extend(d10[0])
    labels.extend(d2[1])

    d11 = createCycle(amount=5, noise=0.05, strange_pos=60, strange_num=40, strange_shift=0.7, strange_static=True)
    values.extend(d11[0])
    labels.extend(d1[1])

    d12 = createCycle(amount=2, cycle_size=1.5, noise=0.05, strange_pos=60, strange_num=40, strange_shift=0.7, strange_static=True)
    values.extend(d12[0])
    labels.extend(d2[1])


    np.random.seed()
    return values, labels





def linear_growth_no_noise():
    np.random.seed(1)

    values = []
    labels = []
    for i in range(1,8):
        d1 = createCycle(cycle_size=i, noise=0.0)
        values.extend(d1[0])
        labels.extend(d1[1])

    np.random.seed()
    return values, labels


def linear_growth_and_stop_grow():
    np.random.seed(1)

    values = []
    labels = []
    for i in range(1,8):
        d1 = createCycle(cycle_size=i, noise=0.2)
        values.extend(d1[0])
        labels.extend(d1[1])

    d2 = createCycle(amount=5, cycle_size=7, noise=0.2)
    values.extend(d2[0])
    labels.extend(d2[1])

    np.random.seed()
    return values, labels


def exponential_growth():
    np.random.seed(1)

    values = []
    labels = []
    for i in range(1,8):
        d1 = createCycle(cycle_size=math.pow(i, 2), noise=0.2)
        values.extend(d1[0])
        labels.extend(d1[1])

    np.random.seed()
    return values, labels


def exponential_growth_and_stop_grow():
    np.random.seed(1)

    values = []
    labels = []
    for i in range(1,8):
        d1 = createCycle(cycle_size=math.pow(i, 2), noise=0.2)
        values.extend(d1[0])
        labels.extend(d1[1])

    d2 = createCycle(amount=5, cycle_size=math.pow(7, 2), noise=0.2)
    values.extend(d2[0])
    labels.extend(d2[1])

    np.random.seed()
    return values, labels


def show_datasets():
    # Without anomalies
    plt.plot(cyclic()[0], color="black")
    plt.show()

    plt.plot(cyclic_diff_length()[0], color="black")
    plt.show()

    plt.plot(linear_growth()[0], color="black")
    plt.show()

    plt.plot(exponential_growth()[0], color="black")
    plt.show()


    # With anomalies
    plt.plot(cyclic_bump()[0], color="black")
    plt.show()

    plt.plot(bump_to_early()[0], color="black")
    plt.show()

    plt.plot(grow_suddenly()[0], color="black")
    plt.show()

    plt.plot(more_noise()[0], color="black")
    plt.show()

    plt.plot(cyclic_sagged()[0], color="black")
    plt.show()

    plt.plot(remove_noise()[0], color="black")
    plt.show()

    plt.plot(small_change()[0], color="black")
    plt.show()

    plt.plot(stop_suddenly_zero()[0], color="black")
    plt.show()

    plt.plot(stop_suddenly_one()[0], color="black")
    plt.show()

    plt.plot(grow_with_error()[0], color="black")
    plt.show()

    plt.plot(white_noise_shift()[0], color="black")
    plt.show()


#show_datasets()