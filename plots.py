import matplotlib.pyplot as plt
from pylab import *


def perceptron_line(w):
    x1 = w[0]/-w[1]  # for x2 = 0
    x2 = w[0]/-w[2]  # for x1 = 0
    return [0, x1], [x2, 0]


def draw_plot(dataset, weights, y):
    fig = figure()
    ax = fig.add_subplot(111)

    if type(dataset) is list:
        for i in range(len(dataset)):
            lab = str(dataset[i][1][0]) + ', ' + str(dataset[i][1][1])
            plot(dataset[i][:, 0], dataset[i][:, 1], 'o', label=lab)
    else:
        plot(dataset[:, 0], dataset[:, 1], 'bo')

    x, y = perceptron_line(weights)
    plot(x, y, 'r')

    draw_axes(ax)
    plt.legend(loc=1, borderaxespad=1)
    grid()
    show()


def draw_axes(ax):
    left, right = ax.get_xlim()
    low, high = ax.get_ylim()
    arrow(left, 0, right - left, 0, length_includes_head=True, head_width=0.05)
    arrow(0, low, 0, high - low, length_includes_head=True, head_width=0.05)


