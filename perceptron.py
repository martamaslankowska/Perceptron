import random
import numpy as np
import copy

# losowanie wag wejściowych
# podanie wzorców uczących
    # obliczenie całkowitego pobudzenia nauronu - sumy
    # zastosowanie funkcji aktywacji
    # obliczenie wyjścia
# skorygowanie wag na podstawie błędu


n = 2  # number of neurons on the first layer


def f(z):
    if z >= 0:
        return 1
    else:
        return 0


def sum_xw(x, w):
    return np.sum(x*w)  # the same as np.dot(x,w)


def return_yz_and(x1, x2):
    return 1 if x1 & x2 else 0


def return_yz_given(i, y):
    return y[i]


def count_error(o, oz):
    return oz - o


def correct_weights(x, w, error, alpha, stop):
        correct_arr = x * error * alpha
        # print('x:', x)
        # print('correct:', correct_arr)
        # print('old w:', w)
        new_w = w + correct_arr
        # print('new w:', w)
        if not np.array_equal(w, new_w):
            stop = 0
        return new_w, stop


def train(x_all, y_all, w, alpha, x_bias):
    e = 0
    stop = 0
    while stop < 1:
        stop = 1
        e = e+1
        print('\nEpoch', e)
        for i in range(x_all.shape[0]):  # whole epoch
            x = np.append([x_bias], x_all[i])  # every step in epoch, each case
            # print('x:', x, ' |  w:', w)
            z = sum_xw(x, w)
            # print('z:', z)
            out = f(z)
            # print('out:', out)
            yz = return_yz_given(i, y_all)  # yz = return_yz_and(x[1], x[2])
            # print('yz:', yz)
            err = count_error(out, yz)
            # print('error:', err)
            w, stop = correct_weights(x, w, err, alpha, stop)
            # print('new w:', w)

            # print('stop?', stop)
            # print('\n')
        # stop = 1

    # printing the result :)
    print('\n-------------------------\n')
    print('Weights on the end:')
    for i in range(n+1):
        print(f'w{i} = {w[i]}')

    return w


def test_model(x_test, y_test, w, x_bias):
    sum_of_errors = 0
    for i in range(x_test.shape[0]):
        x = np.append([x_bias], x_test[i])
        # print('x:', x, ' |  w:', w)
        z = sum_xw(x, w)
        # print('z:', z)
        out = f(z)
        # print('out:', out)
        yz = return_yz_given(i, y_test)  # yz = return_yz_and(x[1], x[2])
        # print('yz:', yz)
        err = count_error(out, yz)
        # print('error:', err)
        if err != 0:
            sum_of_errors += 1

    accuracy = 1 - sum_of_errors/x_test.shape[0]
    print('Accuracy: {0:.0f}%'.format(accuracy*100))

