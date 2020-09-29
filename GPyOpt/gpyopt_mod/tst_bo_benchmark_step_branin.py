# <font color=red>-------------------------------------------------------------</font>
# ## Algorithm Testing: 2D Target Function
# <font color=red>-------------------------------------------------------------</font>
#
# Now we increase the dimension of the space by one and create a target 2-D function with multiple local minima to
# test and visualize how Bayesian Optimization works in higher dimensions. The target function we will try to maximize
# is the Branin function:
#
# $$f(x_1,x_2) = a(x_2 - b x_1^2 + c x_1 - r)^2 + s(1-t)\cos(x_1) + s + 5 x_1. $$
#
# It has two local minima and one global minima (with parameters $a=1, b=5.1/(4\pi)^2, c=5/\pi, r=6, s=10, $ and
# $t=1/(8\pi)$ We will restrict the interval of interest to $x_1 \in [-5, 10], x_2 \in [0,15]$.

from GPyOpt.gpyopt_mod.gpyopt_wrapper import run_step_by_step_original, run_step_by_step_custom

# import pytest
import numpy as np
import random
import time
# from matplotlib import pyplot as plt


def target(x1, x2):
    return -1*branin(x1, x2)


def branin(x1, x2):
    a = 1.
    b = 5.1 / (4. * np.pi ** 2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8. * np.pi)
    ret = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s + 5 * x1
    return ret


def get_sample_grid_2d(x_min, x_max, interval_x, y_min, y_max, interval_y):
    s1 = np.arange(x_min, x_max, interval_x)
    s2 = np.arange(y_min, y_max, interval_y)
    g_temp = np.meshgrid(s2, s1)
    g = np.append(g_temp[1].reshape(-1, 1), g_temp[0].reshape(-1, 1), axis=1)
    return g


# def plot_target(g, v):
#     fig = plt.figure()  # (figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_trisurf(g[:, 0], g[:, 1], v.flatten())
#     plt.grid(True)
#     plt.show()


def target_maximum(x, y):
    mx = (-3.75, 13.75)
    y_mx = target(mx[0], mx[1])
    # print("target theoretical maximum should be at x =", mx, ", y =", y_mx)

    y_mxa = np.max(y)
    ind_mxa = np.argmax(y)
    x_mxa = x[ind_mxa]
    # print("target actual maximum is at x =", x[ind_mna], ", y =", mxa, ", with index of x =", ind_mxa)

    return (mx, y_mx), (x_mxa, y_mxa, ind_mxa)


def tst_bo_benchmark_step_branin(is_custom):
    random.seed(1000)
    # target information
    grid = get_sample_grid_2d(-5, 10, 0.25, 0, 15, 0.25)
    N = grid.shape[0]
    # x = np.atleast_2d(x)
    x1 = grid[:, 0]
    x2 = grid[:, 1]
    values = target(x1, x2).reshape((N, 1))
    # plot_target(grid, values)
    minfo = target_maximum(grid, values)
    print()
    print("target theoretical maximum should be at x =", minfo[0][0], ", y =", minfo[0][1])
    print("target actual maximum is at x =", minfo[1][0], ", y =", minfo[1][1], ", with index of x =", minfo[1][2])

    # input parameters:
    random.seed(0)
    init_points = 10
    max_iters = 100
    maximize = True
    print_all = 0

    x_step = 0.25
    domain_1 = [x * x_step for x in range(-20, 40)]
    domain_2 = [x * x_step for x in range(0, 60)]
    p_bounds = [{'name': 'x1', 'type': 'discrete', 'domain': domain_1},
                {'name': 'x2', 'type': 'discrete', 'domain': domain_2}]

    N = grid.shape[0]
    all_indexes = np.arange(N)
    X_init_indexes = np.random.choice(all_indexes, size=init_points, replace=False)
    X_init = grid[X_init_indexes, :]
    Y_init = values[X_init_indexes, :]

    # # d_n <= d_n-1 -> d_n - d_n-1 <= 0
    constraints = [{'name': 'const_' + str(i),
                    'constraint': "x[:, " + str(i+1) + "] - x[:, " + str(i) + "]"}
                   for i in range(0, 1)]

    # run bo
    current_iter = 0
    X_step = X_init
    Y_step = Y_init

    if is_custom:
        bo = run_step_by_step_custom
    else:
        bo = run_step_by_step_original

    while current_iter < max_iters:
        x_next = bo(black_box_function=target,
                     constraints=constraints,
                     domain=p_bounds,
                     X=X_step,
                     Y=Y_step,
                     maximize=maximize,
                     precomputed_space=grid,
                     log_all=print_all)
        y_next = target(x_next[0][0], x_next[0][1])

        X_step = np.vstack((X_step, x_next))
        Y_step = np.vstack((Y_step, y_next))

        current_iter += 1

    best_index = np.argmax(Y_step)
    best_x = X_step[best_index]
    best_y = Y_step[best_index][0]
    print("found target maximum is at x, y =", best_x, best_y)

    # np.testing.assert_almost_equal(best_x[0], minfo[1][0][0], decimal=0)
    # np.testing.assert_almost_equal(best_x[1], minfo[1][0][1], decimal=0)
    # np.testing.assert_almost_equal(best_y, minfo[1][1], decimal=0)
    # assert best_index == minfo[1][2]


if __name__ == "__main__":
    start_time = time.time()
    tst_bo_benchmark_step_branin(is_custom=True)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    tst_bo_benchmark_step_branin(is_custom=False)
    print("--- %s seconds ---" % (time.time() - start_time))
