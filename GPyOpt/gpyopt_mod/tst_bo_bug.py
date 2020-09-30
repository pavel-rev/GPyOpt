from GPyOpt.gpyopt_mod.gpyopt_wrapper import run_step_by_step_original, run_step_by_step_custom

import numpy as np
import time

D = 60
N = 5000

c = np.random.randint(101, size=(D, 1))


def target(a):
    return a.dot(c)


def tst_bo_bug(is_custom):
    # target information
    global N
    grid = np.concatenate([np.random.randint(low=i*10, high=(i+1)*10, size=(N, 1)) for i in range(D, 0, -1)], axis=1)
    values = target(grid).reshape((N, 1))

    # input parameters:
    init_points = 10
    max_iters = 10
    maximize = True
    print_all = 0

    p_bounds = [{'name': 'x' + str(i), 'type': 'discrete', 'domain': [x for x in range(0, 101)]}
                for i in range(D)]

    all_indexes = np.arange(N)
    X_init_indexes = np.random.choice(all_indexes, size=init_points, replace=False)
    X_init = grid[X_init_indexes, :]
    Y_init = values[X_init_indexes, :]

    # # d_n <= d_n-1 -> d_n - d_n-1 <= 0
    constraints = [{'name': 'const_' + str(i),
                    'constraint': "x[:, " + str(i+1) + "] - x[:, " + str(i) + "]"}
                   for i in range(0, D-1)]

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
        y_next = target(x_next)

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
    tst_bo_bug(is_custom=True)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    tst_bo_bug(is_custom=False)
    print("--- %s seconds ---" % (time.time() - start_time))
