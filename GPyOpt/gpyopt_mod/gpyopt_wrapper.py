# from GPyOpt import Design_space, acquisitions, methods, models, optimization
# from GPyOpt.core import evaluators
import logging

from GPyOpt.gpyopt_mod.gpyopt_custom.bayesian_optimization import BayesianOptimization as custom_bo
from GPyOpt.methods.bayesian_optimization import BayesianOptimization as original_bo


def run_step_by_step_custom(domain, X, Y, constraints=None, precomputed_space=None, maximize=False, log_all=False, black_box_function=None):
    """ Runs BayesianOptimization step by step. """
    bo_step = custom_bo(f=black_box_function,
                                           domain=domain,
                                           constraints=constraints,
                                           X=X,
                                           Y=Y,
                                           maximize=maximize,
                                           normalize_Y=True,
                                           de_duplication=True,
                                           initial_design_type="random",
                                           precomputed_space=precomputed_space)

    x_next = bo_step.suggest_next_locations()

    if log_all:
        logging.info(F"x_next is {x_next}")

    return x_next


def run_step_by_step_original(domain, X, Y, constraints=None, precomputed_space=None, maximize=False, log_all=False, black_box_function=None):
    """ Runs BayesianOptimization step by step. """
    bo_step = original_bo(f=black_box_function,
                                           domain=domain,
                                           constraints=constraints,
                                           X=X,
                                           Y=Y,
                                           maximize=maximize,
                                           normalize_Y=True,
                                           de_duplication=True,
                                           initial_design_type="random",
                                           precomputed_space=precomputed_space)

    x_next = bo_step.suggest_next_locations()

    if log_all:
        logging.info(F"x_next is {x_next}")

    return x_next


# def _run_step_by_step_modular(domain, X, Y, constraints=None, all_x_values=None, black_box_function=None,
#                               maximize=False, print_all=False):
#     """ Runs BayesianOptimization step by step in modular way. Experimental feature. """
#     if maximize:
#         raise Exception("this option is not supported in modular run option")
#
#     space = Design_space(space=domain, constraints=constraints)
#     space.all_x_values = all_x_values
#
#     model = models.GPModel(optimize_restarts=5, verbose=False)
#
#     acquisition_optimizer = optimization.AcquisitionOptimizer(space)
#
#     acquisition = acquisitions.EI.AcquisitionEI(model, space, acquisition_optimizer)
#
#     evaluator = evaluators.Sequential(acquisition)
#
#     bo = methods.ModularBayesianOptimization(model=model,
#                                              space=space,
#                                              objective=black_box_function,
#                                              acquisition=acquisition,
#                                              evaluator=evaluator,
#                                              X_init=X,
#                                              Y_init=Y)
#
#     x_next = bo.suggest_next_locations()
#
#     if print_all:
#         print("x_next is ", x_next)
#
#     return x_next
#
#
# def _run_in_loop(black_box_function, domain, init_points, max_iters, maximize=False, print_all=False):
#     """ Runs BayesianOptimization in loop. Experimental feature. """
#     optimizer = methods.BayesianOptimization(f=black_box_function,
#                                              domain=domain,
#                                              acquisition_type='EI',
#                                              initial_design_numdata=init_points,
#                                              exact_feval=True,
#                                              maximize=maximize
#                                              )
#     # # additional parameters for future:
#     # max_time = 60  # time budget
#     # eps = 10e-6  # Minimum allows distance between the las two observations
#     optimizer.run_optimization(max_iters)
#
#     best_x = optimizer.x_opt
#     best_y = optimizer.fx_opt
#
#     if print_all:
#         print(best_x)
#         print(best_y)
#
#     return best_x, best_y
