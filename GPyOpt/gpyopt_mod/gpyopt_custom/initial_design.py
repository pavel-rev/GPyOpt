from GPyOpt.experiment_design.grid_design import GridDesign
from GPyOpt.experiment_design.latin_design import LatinDesign
from GPyOpt.experiment_design.sobol_design import SobolDesign
from GPyOpt.experiment_design.random_design import RandomDesign

from .random_design_from_precomputed_space import RandomDesignFromPrecomputedSpace


def initial_design(design_name, space, init_points_count, precomputed_space):
    if design_name == 'random_w_precomputed_space':
        design = RandomDesignFromPrecomputedSpace(space, precomputed_space)
    elif design_name == 'random':
        design = RandomDesign(space)
    elif design_name == 'sobol':
        design = SobolDesign(space)
    elif design_name == 'grid':
        design = GridDesign(space)
    elif design_name == 'latin':
        design = LatinDesign(space)
    else:
        raise ValueError('Unknown design type: ' + design_name)

    return design.get_samples(init_points_count)
