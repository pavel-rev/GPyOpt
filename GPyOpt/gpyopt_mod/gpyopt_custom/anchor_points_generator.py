import logging
import numpy as np

from GPyOpt.gpyopt_mod.gpyopt_custom.initial_design import initial_design
from GPyOpt.core.errors import FullyExploredOptimizationDomainError
from GPyOpt.core.task.space import Design_space
from GPyOpt.optimization.acquisition_optimizer import ObjectiveAnchorPointsGenerator

logger = logging.getLogger(__name__)


class ObjectiveAnchorPointsGeneratorWithPrecomputedSpace(ObjectiveAnchorPointsGenerator):

    def __init__(self, space, design_type, objective, precomputed_space, num_samples=1000):
        super(ObjectiveAnchorPointsGenerator, self).__init__(space, design_type, num_samples)
        self.objective = objective
        self.precomputed_space = precomputed_space

    def get_anchor_point_scores(self, X):

        return self.objective(X).flatten()

    def get(self, num_anchor=5, duplicate_manager=None, unique=False, context_manager=None):

        if context_manager and not self.space._has_bandit():
            space_configuration_without_context = [self.space.config_space_expanded[idx] for idx in context_manager.nocontext_index_obj]
            space = Design_space(space_configuration_without_context, context_manager.space.constraints)
            add_context = lambda x : context_manager._expand_vector(x)
        else:
            space = self.space
            add_context = lambda x: x

        X = initial_design(self.design_type, space, self.num_samples, self.precomputed_space)

        if unique:
            sorted_design = sorted(list({tuple(x) for x in X}))
            X = space.unzip_inputs(np.vstack(sorted_design))
        else:
            X = space.unzip_inputs(X)

        X = add_context(X)

        if duplicate_manager:
            is_duplicate = duplicate_manager.is_unzipped_x_duplicate
        else:
            # In absence of duplicate manager, we never detect duplicates
            is_duplicate = lambda _ : False

        non_duplicate_anchor_point_indexes = [index for index, x in enumerate(X) if not is_duplicate(x)]

        if not non_duplicate_anchor_point_indexes:
            raise FullyExploredOptimizationDomainError("No anchor points could be generated ({} used samples, {} requested anchor points).".format(self.num_samples,num_anchor))

        if len(non_duplicate_anchor_point_indexes) < num_anchor:
            # Since logging has not been setup yet, I do not know how to express warnings...I am using standard print for now.
            logger.warning("Expecting {} anchor points, only {} available.".format(num_anchor, len(non_duplicate_anchor_point_indexes)))

        X = X[non_duplicate_anchor_point_indexes,:]

        scores = self.get_anchor_point_scores(X)

        anchor_points = X[np.argsort(scores)[:min(len(scores),num_anchor)], :]

        return anchor_points
