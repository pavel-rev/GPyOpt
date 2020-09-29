from GPyOpt.optimization.acquisition_optimizer import *
from .anchor_points_generator import ObjectiveAnchorPointsGeneratorWithPrecomputedSpace

random_with_precomputed_space_design_type = "random_w_precomputed_space"


class AcquisitionOptimizerWithPrecomputedSpace(AcquisitionOptimizer):
    def __init__(self, space, precomputed_space, optimizer='lbfgs', **kwargs):
        super().__init__(space, optimizer=optimizer, **kwargs)
        self.precomputed_space = precomputed_space

    def optimize(self, f=None, df=None, f_df=None, duplicate_manager=None):
        """
        Optimizes the input function.

        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.

        """
        self.f = f
        self.df = df
        self.f_df = f_df

        ## --- Update the optimizer, in case context has beee passed.
        self.optimizer = choose_optimizer(self.optimizer_name, self.context_manager.noncontext_bounds)

        ## --- Selecting the anchor points and removing duplicates
        if self.type_anchor_points_logic == max_objective_anchor_points_logic:
            anchor_points_generator = ObjectiveAnchorPointsGeneratorWithPrecomputedSpace(self.space, random_with_precomputed_space_design_type, f, self.precomputed_space)
        elif self.type_anchor_points_logic == thompson_sampling_anchor_points_logic:
            anchor_points_generator = ThompsonSamplingAnchorPointsGenerator(self.space, sobol_design_type, self.model)

        ## -- Select the anchor points (with context)
        anchor_points = anchor_points_generator.get(duplicate_manager=duplicate_manager, context_manager=self.context_manager)

        ## --- Applying local optimizers at the anchor points and update bounds of the optimizer (according to the context)
        optimized_points = [apply_optimizer(self.optimizer, a, f=f, df=None, f_df=f_df, duplicate_manager=duplicate_manager, context_manager=self.context_manager, space = self.space) for a in anchor_points]
        x_min, fx_min = min(optimized_points, key=lambda t:t[1])

        #x_min, fx_min = min([apply_optimizer(self.optimizer, a, f=f, df=None, f_df=f_df, duplicate_manager=duplicate_manager, context_manager=self.context_manager, space = self.space) for a in anchor_points], key=lambda t:t[1])

        return x_min, fx_min
