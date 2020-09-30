import numpy as np

from GPyOpt.experiment_design.random_design import RandomDesign


class RandomDesignFromPrecomputedSpace(RandomDesign):
    def __init__(self, space, precomputed_space):
        super(RandomDesignFromPrecomputedSpace, self).__init__(space)
        self.precomputed_space = precomputed_space

    def get_samples_with_constraints(self, init_points_count):
        """
        Draw random samples from pre-computed search space.
        """
        if self.precomputed_space is None:
            raise Exception("precomputed_space is None.")

        if not isinstance(self.precomputed_space, np.ndarray):
            raise Exception("precomputed_space must be a numpy array.")

        N = self.precomputed_space.shape[0]
        all_indexes = np.arange(N)
        if init_points_count < N:
            sample_indexes = np.random.default_rng().choice(all_indexes,
                                                            size=init_points_count,
                                                            replace=False)
        else:
            sample_indexes = all_indexes

        samples = self.precomputed_space[sample_indexes, :]

        # validate samples
        valid_sample_indices = (self.space.indicator_constraints(samples) == 1).flatten()

        if sum(valid_sample_indices) != len(samples):
            print("Warning: Some points in precomputed search space all_x_values do not comply with constraints.")

        return samples[0:init_points_count, :]
