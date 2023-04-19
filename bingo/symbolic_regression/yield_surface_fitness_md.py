import logging
import numpy as np

from ..evaluation.fitness_function import VectorBasedFunction

LOGGER = logging.getLogger(__name__)


class YieldSurfaceFitnessMD(VectorBasedFunction):
    def __init__(self, training_data):
        super().__init__(training_data)
        self.training_data = training_data

    def evaluate_fitness_vector(self, individual):
        self.eval_count += 1
        mapping = individual.evaluate_equation_at([self.training_data])
        err = np.abs(self.training_data.transpose(0, 2, 1) @ mapping @ self.training_data - 1)
        # return err / np.linalg.norm(self.training_data, axis=(1, 2))
        return err.flatten()
