import numpy as np

from bingo.evaluation.fitness_function import VectorBasedFunction


class MappingFitness(VectorBasedFunction):
    def __init__(self, training_data):
        super().__init__(training_data)

    def evaluate_fitness_vector(self, individual):
        mapping_matrices = individual.evaluate_equation_at(self.training_data.x)
        mapped_input = mapping_matrices.transpose((0, 2, 1)) @ self.training_data.x[0] @ mapping_matrices

        err = np.linalg.norm(self.training_data.y - mapped_input, axis=(1, 2))

        # relative error
        denom = np.linalg.norm(self.training_data.y, axis=(1, 2))
        both_zero = (err == 0) & (denom == 0)
        err /= denom
        err[both_zero] = 0

        return err
