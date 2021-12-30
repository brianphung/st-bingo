import numpy as np
from bingo.local_optimizers.continuous_local_opt_md import ContinuousLocalOptimizationMD

from bingo.symbolic_regression.explicit_regression_md import ExplicitTrainingDataMD, ExplicitRegressionMD

from bingo.symbolic_regression.agraphMD.agraphMD import AGraphMD
from bingo.symbolic_regression.agraphMD.operator_definitions import *

if __name__ == '__main__':
    C = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    x = np.array([[[0], [1], [2]]])
    # x = np.array([[list(range(0, 100, 3)), list(range(1, 101, 3)), list(range(2, 102, 3))]])
    y = C @ x
    print(y)

    etd = ExplicitTrainingDataMD(x, y)
    fitness = ExplicitRegressionMD(etd, metric="mse")
    clo = ContinuousLocalOptimizationMD(fitness, algorithm="BFGS", param_init_bounds=[0, 0])

    test_graph = AGraphMD()
    test_graph.command_array = np.array([[CONSTANT, 0, 3, 3],
                                         [VARIABLE, 0, 3, 1],
                                         [MULTIPLICATION, 0, 1, 0]], dtype=int)
    print(test_graph.evaluate_equation_at(x))
    print(fitness(test_graph))
    test_graph._needs_opt = True

    clo(test_graph)
    print(test_graph.constants)
    print(test_graph.evaluate_equation_at(x))
