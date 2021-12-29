import numpy as np

from bingo.symbolic_regression.agraphMD.agraphMD import AGraphMD
from bingo.symbolic_regression.agraphMD.operator_definitions import *

if __name__ == '__main__':
    C = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    x = np.array([[[0], [1], [2]]])
    y = np.matmul(C, x[0])
    print(y)

    test_graph = AGraphMD()
    test_graph.command_array = np.array([[CONSTANT, 0, 3, 3],
                                         [VARIABLE, 0, 3, 1],
                                         [MULTIPLICATION, 0, 1, 0]], dtype=int)
    test_graph.set_local_optimization_params([C])
    print(test_graph.constants)

    print(test_graph.evaluate_equation_at(x))
