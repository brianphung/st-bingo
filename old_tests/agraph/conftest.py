# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
import numpy as np
import pytest

from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.agraph.component_generator \
    import ComponentGenerator

try:
    from bingocpp.build import symbolic_regression as bingocpp
except ImportError:
    bingocpp = None


@pytest.fixture
def sample_component_generator():
    generator = ComponentGenerator(input_x_dimension=2,
                                   num_initial_load_statements=2,
                                   terminal_probability=0.4,
                                   constant_probability=0.5)
    generator.add_operator(2)
    generator.add_operator(6)
    return generator


@pytest.fixture
def sample_agraph_1():
    test_graph = AGraph()
    _set_sample_agraph_1_data(test_graph)
    return test_graph


@pytest.fixture
def sample_agraph_1_cpp():
    if bingocpp is None:
        return None
    test_graph = bingocpp.AGraph()
    _set_sample_agraph_1_data(test_graph)
    return test_graph


def _set_sample_agraph_1_data(test_graph):
    test_graph.command_array = np.array([[0, 0, 0],  # sin(X_0 + 2.0) + 2.0
                                         [1, 0, 0],
                                         [2, 0, 1],
                                         [6, 2, 2],
                                         [2, 0, 1],
                                         [2, 3, 1]])
    test_graph.genetic_age = 10
    _ = test_graph.needs_local_optimization()
    test_graph.set_local_optimization_params([2.0, ])
    test_graph.fitness = 1


@pytest.fixture(
    params=["python",
            pytest.param("cpp",
                         marks=pytest.mark.skipif(not bingocpp,
                                                  reason='BingoCpp import err')
                         )])
def sample_agraph_1_list(request, sample_agraph_1, sample_agraph_1_cpp):
    if request.param == "python":
        return sample_agraph_1
    return sample_agraph_1_cpp


@pytest.fixture
def sample_agraph_2():
    test_graph = AGraph()
    return _set_sample_agraph_2_data(test_graph)


@pytest.fixture
def sample_agraph_2_cpp():
    if bingocpp is None:
        return None
    test_graph = bingocpp.AGraph()
    return _set_sample_agraph_2_data(test_graph)


def _set_sample_agraph_2_data(test_graph):
    test_graph.command_array = np.array([[0, 1, 3],  # sin((c_1-c_1)*X_1)
                                         [1, 1, 1],
                                         [3, 1, 1],
                                         [4, 0, 2],
                                         [2, 0, 1],
                                         [6, 3, 0]], dtype=int)
    test_graph.genetic_age = 20
    _ = test_graph.needs_local_optimization()
    test_graph.set_local_optimization_params([1.0])
    test_graph.fitness = 2
    return test_graph

