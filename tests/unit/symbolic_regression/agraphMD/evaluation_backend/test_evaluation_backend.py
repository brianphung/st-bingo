# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import pytest

from bingo.symbolic_regression.agraphMD.operator_definitions import *
from bingo.symbolic_regression.agraphMD.evaluation_backend import evaluation_backend

OPERATOR_LIST = [INTEGER, VARIABLE, CONSTANT, ADDITION, SUBTRACTION,
                 MULTIPLICATION, TRANSPOSE]


@pytest.fixture
def all_funcs_command_array():
    return np.array([[INTEGER, 5, 2, 3],
                     [VARIABLE, 0, 2, 3],
                     [CONSTANT, 0, 3, 2],
                     [TRANSPOSE, 2, 0, 0],
                     [ADDITION, 1, 0, 0],
                     [SUBTRACTION, 3, 4, 0],
                     [TRANSPOSE, 1, 0, 0],
                     [MULTIPLICATION, 5, 6, 0]])


@pytest.fixture
def multi_var_command_array():
    return np.array([[VARIABLE, 0, 2, 3],
                     [VARIABLE, 1, 4, 3],
                     [CONSTANT, 0, 3, 2],
                     [CONSTANT, 1, 3, 4],
                     [MULTIPLICATION, 2, 0, 0],
                     [MULTIPLICATION, 3, 1, 0],
                     [ADDITION, 4, 5, 0]])


@pytest.fixture
def sample_x():
    x_0 = np.linspace(-1.0, 0.0, 3*2*3).reshape((-1, 2, 3))
    x_1 = np.linspace(0.0, 1.0, 3*4*3).reshape((-1, 4, 3))
    return [x_0, x_1]  # combined[:, i] = x_i


@pytest.fixture
def sample_constants():
    return [np.array([[1, 2],
                      [3, 4],
                      [5, 6]]),
            np.array([[1, 2, 3, 4],
                      [2, 3, 4, 5],
                      [3, 4, 5, 6]])]


def test_all_funcs_eval(sample_x, sample_constants, all_funcs_command_array):
    fives = 5.0 * np.ones((3, 2))
    expected_f_of_x = (sample_constants[0].T - (sample_x[0] + fives.T)) @ np.transpose(sample_x[0], (0, 2, 1))
    f_of_x = evaluation_backend.evaluate(all_funcs_command_array,
                                         sample_x, sample_constants)
    np.testing.assert_array_almost_equal(f_of_x, expected_f_of_x)


def test_higher_dim_func_eval(sample_x, sample_constants, multi_var_command_array):
    c_0, c_1 = sample_constants
    expected_f_of_x = c_0 @ sample_x[0] + c_1 @ sample_x[1]
    f_of_x = evaluation_backend.evaluate(multi_var_command_array,
                                         sample_x, sample_constants)
    np.testing.assert_array_almost_equal(f_of_x, expected_f_of_x)