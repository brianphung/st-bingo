# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import pytest

from bingo.symbolic_regression.agraphMD.operator_definitions import *
from bingo.symbolic_regression.agraphMD.evaluation_backend import evaluation_backend

OPERATOR_LIST = [INTEGER, VARIABLE, CONSTANT, ADDITION, SUBTRACTION, TRANSPOSE]


@pytest.fixture
def sample_x():
    return np.array([[-1.0,
                      [[-1, 2, -3],
                       [4, -5, 6],
                       [-7, 8, -9]],
                      [[8, 7, 6],
                       [5, 4, 3],
                       [2, 1, 0]],
                      3.0]], dtype=object)


@pytest.fixture
def sample_constants():
    return np.array([[[1, 1, 1],
                      [2, 2, 2],
                      [3, 3, 3]], 3.14])


def _terminal_evaluations(terminal, x, constants, dims):
    # assumes parameter is 0
    if terminal == INTEGER:
        return np.zeros(dims)
    elif terminal == VARIABLE:
        return x[:, 0]
    elif terminal == CONSTANT:
        return constants[0]
    raise NotImplementedError("No test for terminal: %d" % terminal)


def _function_evaluations(function, a, b):
    # returns: f(a,b)
    # or if the function takes only a single parameter: f(a)
    if function == ADDITION:
        return a + b
    elif function == SUBTRACTION:
        return a - b
    elif function == MULTIPLICATION:
        if np.shape(a) and np.shape(b):
            return a @ b
        else:
            return a * b
    elif function == TRANSPOSE:
        return np.transpose(a)
    raise NotImplementedError("No test for operator: %d" % function)


@pytest.mark.parametrize("operator", OPERATOR_LIST)
def test_operator_evaluate(sample_x, sample_constants, operator):
    if IS_TERMINAL_MAP[operator]:
        expected_outcome = _terminal_evaluations(operator, sample_x,
                                                 sample_constants, (3, 3))
    else:
        expected_outcome = _function_evaluations(operator,
                                                 sample_x[:, 0], sample_x[:, 1])

    stack = np.array([[VARIABLE, 0, 3, 3],
                      [VARIABLE, 1, 3, 3],
                      [operator, 0, 1, 0]])
    print(sample_x[:, 0])
    print(sample_x[:, 1])
    f_of_x = evaluation_backend.evaluate(stack, sample_x, sample_constants)
    np.testing.assert_allclose(expected_outcome, f_of_x)

# TODO need to separate evaluate testing into load, addition-like, multiplication, and arity-one
