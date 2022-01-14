"""
This module provides the python implementation of the functions for each
mathematical nodes used in Agraph

Attributes
----------
FORWARD_EVAL_MAP : dictionary {int: function}
                   A map of node number to evaluation function
"""

import numpy as np

from bingo.symbolic_regression.agraphMD.operator_definitions \
    import INTEGER, VARIABLE, CONSTANT, ADDITION, SUBTRACTION, MULTIPLICATION, TRANSPOSE


np.seterr(divide='ignore', invalid='ignore')


# Note (David Randall): Using classes here since changing
# all the signatures one by one with functions is a pain
class _ForwardEvalBase:
    @staticmethod
    def evaluate(param1, param2, param3, x, constants, forward_eval):
        raise NotImplementedError


# Integer value
class _IntegerForwardEval(_ForwardEvalBase):
    @staticmethod
    def evaluate(param1, param2, param3, x, constants, forward_eval):
        dims = (param2, param3)
        if dims == (0, 0):
            return float(param1)
        else:
            return float(param1) * np.ones((param2, param3))


# Load x column
class _LoadXForwardEval(_ForwardEvalBase):
    @staticmethod
    def evaluate(param1, param2, param3, x, constants, forward_eval):
        return x[param1]


class _LoadCForwardEval(_ForwardEvalBase):
    @staticmethod
    def evaluate(param1, param2, param3, x, constants, forward_eval):
        return np.array([constants[param1]] * len(x[0]))


class _AddForwardEval(_ForwardEvalBase):
    @staticmethod
    def evaluate(param1, param2, param3, x, constants, forward_eval):
        return forward_eval[param1] + forward_eval[param2]


class _SubtractForwardEval(_ForwardEvalBase):
    @staticmethod
    def evaluate(param1, param2, param3, x, constants, forward_eval):
        return forward_eval[param1] - forward_eval[param2]

"""
class _MultiplyForwardEval(_ForwardEvalBase):
    @staticmethod
    def evaluate(param1, param2, param3, x, constants, forward_eval):
        if np.shape(forward_eval[param1]) and np.shape(forward_eval[param2]):
            return np.matmul(forward_eval[param1], forward_eval[param2])
        else:
            return forward_eval[param1] * forward_eval[param2]
"""

class _MultiplyForwardEval(_ForwardEvalBase):
    @staticmethod
    def evaluate(param1, param2, param3, x, constants, forward_eval):
        fe_1, fe_2 = forward_eval[param1], forward_eval[param2]
        shape_1, shape_2 = fe_1.shape, fe_2.shape
        if len(shape_1) == 1 or len(shape_2) == 1 or shape_1 == () or shape_2 == ():
            if len(shape_1) == 1 and not len(shape_2) == 1:
                fe_1 = fe_1.reshape(-1, 1, 1)
            if len(shape_2) == 1 and not len(shape_1) == 1:
                fe_2 = fe_2.reshape(-1, 1, 1)
            return fe_1 * fe_2
        else:
            return np.matmul(fe_1, fe_2)

        # TODO reshape not necessary if scalars are in (1, 1) form


class _TransposeForwardEval(_ForwardEvalBase):
    @staticmethod
    def evaluate(param1, param2, param3, x, constants, forward_eval):
        if len(np.shape(forward_eval[param1])) == 3:
            return np.transpose(forward_eval[param1], (0, 2, 1))  # TODO not clean
        else:
            return np.transpose(forward_eval[param1])


def forward_eval_function(node, param1, param2, param3, x, constants, forward_eval):
    """Performs calculation of one line of stack"""
    return FORWARD_EVAL_MAP[node](param1, param2, param3, x, constants, forward_eval)


# Node maps
FORWARD_EVAL_MAP = {INTEGER: _IntegerForwardEval.evaluate,
                    VARIABLE: _LoadXForwardEval.evaluate,
                    CONSTANT: _LoadCForwardEval.evaluate,
                    ADDITION: _AddForwardEval.evaluate,
                    SUBTRACTION: _SubtractForwardEval.evaluate,
                    MULTIPLICATION: _MultiplyForwardEval.evaluate,
                    TRANSPOSE: _TransposeForwardEval.evaluate}
