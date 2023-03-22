from bingo.symbolic_regression.agraphMD.operator_definitions import *


# NOTE (David Randall): it is assumed node, param1, param2, param3, are all integers
class _ValidateBase:
    @staticmethod
    def validate_op(param1, param2, param3, dimensions, x_dims, constant_dims):
        raise NotImplementedError


class _IntegerValidate(_ValidateBase):
    @staticmethod
    def validate_op(param1, param2, param3, dimensions, x_dims, constant_dims):
        return param2 >= 0 and param3 >= 0, (param2, param3)


class _LoadVariableValidate(_ValidateBase):
    @staticmethod
    def validate_op(param1, param2, param3, dimensions, x_dims, constant_dims):
        dims_from_command = (param2, param3)
        if param1 < 0:
            return False, dims_from_command
        try:
            return param2 >= 0 and param3 >= 0 and \
                   x_dims[param1] == dims_from_command, dims_from_command
        except IndexError:
            return False, dims_from_command


class _LoadConstantValidate(_ValidateBase):
    @staticmethod
    def validate_op(param1, param2, param3, dimensions, x_dims, constant_dims):
        dims_from_command = (param2, param3)
        if param1 < -1:
            return False, dims_from_command
        try:
            if dims_from_command != (0, 0):
                return param2 >= 0 and param3 >= 0 and \
                       constant_dims[param1] == dims_from_command, dims_from_command
            else:
                return param2 >= 0 and param3 >= 0 and \
                       constant_dims[param1] == (), dims_from_command
        except IndexError:
            return False, dims_from_command


class _AdditionLikeValidate(_ValidateBase):
    @staticmethod
    def validate_op(param1, param2, param3, dimensions, x_dims, constant_dims):
        return tuple(dimensions[param1]) == tuple(dimensions[param2]), tuple(dimensions[param1])


class _MultiplyValidate(_ValidateBase):
    @staticmethod
    def validate_op(param1, param2, param3, dimensions, x_dims, constant_dims):
        dim_1, dim_2 = tuple(dimensions[param1]), tuple(dimensions[param2])
        # scalar * scalar
        if dim_1 == (0, 0) and dim_2 == (0, 0):
            return True, (0, 0)
        # scalar * matrix
        elif dim_1 == (0, 0) and dim_2[0] > 0 and dim_2[1] > 0:
            return True, dim_2
        # matrix * scalar
        elif dim_2 == (0, 0) and dim_1[0] > 0 and dim_1[1] > 0:
            return True, dim_1
        # matrix * matrix
        elif dim_1[1] == dim_2[0]:
            return True, (dim_1[0], dim_2[1])
        else:
            return False, (0, 0)


class _TransposeValidate(_ValidateBase):
    @staticmethod
    def validate_op(param1, param2, param3, dimensions, x_dims, constant_dims):
        return param1 >= 0, (dimensions[param1][1], dimensions[param1][0])


class _ArityOneOperatorWithNoShapeChange(_ValidateBase):
    @staticmethod
    def validate_op(param1, param2, param3, dimensions, x_dims, constant_dims):
        return param1 >= 0, tuple(dimensions[param1])


class _CrossValidate(_ValidateBase):
    @staticmethod
    def validate_op(param1, param2, param3, dimensions, x_dims, constant_dims):
        if param1 < 0 or param2 < 0:
            return False, (0, 0)

        dim_1 = dimensions[param1]
        dim_2 = dimensions[param2]
        valid = dim_1 == dim_2 and 3 in dim_1

        return valid, dim_1


class _NormalizeValidate(_ValidateBase):
    @staticmethod
    def validate_op(param1, param2, param3, dimensions, x_dims, constant_dims):
        if param1 < 0:
            return False, (0, 0)
        dim_1 = dimensions[param1]
        return 3 in dim_1, dim_1

def validate_operator(node, param1, param2, param3, dimensions, x_dims, constant_dims):
    return VALIDATE_MAP[node](param1, param2, param3, dimensions, x_dims, constant_dims)


VALIDATE_MAP = {INTEGER: _IntegerValidate.validate_op,
                VARIABLE: _LoadVariableValidate.validate_op,
                CONSTANT: _LoadConstantValidate.validate_op,
                ADDITION: _AdditionLikeValidate.validate_op,
                SUBTRACTION: _AdditionLikeValidate.validate_op,
                MULTIPLICATION: _MultiplyValidate.validate_op,
                SIN: _ArityOneOperatorWithNoShapeChange.validate_op,
                COS: _ArityOneOperatorWithNoShapeChange.validate_op,
                EXPONENTIAL: _ArityOneOperatorWithNoShapeChange.validate_op,
                LOGARITHM: _ArityOneOperatorWithNoShapeChange.validate_op,
                ABS: _ArityOneOperatorWithNoShapeChange.validate_op,
                SQRT: _ArityOneOperatorWithNoShapeChange.validate_op,
                SINH: _ArityOneOperatorWithNoShapeChange.validate_op,
                COSH: _ArityOneOperatorWithNoShapeChange.validate_op,
                TRANSPOSE: _TransposeValidate.validate_op,
                ARCTAN: _ArityOneOperatorWithNoShapeChange.validate_op,
                ARCCOS: _ArityOneOperatorWithNoShapeChange.validate_op,
                CROSS: _CrossValidate.validate_op,
                NORMALIZE: _NormalizeValidate.validate_op}
# TODO dot product, another operator???
