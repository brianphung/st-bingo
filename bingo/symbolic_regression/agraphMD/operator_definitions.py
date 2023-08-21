"""
Attributes
----------
IS_ARITY_2_MAP : dict {int: bool}
                 A map of node number to boolean that states whether the
                 node has arity 2 (as opposed to 1)
IS_TERMINAL_MAP : dict {int: bool}
                 A map of node number to boolean that states whether the
                 node is a terminal
OPERATOR_NAMES : dict{int: list(string)}
                 A map of node number to common names for the node
"""
INTEGER = -1
VARIABLE = 0
CONSTANT = 1
ADDITION = 2
SUBTRACTION = 3
MULTIPLICATION = 4
DIVISION = 5
SIN = 6
COS = 7
EXPONENTIAL = 8
LOGARITHM = 9
POWER = 10
ABS = 11
SQRT = 12
SAFE_POWER = 13
SINH = 14
COSH = 15
TRANSPOSE = 16
ARCTAN = 17
ARCCOS = 18
CROSS = 20
NORMALIZE = 21

# (David Randall) Note: Not for use in stacks, purely for string formatting
#   see normal mult where matrix multiplication can be performed
MATRIX_MULT = 99

IS_ARITY_2_MAP = {INTEGER: False,
                  VARIABLE: False,
                  CONSTANT: False,
                  ADDITION: True,
                  SUBTRACTION: True,
                  MULTIPLICATION: True,
                  DIVISION: True,
                  SIN: False,
                  COS: False,
                  SINH: False,
                  COSH: False,
                  EXPONENTIAL: False,
                  LOGARITHM: False,
                  POWER: True,
                  ABS: False,
                  SQRT: False,
                  SAFE_POWER: True,
                  TRANSPOSE: False,
                  ARCTAN: False,
                  ARCCOS: False,
                  CROSS: True,
                  NORMALIZE: False,
                  MATRIX_MULT: True}
IS_TERMINAL_MAP = {INTEGER: True,
                   VARIABLE: True,
                   CONSTANT: True,
                   ADDITION: False,
                   SUBTRACTION: False,
                   MULTIPLICATION: False,
                   DIVISION: False,
                   SIN: False,
                   COS: False,
                   SINH: False,
                   COSH: False,
                   EXPONENTIAL: False,
                   LOGARITHM: False,
                   POWER: False,
                   ABS: False,
                   SQRT: False,
                   SAFE_POWER: False,
                   TRANSPOSE: False,
                   ARCTAN: False,
                   ARCCOS: False,
                   CROSS: False,
                   NORMALIZE: False,
                   MATRIX_MULT: False}
OPERATOR_NAMES = {INTEGER: ["integer"],
                  VARIABLE: ["load", "x"],
                  CONSTANT: ["constant", "c"],
                  ADDITION: ["add", "addition", "+"],
                  SUBTRACTION: ["subtract", "subtraction", "-"],
                  MULTIPLICATION: ["multiply", "multiplication", "*"],
                  DIVISION: ["divide", "division", "/"],
                  SIN: ["sine", "sin"],
                  COS: ["cosine", "cos"],
                  EXPONENTIAL: ["exponential", "exp", "e"],
                  LOGARITHM: ["logarithm", "log"],
                  # POWER: ["power", "pow", "^"],
                  ABS: ["absolute value", "||", "|"],
                  SQRT: ["square root", "sqrt"],
                  # SAFE_POWER: ["safe power", "safe pow"],
                  SINH: ["sineh", "sinh"],
                  COSH: ["cosineh", "cosh"],
                  TRANSPOSE: ["transpose", "T"],
                  ARCTAN: ["arctan", "atan", "invtan", "inverse tangent",
                           "inverse tan"],
                  ARCCOS: ["arccos", "acos", "invcos", "inverse cosgent",
                           "inverse cos"],
                  CROSS: ["cross", "cross product"],
                  NORMALIZE: ["normalize"],
                  MATRIX_MULT: ["matrix multiplication", "@"]}
