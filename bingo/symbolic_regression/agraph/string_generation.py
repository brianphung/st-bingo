"""
Attributes
----------
STACK_PRINT_MAP : dict {int: str}
                  A map of node number to a format string for stack output
LATEX_PRINT_MAP : dict {int: str}
                  A map of node number to a format string for latex output
CONSOLE_PRINT_MAP : dict {int: str}
                  A map of node number to a format string for console output
"""
STACK_PRINT_MAP = {2: "({}) + ({})",
                   3: "({}) - ({})",
                   4: "({}) * ({})",
                   5: "({}) / ({}) ",
                   6: "sin ({})",
                   7: "cos ({})",
                   8: "exp ({})",
                   9: "log ({})",
                   10: "({}) ^ ({})",
                   11: "abs ({})",
                   12: "sqrt ({})"}
LATEX_PRINT_MAP = {2: "{} + {}",
                   3: "{} - ({})",
                   4: "({})({})",
                   5: "\\frac{{ {} }}{{ {} }}",
                   6: "sin{{ {} }}",
                   7: "cos{{ {} }}",
                   8: "exp{{ {} }}",
                   9: "log{{ {} }}",
                   10: "({})^{{ ({}) }}",
                   11: "|{}|",
                   12: "\\sqrt{{ {} }}"}
CONSOLE_PRINT_MAP = {2: "{} + {}",
                     3: "{} - ({})",
                     4: "({})({})",
                     5: "({})/({}) ",
                     6: "sin({})",
                     7: "cos({})",
                     8: "exp({})",
                     9: "log({})",
                     10: "({})^({})",
                     11: "|{}|",
                     12: "sqrt({})"}


def get_formatted_string(eq_format, command_array, constants):
    """ Builds a formatted string from command array and constants

    Parameters
    ----------
    eq_format : str
        "stack", "latex", or "console"
    command_array : Nx3 array of int
        stack representation of an equation
    constants : list(float)
        list of numerical constants in the equation

    Returns
    -------
    str
        equation formatted in the way specified
    """
    if eq_format == "stack":
        return _get_stack_string(command_array, constants)

    if eq_format == "latex":
        format_dict = LATEX_PRINT_MAP
    else:  # "console"
        format_dict = CONSOLE_PRINT_MAP
    str_list = []
    for stack_element in command_array:
        tmp_str = _get_formatted_element_string(stack_element, str_list,
                                                format_dict, constants)
        str_list.append(tmp_str)
    return str_list[-1]


def _get_stack_string(command_array, constants):
    tmp_str = ""
    for i, stack_element in enumerate(command_array):
        tmp_str += _get_stack_element_string(i, stack_element, constants)

    return tmp_str


def _get_stack_element_string(command_index, stack_element, constants):
    node, param1, param2 = stack_element
    tmp_str = "(%d) <= " % command_index
    if node == 0:
        tmp_str += "X_%d" % param1
    elif node == 1:
        if param1 == -1 or param1 >= len(constants):
            tmp_str += "C"
        else:
            tmp_str += "C_{} = {}".format(param1, constants[param1])
    elif node == -1:
        tmp_str += "{} (integer)".format(param1)
    else:
        tmp_str += STACK_PRINT_MAP[node].format(param1, param2)
    tmp_str += "\n"
    return tmp_str


def _get_formatted_element_string(stack_element, str_list,
                                  format_dict, constants):
    node, param1, param2 = stack_element
    if node == 0:
        tmp_str = "X_%d" % param1
    elif node == 1:
        if param1 == -1 or param1 >= len(constants):
            tmp_str = "?"
        else:
            tmp_str = str(constants[param1])
    elif node == -1:
        tmp_str = str(int(param1))
    else:
        tmp_str = format_dict[node].format(str_list[param1], str_list[param2])
    return tmp_str
