from langchain_core.tools import tool


@tool
def multiply(first_int: int, second_int: int) -> int:
    """
    Multiply two integers together.
    """

    if not isinstance(first_int, int):
        raise TypeError("The first integer must be a number")

    if not isinstance(second_int, int):
        raise TypeError("The second integer must be a number")

    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    """
    Add two integers.
    """

    if not isinstance(first_int, int):
        raise TypeError("The first integer must be a number")

    if not isinstance(second_int, int):
        raise TypeError("The second integer must be a number")

    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    """
    Exponentiate the base to the exponent power.
    """

    if not isinstance(base, int):
        raise TypeError("The base must be a number")

    if not isinstance(exponent, int):
        raise TypeError("The exponent must be a number")

    return base**exponent
