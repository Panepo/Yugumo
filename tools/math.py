from langchain_core.tools import tool


@tool
def multiply(first_int: int, second_int: int) -> int | str:
    """
    Multiply two integers together.
    """

    if not isinstance(first_int, int):
        return "The first integer must be a number"

    if not isinstance(second_int, int):
        return "The second integer must be a number"

    return first_int * second_int


@tool
def add(first_int: int | str, second_int: int | str) -> int | str:
    """
    Add two integers.
    """

    if not isinstance(first_int, int):
        print("FQ")
        return "The first integer must be a number"

    if not isinstance(second_int, int):
        print("FQQ")
        return "The second integer must be a number"

    print(first_int + second_int)
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int | str:
    """
    Exponentiate the base to the exponent power.
    """

    if not isinstance(base, int):
        return "The base must be a number"

    if not isinstance(exponent, int):
        return "The exponent must be a number"

    return base**exponent
