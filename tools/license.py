from langchain_core.tools import tool

@tool
def license_plate_query(
    plate_state: str,
    plate_number: str,
) -> str:
    """
    Query the car information of plate number of `plate_state` state with plate number `plate_number`
    """

    return "Confirm querying for `plate_state` state plate `plate_number`?"

@tool
def license_plate_query_confirm(
    plate_state: str,
    plate_number: str,
) -> str:
    """
    User confirm the query the car information of plate number of `plate_state` state with plate number `plate_number`
    """

    return "This is a stolen car."
