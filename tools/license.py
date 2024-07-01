from langchain_core.tools import tool

@tool
def license_plate_query(
    plate_state: str,
    plate_number: str,
) -> str:
    """
    Query the car information of plate number of `plate_state` state with plate number `plate_number`
    """

    if not isinstance(plate_state, str):
        return "the state where license plate registered is required for querying"

    if not isinstance(plate_number, str):
        return "license plate number is required for querying"

    return "This is a stolen car"
