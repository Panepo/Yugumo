from langchain_core.tools import tool

@tool
def driver_query(
    project_name: str,
    query_condition: str,
) -> str:
    """
    Query the driver information of the `project_name` project with specific `query_condition`
    """

    return "Confirm querying for `project_name` project with `query_condition`?"
