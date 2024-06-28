from langchain_core.tools import tool

@tool
def device_list(
    project_name: str,
    project_stage: str,
) -> str:
    """
    Query the device list for `project_name` project with `project_stage` stage
    """

    if not isinstance(project_name, str):
        return "project name is required for querying"

    if not isinstance(project_stage, str):
        return "project stage is required for querying"

    return "The query is successful"
