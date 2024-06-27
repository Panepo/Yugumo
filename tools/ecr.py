from langchain_core.tools import tool
import requests


@tool
def ecr(
    ecr_number: str,
) -> str:
    """
    Get the current ecr status for `ecr_number`
    """

    if not isinstance(ecr_number, str):
        raise TypeError("ecr number must be a string")

    try:
        resp = requests.get(
            f"http://10.68.90.166/get_ecr_status/default?ecr_no={ecr_number}"
        )
        resp.raise_for_status()
        resp = resp.json()

        if len(resp) > 0:
            if resp[len(resp) - 1]["STATUS"] == "K":
                return str("the ecr status of {ecr_number} is pass")
            else:
                return str("the ecr status of {ecr_number} is on progress")
        else:
            return str("the ecr number {ecr_number} is wrong")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return str("the ecr number {ecr_number} is wrong")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Error connecting: {conn_err}")
        return str("please check the network status")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error: {timeout_err}")
        return str("please check the network status")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred: {req_err}")
        return str("please check the network status")
