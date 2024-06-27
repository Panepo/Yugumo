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

    resp = requests.get(f"http://10.68.90.166/get_ecr_status/default?ecr_no={ecr_number}")
    resp.raise_for_status()
    resp = resp.json()

    if (len(resp) > 0):
      if (resp[len(resp) - 1]['STATUS'] == 'K'):
          return str("pass")
      else:
          return str("on progress")
    else:
      return str("ecr wrong number")
