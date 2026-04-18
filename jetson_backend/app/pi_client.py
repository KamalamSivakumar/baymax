import requests
from app.schemas import Mode


PI_BASE_URL = "http://127.0.0.1:9000" #set to the raspberry pi's endpoint/base


def push_mode_to_pi(mode: Mode, timeout: float = 2.0) -> dict:
    payload = {
        "mode": mode.value
    }

    try:
        response = requests.post(
            f"{PI_BASE_URL}/device/mode",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        return {
            "success": True,
            "pi_response": response.json(),
        }

    except requests.RequestException as error:
        return {
            "success": False,
            "error": str(error),
            "message": "Could not reach Raspberry Pi device service.",
        }


def push_action_to_pi(action: str, timeout: float = 2.0) -> dict:
    payload = {
        "action": action
    }

    try:
        response = requests.post(
            f"{PI_BASE_URL}/device/action",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        return {
            "success": True,
            "pi_response": response.json(),
        }

    except requests.RequestException as error:
        return {
            "success": False,
            "error": str(error),
            "message": "Could not send action to Raspberry Pi.",
        }


def get_pi_status(timeout: float = 2.0) -> dict:
    try:
        response = requests.get(
            f"{PI_BASE_URL}/device/status",
            timeout=timeout,
        )
        response.raise_for_status()

        return {
            "success": True,
            "pi_response": response.json(),
        }

    except requests.RequestException as error:
        return {
            "success": False,
            "error": str(error),
            "message": "Could not get Raspberry Pi status.",
        }