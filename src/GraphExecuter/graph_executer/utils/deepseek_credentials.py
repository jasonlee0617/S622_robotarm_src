import os

import keyring
from keyring.errors import PasswordDeleteError


SERVICE_NAME = "GraphExecuter"
ACCOUNT_NAME = "deepseek_api_key"
MISSING_MESSAGE = (
    "Set DeepSeek API key from Tools > DeepSeek API Settings "
    "or export DEEPSEEK_API_KEY"
)


class DeepSeekCredentialError(RuntimeError):
    pass


def _keyring_password():
    try:
        return (keyring.get_password(SERVICE_NAME, ACCOUNT_NAME) or "").strip()
    except Exception as exc:
        raise DeepSeekCredentialError(f"System keyring is unavailable: {exc}") from exc


def credential_source():
    if os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return "environment"
    return "keyring" if _keyring_password() else "missing"


def get_deepseek_api_key():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip() or _keyring_password()
    if not api_key:
        raise DeepSeekCredentialError(MISSING_MESSAGE)
    return api_key


def save_deepseek_api_key(api_key):
    api_key = str(api_key).strip()
    if not api_key:
        raise ValueError("DeepSeek API key cannot be empty")
    try:
        keyring.set_password(SERVICE_NAME, ACCOUNT_NAME, api_key)
    except Exception as exc:
        raise DeepSeekCredentialError(f"Cannot save to system keyring: {exc}") from exc


def delete_deepseek_api_key():
    try:
        keyring.delete_password(SERVICE_NAME, ACCOUNT_NAME)
        return True
    except PasswordDeleteError:
        return False
    except Exception as exc:
        raise DeepSeekCredentialError(f"Cannot delete from system keyring: {exc}") from exc
