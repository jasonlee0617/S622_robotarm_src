"""Shared DeepSeek credentials for the terminal and Qt clients."""

from __future__ import annotations

import os

import keyring
from keyring.errors import PasswordDeleteError


SERVICE_NAME = "GraphExecuter"
ACCOUNT_NAME = "deepseek_api_key"
MISSING_MESSAGE = (
    "Set DeepSeek API key with `key set` in llm_yolo_cli, "
    "Tools > DeepSeek API Settings, or DEEPSEEK_API_KEY"
)


class DeepSeekCredentialError(RuntimeError):
    pass


def _keyring_password() -> str:
    try:
        return (keyring.get_password(SERVICE_NAME, ACCOUNT_NAME) or "").strip()
    except Exception as exc:
        raise DeepSeekCredentialError(f"System keyring is unavailable: {exc}") from exc


def credential_status() -> str:
    if os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return "environment"
    return "keyring" if _keyring_password() else "missing"


credential_source = credential_status


def get_deepseek_api_key() -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if api_key:
        return api_key
    api_key = _keyring_password()
    if not api_key:
        raise DeepSeekCredentialError(MISSING_MESSAGE)
    return api_key


def set_deepseek_api_key(api_key: str) -> None:
    api_key = str(api_key).strip()
    if not api_key:
        raise ValueError("DeepSeek API key cannot be empty")
    try:
        keyring.set_password(SERVICE_NAME, ACCOUNT_NAME, api_key)
    except Exception as exc:
        raise DeepSeekCredentialError(f"Cannot save to system keyring: {exc}") from exc


save_deepseek_api_key = set_deepseek_api_key


def delete_deepseek_api_key() -> bool:
    try:
        keyring.delete_password(SERVICE_NAME, ACCOUNT_NAME)
        return True
    except PasswordDeleteError:
        return False
    except Exception as exc:
        raise DeepSeekCredentialError(f"Cannot delete from system keyring: {exc}") from exc
