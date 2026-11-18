import secrets
import string
from datetime import UTC, datetime


_ALPHABET = string.digits + string.ascii_letters


def generate_model_id(model_family: str) -> str:
    """Generate a model ID with a UTC minute timestamp and a random suffix."""

    timestamp = datetime.now(UTC).strftime("%y%m%dT%H%M")
    suffix = "".join(secrets.choice(_ALPHABET) for _ in range(4))

    return f"{model_family}-{timestamp}-{suffix}"
