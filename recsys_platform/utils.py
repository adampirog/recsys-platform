import secrets
import string
from datetime import UTC, datetime


_ALPHABET = string.digits + string.ascii_letters


def generate_model_id(model_family: str) -> str:
    """Generate a readable, time-sortable model identifier."""

    timestamp = datetime.now(UTC).strftime("%y%m%dT%H%M")
    suffix = "".join(secrets.choice(_ALPHABET) for _ in range(5))

    return f"{model_family}-{timestamp}-{suffix}"
