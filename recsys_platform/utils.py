import secrets
from datetime import UTC, datetime


_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"


def generate_model_id(model_type: str) -> str:
    """Generate a readable, time-sortable model identifier."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = "".join(secrets.choice(_ALPHABET) for _ in range(6))

    return f"{model_type}-{timestamp}-{suffix}"
