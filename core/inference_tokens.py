"""Small compatibility helpers for inference backend dispatch."""


class InferenceTypeToken(str):
    """String token that also exposes ``.value`` for enum-like callers."""

    @property
    def value(self) -> str:
        """Return the token value as a plain string."""
        return str(self)
