class NumbaError(Exception):
    """Base error for mojo-numba compilation failures."""


class TypingError(NumbaError, TypeError):
    """Raised when an argument or operation is outside the typed subset."""


class UnsupportedError(NumbaError, NotImplementedError):
    """Raised when valid Python syntax is outside the compiled subset."""
