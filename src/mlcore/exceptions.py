class MLCoreError(Exception):
    """Base exception for all MLCore client errors."""

    pass


class MLCoreConnectionError(MLCoreError):
    """Raised when the client cannot connect to the MLCore server."""

    pass


class MLCoreAuthenticationError(MLCoreError):
    """Raised when authentication fails (invalid credentials or token)."""

    pass


class MLCoreValidationError(MLCoreError):
    """Raised when input validation fails locally or on the server."""

    pass


class MLCoreResourceNotFoundError(MLCoreError):
    """Raised when a requested resource (dataset, model, file) is not found."""

    pass


class MLCoreApiError(MLCoreError):
    """Raised when the server returns an error response."""

    def __init__(self, message: str, status_code: int = None, detail: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail
