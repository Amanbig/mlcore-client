from typing import Any
from urllib.parse import urlparse

import requests

from .datasets import DatasetManager
from .exceptions import MLCoreAuthenticationError, MLCoreConnectionError
from .models import ModelManager


class MLCore:
    """
    The main client for MLCore. Provides a database-like connection interface.

    Usage:
        client = MLCore("mlcore://admin:password@localhost:8000")
        # or
        client = MLCore(host="localhost", port=8000, email="admin", password="password")
    """

    def __init__(
        self,
        connection_url: str | None = None,
        host: str | None = None,
        port: int | None = None,
        email: str | None = None,
        password: str | None = None,
        use_https: bool = False,
    ):
        self.base_url = ""
        self.token = None
        self._session = requests.Session()

        if connection_url:
            self._parse_url(connection_url)
        else:
            protocol = "https" if use_https else "http"
            self.host = host or "localhost"
            self.port = port or 8000
            self.base_url = f"{protocol}://{self.host}:{self.port}/api"
            self.email = email
            self.password = password

        # Initialize Managers
        self.models = ModelManager(self)
        self.datasets = DatasetManager(self)

        # Authenticate if credentials provided
        if self.email and self.password:
            self.connect()

    def _parse_url(self, url: str):
        if not url.startswith("mlcore://"):
            raise MLCoreConnectionError("URL must start with 'mlcore://'")

        # Replace protocol for standard parsing
        parsed = urlparse(url.replace("mlcore://", "http://", 1))

        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or 8000
        self.email = parsed.username
        self.password = parsed.password

        protocol = "https" if parsed.scheme == "https" else "http"
        self.base_url = f"{protocol}://{self.host}:{self.port}/api"

    def connect(self):
        """Authenticates with the MLCore server and establishes a session."""
        login_url = f"{self.base_url}/auth/login"
        payload = {"email": self.email, "password": self.password}

        try:
            response = self._session.post(login_url, json=payload)
            if response.status_code == 401:
                raise MLCoreAuthenticationError("Invalid credentials")
            response.raise_for_status()

            data = response.json()
            self.token = data.get("token")
            # Update session headers for subsequent requests
            self._session.headers.update({"Authorization": f"Bearer {self.token}"})

        except requests.RequestException as e:
            raise MLCoreConnectionError(f"Failed to connect to MLCore server: {e}") from e

    def request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Internal helper for making authenticated requests."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = self._session.request(method, url, **kwargs)
            if response.status_code == 401:
                # Try to reconnect once if token expired
                if self.email and self.password:
                    self.connect()
                    response = self._session.request(method, url, **kwargs)
                else:
                    raise MLCoreAuthenticationError("Session expired or unauthorized")

            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if hasattr(e.response, "json"):
                detail = e.response.json().get("detail", str(e))
                raise Exception(f"MLCore Error: {detail}") from e
            raise e

    def get_stats(self) -> dict[str, Any]:
        """Returns platform-wide statistics for the authenticated user."""
        return self.request("GET", "stats")

    def health_check(self) -> dict[str, Any]:
        """Checks the health of the MLCore server."""
        return self.request("GET", "health")

    def __repr__(self):
        auth_status = self.token is not None
        return f"<MLCore Client(host='{self.host}', port={self.port}, authenticated={auth_status})>"
