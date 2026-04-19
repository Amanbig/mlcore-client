from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from .client import MLCore, MLCoreAsync


class ModelManager:
    """
    Manages machine learning model operations including training, prediction,
    uploading, and version control.
    """

    def __init__(self, client: MLCore):
        self.client = client

    def list(self) -> builtins.list[dict[str, Any]]:
        """List all models available to the user."""
        return self.client.request("GET", "ml_models")

    def get(self, model_id: str | UUID) -> dict[str, Any]:
        """Get metadata for a specific model."""
        return self.client.request("GET", f"ml_model/{model_id}")

    def train(
        self,
        dataset_id: str | UUID,
        algorithm: str,
        target_column: str,
        features: builtins.list[str] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """
        Train a new model on the server.
        """
        payload = {
            "dataset_id": str(dataset_id),
            "model_algorithm": algorithm,
            "target_column": target_column,
            "features": features,
            "hyperparameters": hyperparameters or {},
            "name": name,
            "description": description,
        }
        return self.client.request("POST", "ml_model/train", json=payload)

    def predict(self, model_id: str | UUID, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Run inference on a trained model.
        Inputs should be a dictionary mapping feature names to values.
        """
        payload = {"inputs": inputs}
        return self.client.request("POST", f"ml_model/{model_id}/predict", json=payload)

    def create(
        self,
        file_path: str,
        name: str,
        version: str,
        description: str,
        model_type: str,
        inputs: str,
        outputs: str,
        accuracy: float,
        error: float,
    ) -> dict[str, Any]:
        """
        Upload an existing model file and register it as a model.
        """
        params = {
            "name": name,
            "version": version,
            "description": description,
            "model_type": model_type,
            "inputs": inputs,
            "outputs": outputs,
            "accuracy": accuracy,
            "error": error,
        }
        with open(file_path, "rb") as f:
            files = {"file": f}
            return self.client.request("POST", "ml_model", params=params, files=files)

    def retrain(
        self,
        model_id: str | UUID,
        dataset_id: str | UUID,
        algorithm: str,
        target_column: str,
        features: builtins.list[str] | None = None,
        hyperparameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrain an existing model with new parameters or data."""
        payload = {
            "dataset_id": str(dataset_id),
            "model_algorithm": algorithm,
            "target_column": target_column,
            "features": features,
            "hyperparameters": hyperparameters or {},
        }
        return self.client.request("POST", f"ml_model/{model_id}/retrain", json=payload)

    def get_hyperparameters(self, algorithm: str) -> dict[str, Any]:
        """Get the allowed hyperparameter schema for a specific algorithm."""
        return self.client.request("GET", f"ml_model/hyperparameters/{algorithm}")

    def get_versions(self, model_id: str | UUID) -> builtins.list[dict[str, Any]]:
        """Get the version history for a specific model lineage."""
        return self.client.request("GET", f"ml_model/{model_id}/versions")

    def download(self, model_id: str | UUID, output_path: str):
        """Download the trained model file (.joblib) to the specified path."""
        url = f"{self.client.base_url}/ml_model/{model_id}/download"
        response = self.client._session.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def delete(self, model_id: str | UUID) -> dict[str, Any]:
        """Delete a model and its associated files."""
        return self.client.request("DELETE", f"ml_model/{model_id}")

    def update_meta(
        self,
        model_id: str | UUID,
        name: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Update a model's name and description."""
        payload = {"name": name, "description": description}
        return self.client.request("PATCH", f"ml_model/{model_id}", json=payload)


class AsyncModelManager:
    """
    Asynchronous version of ModelManager.
    """

    def __init__(self, client: MLCoreAsync):
        self.client = client

    async def list(self) -> builtins.list[dict[str, Any]]:
        """List all models available to the user."""
        return await self.client.request("GET", "ml_models")

    async def get(self, model_id: str | UUID) -> dict[str, Any]:
        """Get metadata for a specific model."""
        return await self.client.request("GET", f"ml_model/{model_id}")

    async def train(
        self,
        dataset_id: str | UUID,
        algorithm: str,
        target_column: str,
        features: builtins.list[str] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Train a new model on the server asynchronously."""
        payload = {
            "dataset_id": str(dataset_id),
            "model_algorithm": algorithm,
            "target_column": target_column,
            "features": features,
            "hyperparameters": hyperparameters or {},
            "name": name,
            "description": description,
        }
        return await self.client.request("POST", "ml_model/train", json=payload)

    async def predict(self, model_id: str | UUID, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run inference on a trained model asynchronously."""
        payload = {"inputs": inputs}
        return await self.client.request("POST", f"ml_model/{model_id}/predict", json=payload)

    async def create(
        self,
        file_path: str,
        name: str,
        version: str,
        description: str,
        model_type: str,
        inputs: str,
        outputs: str,
        accuracy: float,
        error: float,
    ) -> dict[str, Any]:
        """Upload and register a model asynchronously."""
        params = {
            "name": name,
            "version": version,
            "description": description,
            "model_type": model_type,
            "inputs": inputs,
            "outputs": outputs,
            "accuracy": accuracy,
            "error": error,
        }
        with open(file_path, "rb") as f:
            files = {"file": f}
            return await self.client.request("POST", "ml_model", params=params, files=files)

    async def retrain(
        self,
        model_id: str | UUID,
        dataset_id: str | UUID,
        algorithm: str,
        target_column: str,
        features: builtins.list[str] | None = None,
        hyperparameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrain an existing model asynchronously."""
        payload = {
            "dataset_id": str(dataset_id),
            "model_algorithm": algorithm,
            "target_column": target_column,
            "features": features,
            "hyperparameters": hyperparameters or {},
        }
        return await self.client.request("POST", f"ml_model/{model_id}/retrain", json=payload)

    async def get_hyperparameters(self, algorithm: str) -> dict[str, Any]:
        """Get hyperparameter schema asynchronously."""
        return await self.client.request("GET", f"ml_model/hyperparameters/{algorithm}")

    async def get_versions(self, model_id: str | UUID) -> builtins.list[dict[str, Any]]:
        """Get version history asynchronously."""
        return await self.client.request("GET", f"ml_model/{model_id}/versions")

    async def download(self, model_id: str | UUID, output_path: str):
        """Download the trained model file asynchronously."""
        url = f"{self.client.base_url}/ml_model/{model_id}/download"
        async with self.client._client.stream("GET", url) as response:
            response.raise_for_status()
            with open(output_path, "wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)

    async def delete(self, model_id: str | UUID) -> dict[str, Any]:
        """Delete a model asynchronously."""
        return await self.client.request("DELETE", f"ml_model/{model_id}")

    async def update_meta(
        self,
        model_id: str | UUID,
        name: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Update a model's metadata asynchronously."""
        payload = {"name": name, "description": description}
        return await self.client.request("PATCH", f"ml_model/{model_id}", json=payload)
