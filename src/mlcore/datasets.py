from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from .client import MLCore, MLCoreAsync


class DatasetManager:
    """
    Manages dataset operations including uploading, listing, cleaning, and transforming.
    """

    def __init__(self, client: "MLCore"):
        self.client = client

    def list(self) -> list[dict[str, Any]]:
        """List all datasets available to the user."""
        return self.client.request("GET", "datasets")

    def get(self, dataset_id: str | UUID) -> dict[str, Any]:
        """Get metadata for a specific dataset."""
        return self.client.request("GET", f"dataset/{dataset_id}")

    def upload_file(self, file_path: str) -> dict[str, Any]:
        """
        Uploads a raw file to the server.
        Returns file metadata including the file_id needed for create().
        """
        with open(file_path, "rb") as f:
            files = {"file": f}
            return self.client.request("POST", "dataset/upload", files=files)

    def create(
        self,
        name: str,
        description: str,
        file_id: str | UUID,
        rows: int,
        columns: int,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Register an uploaded file as a dataset."""
        payload = {
            "name": name,
            "description": description,
            "file_id": str(file_id),
            "rows": rows,
            "columns": columns,
            "dataset_metadata": metadata,
        }
        return self.client.request("POST", "dataset", json=payload)

    def get_data(
        self,
        dataset_id: str | UUID,
        page: int = 1,
        limit: int = 50,
        as_df: bool = False,
    ) -> dict[str, Any] | Any:
        """
        Fetch paginated data from the dataset.
        If as_df is True, attempts to return a pandas DataFrame (requires pandas).
        """
        data = self.client.request(
            "GET", f"dataset/{dataset_id}/data", params={"page": page, "limit": limit}
        )

        if as_df:
            try:
                import pandas as pd

                return pd.DataFrame(data)
            except ImportError:
                return data
        return data

    def clean(
        self,
        dataset_id: str | UUID,
        strategy: str,
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Apply cleaning strategy (e.g., 'drop_nulls', 'fill_mean') to the dataset.
        Returns the new dataset version.
        """
        payload = {"strategy": strategy, "columns": columns}
        return self.client.request("POST", f"dataset/{dataset_id}/clean", json=payload)

    def transform(
        self, dataset_id: str | UUID, strategy: str, columns: list[str]
    ) -> dict[str, Any]:
        """
        Apply transformation strategy (e.g., 'standard_scaler', 'label_encoder') to columns.
        Returns the new dataset version.
        """
        payload = {"strategy": strategy, "columns": columns}
        return self.client.request("POST", f"dataset/{dataset_id}/transform", json=payload)

    def get_versions(self, dataset_id: str | UUID) -> list[dict[str, Any]]:
        """Get the full version history (lineage) for a dataset."""
        return self.client.request("GET", f"dataset/{dataset_id}/versions")

    def refresh(self, dataset_id: str | UUID) -> dict[str, Any]:
        """Recompute metadata from the physical file."""
        return self.client.request("POST", f"dataset/{dataset_id}/refresh")

    def delete(self, dataset_id: str | UUID) -> dict[str, Any]:
        """Delete a dataset."""
        return self.client.request("DELETE", f"dataset/{dataset_id}")

    def update(
        self,
        dataset_id: str | UUID,
        name: str,
        description: str,
        file_id: str | UUID,
        rows: int,
        columns: int,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Update dataset metadata."""
        payload = {
            "name": name,
            "description": description,
            "file_id": str(file_id),
            "rows": rows,
            "columns": columns,
            "dataset_metadata": metadata,
        }
        return self.client.request("PUT", f"dataset/{dataset_id}", json=payload)


class AsyncDatasetManager:
    """
    Asynchronous version of DatasetManager.
    """

    def __init__(self, client: "MLCoreAsync"):
        self.client = client

    async def list(self) -> list[dict[str, Any]]:
        """List all datasets available to the user."""
        return await self.client.request("GET", "datasets")

    async def get(self, dataset_id: str | UUID) -> dict[str, Any]:
        """Get metadata for a specific dataset."""
        return await self.client.request("GET", f"dataset/{dataset_id}")

    async def upload_file(self, file_path: str) -> dict[str, Any]:
        """
        Uploads a raw file to the server.
        """
        with open(file_path, "rb") as f:
            files = {"file": f}
            return await self.client.request("POST", "dataset/upload", files=files)

    async def create(
        self,
        name: str,
        description: str,
        file_id: str | UUID,
        rows: int,
        columns: int,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Register an uploaded file as a dataset."""
        payload = {
            "name": name,
            "description": description,
            "file_id": str(file_id),
            "rows": rows,
            "columns": columns,
            "dataset_metadata": metadata,
        }
        return await self.client.request("POST", "dataset", json=payload)

    async def get_data(
        self,
        dataset_id: str | UUID,
        page: int = 1,
        limit: int = 50,
        as_df: bool = False,
    ) -> dict[str, Any] | Any:
        """Fetch paginated data from the dataset."""
        data = await self.client.request(
            "GET", f"dataset/{dataset_id}/data", params={"page": page, "limit": limit}
        )

        if as_df:
            try:
                import pandas as pd

                return pd.DataFrame(data)
            except ImportError:
                return data
        return data

    async def clean(
        self,
        dataset_id: str | UUID,
        strategy: str,
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Apply cleaning strategy to the dataset."""
        payload = {"strategy": strategy, "columns": columns}
        return await self.client.request("POST", f"dataset/{dataset_id}/clean", json=payload)

    async def transform(
        self, dataset_id: str | UUID, strategy: str, columns: list[str]
    ) -> dict[str, Any]:
        """Apply transformation strategy to columns."""
        payload = {"strategy": strategy, "columns": columns}
        return await self.client.request("POST", f"dataset/{dataset_id}/transform", json=payload)

    async def get_versions(self, dataset_id: str | UUID) -> list[dict[str, Any]]:
        """Get the full version history for a dataset."""
        return await self.client.request("GET", f"dataset/{dataset_id}/versions")

    async def refresh(self, dataset_id: str | UUID) -> dict[str, Any]:
        """Recompute metadata from the physical file."""
        return await self.client.request("POST", f"dataset/{dataset_id}/refresh")

    async def delete(self, dataset_id: str | UUID) -> dict[str, Any]:
        """Delete a dataset."""
        return await self.client.request("DELETE", f"dataset/{dataset_id}")

    async def update(
        self,
        dataset_id: str | UUID,
        name: str,
        description: str,
        file_id: str | UUID,
        rows: int,
        columns: int,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Update dataset metadata."""
        payload = {
            "name": name,
            "description": description,
            "file_id": str(file_id),
            "rows": rows,
            "columns": columns,
            "dataset_metadata": metadata,
        }
        return await self.client.request("PUT", f"dataset/{dataset_id}", json=payload)
