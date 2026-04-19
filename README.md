# MLCore Python Client

A powerful Python client for [MLCore](https://github.com/Amanbig/MLCore), providing a database-like interface for managing datasets and machine learning models. Built for data scientists and engineers who want to automate their machine learning workflows.

## Links
- **GitHub Repository**: [Amanbig/MLCore](https://github.com/Amanbig/MLCore)
- **Docker Hub**: [procoder588/mlcore](https://hub.docker.com/r/procoder588/mlcore)

## Installation

```bash
pip install mlcore-client
```

## Connection

Connect to your MLCore instance using a connection string (URI) or explicit parameters.

```python
from mlcore import MLCore

# Option 1: Connection string (Recommended)
# Format: mlcore://user:password@host:port
client = MLCore("mlcore://admin:password@localhost:8000")

# Option 2: Explicit parameters
client = MLCore(
    host="localhost", 
    port=8000, 
    email="admin@example.com", 
    password="password"
)
```

### Asynchronous Support
For `asyncio` applications, use `MLCoreAsync`:

```python
from mlcore import MLCoreAsync

client = MLCoreAsync("mlcore://admin:password@localhost:8000")
await client.connect()
# ... use await with all methods ...
await client.close()
```

---

## API Documentation

### 📊 Dataset Manager (`client.datasets`)

Manage data lifecycle from raw files to cleaned versions.

| Method | Description | Parameters |
|:-------|:------------|:-----------|
| `list()` | List all accessible datasets. | - |
| `get(id)` | Get metadata for a dataset. | `id`: UUID or string |
| `upload_file(path)` | Upload a raw CSV/Excel file. | `path`: Local file path |
| `create(...)` | Register a file as a dataset. | `name`, `description`, `file_id`, `rows`, `columns`, `metadata` |
| `get_data(id, ...)` | Fetch paginated rows. | `id`, `page=1`, `limit=50`, `as_df=False` |
| `clean(id, ...)` | Apply cleaning logic. | `id`, `strategy` ('drop_nulls', 'fill_mean', etc.), `columns` |
| `transform(id, ...)` | Apply transformations. | `id`, `strategy` ('standard_scaler', etc.), `columns` |
| `get_versions(id)` | Get history/lineage. | `id` |
| `delete(id)` | Permanently remove dataset. | `id` |

### 🤖 Model Manager (`client.models`)

Train, evaluate, and deploy machine learning models.

| Method | Description | Parameters |
|:-------|:------------|:-----------|
| `list()` | List all trained models. | - |
| `get(id)` | Get model specs & metrics. | `id`: UUID or string |
| `train(...)` | Start a training job. | `dataset_id`, `algorithm`, `target_column`, `features`, `hyperparameters`, `name` |
| `predict(id, inputs)` | Run real-time inference. | `id`, `inputs`: Dict[feature_name, value] |
| `download(id, path)` | Download `.joblib` file. | `id`, `path`: Local destination path |
| `retrain(id, ...)` | Run new training session. | `id`, `dataset_id`, `algorithm`, etc. |
| `get_hyperparameters(algo)` | Get valid hyperparams. | `algo`: Algorithm name |
| `get_versions(id)` | Get model history. | `id` |
| `update_meta(id, ...)` | Rename/update description. | `id`, `name`, `description` |
| `delete(id)` | Delete model artifacts. | `id` |

### 📈 General Methods

| Method | Description |
|:-------|:------------|
| `get_stats()` | Get platform-wide statistics (counts of models, datasets, files). |
| `health_check()` | Check server connectivity and version. |

---

## Usage Examples

### End-to-End Pipeline
```python
# 1. Prepare Data
file_info = client.datasets.upload_file("raw_data.csv")
ds = client.datasets.create(name="Training Set", file_id=file_info["id"], ...)
cleaned_ds = client.datasets.clean(ds["id"], strategy="drop_nulls")

# 2. Train Model
model = client.models.train(
    dataset_id=cleaned_ds["id"],
    algorithm="random_forest",
    target_column="label",
    hyperparameters={"n_estimators": 200}
)

# 3. Monitor Specs
print(f"Model trained with {model['accuracy']}% accuracy")

# 4. Predict
result = client.models.predict(model["id"], inputs={"feature_1": 0.5, "feature_2": 1.2})
print(f"Prediction: {result['predictions']}")
```

## Features
- **Database-like Connection**: URI-based connection strings (`mlcore://`).
- **Session Persistence**: Automatic token management and re-auth logic.
- **Pandas Ready**: One-click conversion from server data to DataFrames.
- **Async First**: First-class support for `httpx`-based asynchronous I/O.
- **Developer Friendly**: Strictly typed and linted with **Ruff**.

## License
MIT