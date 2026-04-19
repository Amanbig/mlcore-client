# MLCore Python Client

A Python client for [MLCore](https://github.com/aman-preeti/MLCore), providing a database-like interface for managing datasets and machine learning models.

## Installation

```bash
pip install mlcore-client
```

## Quick Start

### Connecting to MLCore

You can connect using a connection string (similar to database URLs) or by providing host and credentials explicitly.

```python
from mlcore.client import MLCore

# Using a connection string
client = MLCore("mlcore://admin:password@localhost:8000")

# Or using explicit parameters
client = MLCore(
    host="localhost",
    port=8000,
    email="admin@example.com",
    password="secure_password"
)

# Check connection
print(client.health_check())
```

### Working with Datasets

The client allows you to manage the entire dataset lifecycle, from raw file upload to cleaning and transformation.

```python
# 1. Upload a raw file (CSV/Excel)
file_info = client.datasets.upload_file("data/iris.csv")

# 2. Register it as a dataset
dataset = client.datasets.create(
    name="Iris Dataset",
    description="Classic iris flowers dataset",
    file_id=file_info["id"],
    rows=150,
    columns=5,
    metadata={"source": "local"}
)

# 3. List all datasets
all_datasets = client.datasets.list()

# 4. Fetch data as a Pandas DataFrame
df = client.datasets.get_data(dataset["id"], as_df=True)

# 5. Clean dataset (e.g., drop nulls)
cleaned_version = client.datasets.clean(dataset["id"], strategy="drop_nulls")
```

### Managing Models

You can train models on the server or upload pre-trained ones.

```python
# 1. Train a new model
model = client.models.train(
    dataset_id=dataset["id"],
    algorithm="random_forest",
    target_column="species",
    features=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    hyperparameters={"n_estimators": 100},
    name="Iris RF Classifier"
)

# 2. Run inference (Predict)
prediction = client.models.predict(
    model_id=model["id"],
    inputs={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
)
print(f"Predicted species: {prediction['predictions'][0]}")

# 3. Download the trained .joblib file
client.models.download(model["id"], "trained_model.joblib")
```

## Features

- **Database-like Connection**: Simple URL-based connection management.
- **Session Management**: Automatic token handling and re-authentication.
- **Dataset Lineage**: Track versions of datasets as you clean and transform them.
- **Model Versioning**: Keep track of model iterations and performance metrics.
- **Pandas Integration**: Seamlessly convert dataset samples to DataFrames.

## Development

```bash
# Clone the repository
git clone https://github.com/Amanbig/mlcore-client.git
cd mlcore-client

# Install dependencies
pip install -e .
```

## License

MIT