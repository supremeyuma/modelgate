"""ModelGate: forecast, gate and monitor the models that reach production."""

import os

# Keep the local MLflow file store usable (MLflow 3 makes it opt-in) and quiet its agent hint.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

__version__ = "1.0.0"
