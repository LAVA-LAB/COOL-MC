#!/bin/bash
# Start the MLflow UI server in the background, then start the FastAPI server
# in the foreground so the container exits if uvicorn dies.
mlflow ui \
    --backend-store-uri "file:///workspaces/coolmc/mlruns" \
    --host 0.0.0.0 \
    --port 5000 \
    &

exec uvicorn server.server:app --host 0.0.0.0 --port 8765
